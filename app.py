import os
import re
import json
import time
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import requests
import streamlit as st
from bs4 import BeautifulSoup
from pypdf import PdfReader

# Word documents (.docx)
try:
    from docx import Document  # python-docx
except Exception:
    Document = None

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# =========================
# CONFIG / SECRETS
# =========================
st.set_page_config(page_title="RTT Chatbot", layout="wide")

try:
    key = st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    if key:
        os.environ["OPENAI_API_KEY"] = key
except Exception:
    pass

GOVUK_URL = "https://www.gov.uk/government/publications/right-to-start-consultant-led-treatment-within-18-weeks/referral-to-treatment-consultant-led-waiting-times-rules-suite-october-2022"

DEFAULT_PDF_PATHS = [
    "data/Recording-and-reporting-RTT-waiting-times-guidance-v5.2-Feb25.pdf",
    "data/Recording-and-reporting-RTT-waiting-times-guidance-Accompanying-FAQs-v1.4-Feb25.pdf",
]

# Local access policy (Word document)
DEFAULT_DOCX_PATHS = [
    "data/South East London Access Policy.docx",
    "data/South East London Access Policy.DOCX",
]

CACHE_DIR = ".cache_rtt_bot"
INDEX_DIR = os.path.join(CACHE_DIR, "index")
os.makedirs(INDEX_DIR, exist_ok=True)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gpt-5.2"


# =========================
# DATA MODELS
# =========================
@dataclass
class Chunk:
    text: str
    source: str
    citation: str
    url: Optional[str] = None
    page: Optional[int] = None
    heading: Optional[str] = None


# =========================
# UTILITIES
# =========================
def _clean_text(s: str) -> str:
    s = s or ""
    s = s.replace("\u00a0", " ")  # non-breaking spaces
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _hash_sources(url: str, pdf_paths: List[str], docx_paths: List[str]) -> str:
    h = hashlib.sha256()
    h.update(url.encode("utf-8"))
    for p in (pdf_paths + docx_paths):
        h.update(p.encode("utf-8"))
        try:
            stat = os.stat(p)
            h.update(str(stat.st_mtime).encode("utf-8"))
            h.update(str(stat.st_size).encode("utf-8"))
        except FileNotFoundError:
            h.update(b"missing")
    return h.hexdigest()[:16]


def _split_into_chunks(
    text: str,
    source: str,
    base_citation: str,
    url: Optional[str] = None,
    heading: Optional[str] = None,
    page: Optional[int] = None,
    max_chars: int = 1800,
    overlap_chars: int = 250,
) -> List[Chunk]:
    text = _clean_text(text)
    if not text:
        return []

    chunks: List[Chunk] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk_text = text[start:end].strip()

        if chunk_text:
            citation = base_citation
            if heading:
                citation = f"{citation} – {heading}"
            if page is not None:
                citation = f"{citation} (p{page})"

            chunks.append(
                Chunk(
                    text=chunk_text,
                    source=source,
                    citation=citation,
                    url=url,
                    page=page,
                    heading=heading,
                )
            )

        if end == len(text):
            break
        start = max(0, end - overlap_chars)

    return chunks


# =========================
# GOV.UK SCRAPE
# =========================
@st.cache_data(show_spinner=False)
def fetch_govuk_article(url: str) -> Dict[str, str]:
    """
    Fetches GOV.UK Rules Suite and returns a dict of {heading: text}.
    This is intentionally a light scrape (not perfect), but robust enough
    for semantic retrieval.
    """
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except Exception:
        return {"GOV.UK": ""}

    soup = BeautifulSoup(resp.text, "html.parser")

    content = soup.find("div", class_=re.compile(r"govuk-body"))
    if not content:
        content = soup

    sections: Dict[str, List[str]] = {}
    current_heading = "GOV.UK"
    sections[current_heading] = []

    for el in content.find_all(["h2", "h3", "h4", "p", "li"]):
        tag = el.name.lower()
        txt = _clean_text(el.get_text(" ", strip=True))
        if not txt:
            continue

        if tag in ("h2", "h3", "h4"):
            current_heading = txt
            sections.setdefault(current_heading, [])
        else:
            sections.setdefault(current_heading, []).append(txt)

    return {k: _clean_text("\n".join(v)) for k, v in sections.items()}


# =========================
# PDF READER
# =========================
@st.cache_data(show_spinner=False)
def read_pdf_pages(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    pages: List[str] = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        pages.append(_clean_text(txt))
    return pages


@st.cache_data(show_spinner=False)
def read_docx_text(docx_path: str) -> str:
    """Extract text from a .docx (paragraphs + tables).

    Notes:
    - Access policies are often heavy on tables, so we include them.
    - We keep this deliberately simple and robust rather than trying to
      perfectly preserve formatting.
    """
    if Document is None:
        # python-docx isn't installed. Return empty so the app still runs.
        return ""

    try:
        doc = Document(docx_path)
    except Exception:
        return ""

    parts: List[str] = []

    # Paragraphs
    for p in getattr(doc, "paragraphs", []) or []:
        t = _clean_text(getattr(p, "text", "") or "")
        if t:
            parts.append(t)

    # Tables
    for table in getattr(doc, "tables", []) or []:
        for row in getattr(table, "rows", []) or []:
            cells = []
            for cell in getattr(row, "cells", []) or []:
                cells.append(_clean_text(getattr(cell, "text", "") or ""))
            line = " | ".join([c for c in cells if c])
            line = _clean_text(line)
            if line:
                parts.append(line)

    return _clean_text("\n".join(parts))


# =========================
# CHUNK BUILDER
# =========================
def build_chunks(url: str, pdf_paths: List[str], docx_paths: List[str]) -> List[Chunk]:
    chunks: List[Chunk] = []

    sections = fetch_govuk_article(url)
    for heading, block in sections.items():
        chunks.extend(
            _split_into_chunks(
                text=block,
                source="GOVUK",
                base_citation="GOV.UK Rules Suite (Oct 2022)",
                url=url,
                heading=heading,
                page=None,
            )
        )

    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            continue

        filename = os.path.basename(pdf_path)
        pages = read_pdf_pages(pdf_path)

        for page_num, page_text in enumerate(pages, start=1):
            if not page_text:
                continue
            chunks.extend(
                _split_into_chunks(
                    text=page_text,
                    source=filename,
                    base_citation=filename,
                    url=None,
                    heading=None,
                    page=page_num,
                    max_chars=1800,
                    overlap_chars=200,
                )
            )

    # Access policy (Word document)
    for docx_path in docx_paths:
        if not os.path.exists(docx_path):
            continue

        filename = os.path.basename(docx_path)
        txt = read_docx_text(docx_path)
        if not txt:
            continue

        chunks.extend(
            _split_into_chunks(
                text=txt,
                source=filename,
                base_citation=filename,
                url=None,
                heading="Access policy",
                page=None,
                max_chars=1800,
                overlap_chars=200,
            )
        )

    return chunks


# =========================
# INDEX (EMBED + FAISS)
# =========================
@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str = EMBED_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def _index_paths(source_hash: str) -> Tuple[str, str]:
    idx_path = os.path.join(INDEX_DIR, f"{source_hash}.faiss")
    meta_path = os.path.join(INDEX_DIR, f"{source_hash}.meta.json")
    return idx_path, meta_path


def _load_chunks_meta(meta_path: str) -> List[Chunk]:
    if meta_path.endswith(".json") and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [Chunk(**item) for item in raw]

    pkl_path = meta_path.replace(".meta.json", ".meta.pkl")
    if os.path.exists(pkl_path):
        try:
            import pickle

            with open(pkl_path, "rb") as f:
                chunks = pickle.load(f)

            # migrate to JSON
            try:
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump([c.__dict__ for c in chunks], f, ensure_ascii=False, indent=2)
            except Exception:
                pass

            return chunks
        except Exception:
            return []

    return []


def _save_chunks_meta(meta_path: str, chunks: List[Chunk]) -> None:
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump([c.__dict__ for c in chunks], f, ensure_ascii=False, indent=2)


def _build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype(np.float32))
    return index


def _embed_texts(embedder: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    # SentenceTransformer returns numpy arrays; ensure float32
    emb = embedder.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    emb = np.array(emb, dtype=np.float32)
    return emb


def build_or_load_index(url: str, pdf_paths: List[str], docx_paths: List[str]) -> Tuple[faiss.Index, List[Chunk], str]:
    source_hash = _hash_sources(url, pdf_paths, docx_paths)
    idx_path, meta_path = _index_paths(source_hash)

    embedder = load_embedder()

    if os.path.exists(idx_path) and os.path.exists(meta_path):
        try:
            index = faiss.read_index(idx_path)
            chunks = _load_chunks_meta(meta_path)
            if len(chunks) > 0:
                return index, chunks, source_hash
        except Exception:
            pass  # fall through to rebuild

    # Build from scratch
    chunks = build_chunks(url, pdf_paths, docx_paths)
    texts = [c.text for c in chunks]
    embeddings = _embed_texts(embedder, texts)
    index = _build_faiss_index(embeddings)

    faiss.write_index(index, idx_path)
    _save_chunks_meta(meta_path, chunks)

    return index, chunks, source_hash


def retrieve_chunks(
    query: str,
    index: faiss.Index,
    chunks: List[Chunk],
    embedder: SentenceTransformer,
    k: int = 4,
) -> List[Tuple[Chunk, float]]:
    q_emb = _embed_texts(embedder, [query])
    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, k)
    results: List[Tuple[Chunk, float]] = []
    for i, score in zip(idxs[0], scores[0]):
        if i < 0 or i >= len(chunks):
            continue
        results.append((chunks[i], float(score)))
    return results


# =========================
# LLM / RESPONSE GENERATION
# =========================
def format_context_for_prompt(retrieved: List[Tuple[Chunk, float]]) -> str:
    parts = []
    for n, (ch, score) in enumerate(retrieved, start=1):
        header = f"[S{n}] {ch.citation}"
        parts.append(header)
        parts.append(ch.text)
        parts.append("")  # blank line
    return "\n".join(parts).strip()


def answer_from_context(
    user_question: str,
    retrieved: List[Tuple[Chunk, float]],
    use_llm: bool = True,
    llm_model: str = DEFAULT_LLM_MODEL,
) -> Tuple[str, List[str]]:
    
    sources_used = [ch.citation for ch, _ in retrieved]

    context = format_context_for_prompt(retrieved)
    if not use_llm:
        # Naive extractive fallback: return the top chunk(s)
        if not retrieved:
            return "I couldn't find anything relevant in the indexed guidance.", []
        top = retrieved[0][0]
        return f"Best match from sources:\n\n{top.text}\n\nSources:\n- " + "\n- ".join(sources_used), sources_used

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return (
            "OpenAI API key not set. Add OPENAI_API_KEY to Streamlit secrets or environment, "
            "or toggle 'Use LLM' off for extractive mode.",
            sources_used,
        )

    system = (
        "You are an assistant helping an NHS analyst interpret RTT (Referral to Treatment) guidance. "
        "Answer using ONLY the provided sources context. "
        "If the sources do not contain the answer, say so clearly. "
        "Use UK English. Be precise and practical."
    )

    user = (
        f"Question:\n{user_question}\n\n"
        f"Sources context:\n{context}\n\n"
        "Answer in a structured way. Include short bullet points where helpful. "
        "Cite sources by referencing [S1], [S2], etc inline."
    )

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": llm_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.2,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
        return answer, sources_used
    except Exception as e:
        return f"LLM call failed: {e}", sources_used


# =========================
# STREAMLIT UI
# =========================
st.title("RTT Guidance Chatbot")
st.caption("GOV.UK Rules Suite + NHSE RTT Guidance PDFs + Access Policy (.docx)")

NHSE_LINKS = {
    "Recording and reporting RTT waiting times guidance v5.2 (Feb 2025)": "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2025/02/Recording-and-reporting-RTT-waiting-times-guidance-v5.2-Feb25.pdf",
    "Accompanying FAQs v1.4 (Feb 2025)": "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2025/02/Recording-and-reporting-RTT-waiting-times-guidance-Accompanying-FAQs-v1.4-Feb25.pdf",
}

pdf_paths = []
for p in DEFAULT_PDF_PATHS:
    pdf_paths.append(p)

docx_paths = []
for p in DEFAULT_DOCX_PATHS:
    if os.path.exists(p):
        docx_paths.append(p)

with st.sidebar:
    st.header("Sources")

    st.markdown("**GOV.UK**")
    st.markdown(f"[Referral to Treatment Rules Suite]({GOVUK_URL})")

    st.markdown("**NHSE Guidance Docs**")
    for filename, url in NHSE_LINKS.items():
        st.markdown(f"[{filename}]({url})")

    st.markdown("**Access Policy (.docx)**")
    
    if not docx_paths:
        st.warning("No access policy .docx found in ./data.")
    else:
        for p in docx_paths:
            filename = os.path.basename(p)
    
            with open(p, "rb") as f:
                file_bytes = f.read()
    
            st.download_button(
                label=f"{filename}",
                data=file_bytes,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

    k = 4
    gate = 0.38
    use_llm = True
    llm_model = DEFAULT_LLM_MODEL
    verifier = True


# Build/load index
with st.spinner("Building index..."):
    index, chunks, source_hash = build_or_load_index(GOVUK_URL, pdf_paths, docx_paths)

st.success("Ready")

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Ask me a question about RTT rules/guidance. I’ll answer using the provided sources.\n\n"
                "Please be specific (e.g. 'Should trauma pathways be included in RTT?')."
            ),
        }
    ]

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
prompt = st.chat_input("Ask about RTT guidance...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    embedder = load_embedder()
    retrieved = retrieve_chunks(prompt, index=index, chunks=chunks, embedder=embedder, k=k)

    # if top score is low warn the user
    if retrieved and retrieved[0][1] < gate:
        notice = (
            f"I found some potentially relevant text, but similarity is low (top score {retrieved[0][1]:.2f}). "
            "I may be missing the specific rule you're asking about."
        )
    else:
        notice = ""

    with st.chat_message("assistant"):
        if notice:
            st.info(notice)

        answer, sources_used = answer_from_context(
            user_question=prompt,
            retrieved=retrieved,
            use_llm=use_llm,
            llm_model=llm_model,
        )
        st.markdown(answer)

        with st.expander("Sources used"):
            if not retrieved:
                st.write("No sources retrieved.")
            else:
                for i, (ch, score) in enumerate(retrieved, start=1):
                    st.markdown(f"**[S{i}] {ch.citation}**  \nSimilarity: `{score:.3f}`")
                    st.write(ch.text)
                    if ch.url:
                        st.markdown(f"Link: {ch.url}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
