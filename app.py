import os
import re
import json
import time
import pickle
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import requests
import streamlit as st
from bs4 import BeautifulSoup

from pypdf import PdfReader

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


# =========================
# CONFIG
# =========================
GOVUK_URL = "https://www.gov.uk/government/publications/right-to-start-consultant-led-treatment-within-18-weeks/referral-to-treatment-consultant-led-waiting-times-rules-suite-october-2022"

# Put your PDFs in the repo under ./data/ with these filenames (or adjust paths below)
DEFAULT_PDF_PATHS = [
    "data/Recording-and-reporting-RTT-waiting-times-guidance-v5.2-Feb25.pdf",
    "data/Recording-and-reporting-RTT-waiting-times-guidance-Accompanying-FAQs-v1.4-Feb25.pdf",
]

# Local cache folder (Community Cloud filesystem is ephemeral but fine for runtime)
CACHE_DIR = ".cache_rtt_bot"
INDEX_DIR = os.path.join(CACHE_DIR, "index")
os.makedirs(INDEX_DIR, exist_ok=True)

# Embedding model (small + decent)
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# If you provide OPENAI_API_KEY in Streamlit secrets, we’ll use it for generative answers
# Otherwise the app falls back to extractive answers (top passages + minimal synthesis)
DEFAULT_LLM_MODEL = "gpt-4o-mini"


# =========================
# DATA STRUCTURES
# =========================
@dataclass
class Chunk:
    text: str
    source: str                 # e.g. "GOVUK" or filename
    citation: str               # e.g. "GOV.UK – Clock starts – Rule 1" or "PDF – p12 – Section ..."
    url: Optional[str] = None
    page: Optional[int] = None
    heading: Optional[str] = None


# =========================
# HELPERS
# =========================
def _clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _hash_sources(url: str, pdf_paths: List[str]) -> str:
    h = hashlib.sha256()
    h.update(url.encode("utf-8"))
    for p in pdf_paths:
        h.update(p.encode("utf-8"))
        try:
            stat = os.stat(p)
            h.update(str(stat.st_mtime).encode("utf-8"))
            h.update(str(stat.st_size).encode("utf-8"))
        except FileNotFoundError:
            h.update(b"missing")
    return h.hexdigest()[:16]


def _split_into_chunks(text: str, source: str, base_citation: str, url: Optional[str] = None,
                      heading: Optional[str] = None, page: Optional[int] = None,
                      max_chars: int = 1800, overlap_chars: int = 250) -> List[Chunk]:
    """
    Simple character-based chunking with overlap.
    Good enough for policy text where headings matter; we pass headings in metadata.
    """
    text = _clean_text(text)
    if not text:
        return []

    chunks: List[Chunk] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk_text = text[start:end]
        chunk_text = chunk_text.strip()

        if chunk_text:
            citation = base_citation
            if page is not None:
                citation = f"{base_citation} (p{page})"
            if heading:
                citation = f"{base_citation} – {heading}" + (f" (p{page})" if page is not None else "")

            chunks.append(
                Chunk(
                    text=chunk_text,
                    source=source,
                    citation=citation,
                    url=url,
                    page=page,
                    heading=heading
                )
            )

        if end == len(text):
            break
        start = max(0, end - overlap_chars)

    return chunks


# =========================
# LOADERS
# =========================
@st.cache_data(show_spinner=False)
def fetch_govuk_article(url: str) -> Dict[str, str]:
    """
    Fetch GOV.UK page and extract main article text grouped by headings.
    Returns a dict: {heading: text_block}
    """
    r = requests.get(url, timeout=30, headers={"User-Agent": "RTT-Policy-Bot/1.0"})
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # GOV.UK main content often sits here
    main = soup.find("main")
    if main is None:
        main = soup

    # Extract headings and paragraphs in order
    headings = []
    current_heading = "Intro"
    sections: Dict[str, List[str]] = {current_heading: []}

    for el in main.find_all(["h2", "h3", "h4", "p", "li"]):
        if el.name in ["h2", "h3", "h4"]:
            current_heading = _clean_text(el.get_text(" "))
            if current_heading not in sections:
                sections[current_heading] = []
            headings.append(current_heading)
        else:
            txt = _clean_text(el.get_text(" "))
            if txt:
                sections.setdefault(current_heading, []).append(txt)

    # Join section text
    out = {}
    for h, lines in sections.items():
        block = "\n".join(lines).strip()
        if block:
            out[h] = block

    return out


@st.cache_data(show_spinner=False)
def read_pdf_pages(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        pages.append(_clean_text(txt))
    return pages


def build_chunks(url: str, pdf_paths: List[str]) -> List[Chunk]:
    chunks: List[Chunk] = []

    # GOV.UK sections by heading
    sections = fetch_govuk_article(url)
    for heading, block in sections.items():
        base_cite = "GOV.UK Rules Suite (Oct 2022)"
        chs = _split_into_chunks(
            text=block,
            source="GOVUK",
            base_citation=base_cite,
            url=url,
            heading=heading,
            page=None
        )
        chunks.extend(chs)

    # PDFs by page
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            continue

        filename = os.path.basename(pdf_path)
        pages = read_pdf_pages(pdf_path)

        for page_num, page_text in enumerate(pages, start=1):
            if not page_text:
                continue

            base_cite = f"{filename}"
            # Keep page as a “heading-like” anchor
            chs = _split_into_chunks(
                text=page_text,
                source=filename,
                base_citation=base_cite,
                url=None,
                heading=None,
                page=page_num,
                max_chars=1800,
                overlap_chars=200
            )
            chunks.extend(chs)

    return chunks


# =========================
# INDEX (EMBED + FAISS)
# =========================
@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str = EMBED_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def _index_paths(source_hash: str) -> Tuple[str, str]:
    idx_path = os.path.join(INDEX_DIR, f"{source_hash}.faiss")
    meta_path = os.path.join(INDEX_DIR, f"{source_hash}.meta.pkl")
    return idx_path, meta_path


def build_or_load_index(url: str, pdf_paths: List[str]) -> Tuple[faiss.Index, List[Chunk], str]:
    source_hash = _hash_sources(url, pdf_paths)
    idx_path, meta_path = _index_paths(source_hash)

    embedder = load_embedder()

    if os.path.exists(idx_path) and os.path.exists(meta_path):
        index = faiss.read_index(idx_path)
        with open(meta_path, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks, source_hash

    # Build fresh
    chunks = build_chunks(url, pdf_paths)
    if not chunks:
        raise RuntimeError("No chunks found. Check the URL and PDF paths.")

    texts = [c.text for c in chunks]
    embs = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine if vectors are normalised
    index.add(embs.astype(np.float32))

    faiss.write_index(index, idx_path)
    with open(meta_path, "wb") as f:
        pickle.dump(chunks, f)

    return index, chunks, source_hash


def retrieve(index: faiss.Index, chunks: List[Chunk], query: str, k: int = 8) -> List[Tuple[Chunk, float]]:
    embedder = load_embedder()
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    scores, ids = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx < 0:
            continue
        results.append((chunks[idx], float(score)))
    return results


# =========================
# LLM CALL (OPTIONAL)
# =========================
def _openai_client():
    """
    Lazy import so requirements don’t break if you remove openai.
    """
    try:
        from openai import OpenAI
        return OpenAI()
    except Exception:
        return None


def call_llm_answer(question: str, evidence: List[Tuple[Chunk, float]], model: str) -> str:
    """
    Generate a grounded answer with citations.
    """
    client = _openai_client()
    if client is None:
        raise RuntimeError("OpenAI client not available. Check requirements / secrets.")

    # Build context with numbered sources
    ctx_lines = []
    for i, (ch, score) in enumerate(evidence, start=1):
        ctx_lines.append(f"[S{i}] {ch.citation}")
        if ch.url:
            ctx_lines.append(f"URL: {ch.url}")
        ctx_lines.append(ch.text)
        ctx_lines.append("")

    context = "\n".join(ctx_lines).strip()

    system = (
        "You are a policy Q&A assistant for NHS RTT (Referral to Treatment) rules and guidance.\n"
        "You MUST answer ONLY using the provided sources.\n"
        "If the sources do not contain the answer, say you cannot find it in the provided policies.\n"
        "Every key claim MUST include one or more citations in the form [S1], [S2]...\n"
        "Keep the answer concise, operational, and avoid speculation.\n"
        "If the question is ambiguous, ask a single clarifying question, unless the sources clearly resolve it.\n"
    )

    user = (
        f"Question: {question}\n\n"
        f"Sources:\n{context}\n\n"
        "Write the answer. Use bullet points if helpful. Add a short 'Sources used' list at the end.\n"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def call_llm_verify(answer: str, evidence: List[Tuple[Chunk, float]], model: str) -> Dict:
    """
    Verification pass: label claims Supported / Not Supported / Ambiguous using only evidence.
    Returns a structured dict.
    """
    client = _openai_client()
    if client is None:
        raise RuntimeError("OpenAI client not available. Check requirements / secrets.")

    ctx_lines = []
    for i, (ch, score) in enumerate(evidence, start=1):
        ctx_lines.append(f"[S{i}] {ch.citation}")
        if ch.url:
            ctx_lines.append(f"URL: {ch.url}")
        ctx_lines.append(ch.text)
        ctx_lines.append("")
    context = "\n".join(ctx_lines).strip()

    system = (
        "You are a strict verifier. Only use the provided sources.\n"
        "Task: check whether the answer's claims are supported by the sources.\n"
        "Return JSON only."
    )

    user = {
        "answer": answer,
        "sources": context,
        "instructions": {
            "extract_atomic_claims": True,
            "label_each_claim": ["SUPPORTED", "NOT_SUPPORTED", "AMBIGUOUS"],
            "cite_supporting_sources": True,
            "if_not_supported_suggest_fix": True
        }
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)},
        ],
        temperature=0.0,
    )
    txt = resp.choices[0].message.content.strip()
    # best effort JSON parse
    try:
        return json.loads(txt)
    except Exception:
        return {"parse_error": True, "raw": txt}


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="RTT Policy Chatbot", layout="wide")

st.title("RTT Policy Chatbot (GOV.UK Rules Suite + NHS England Guidance PDFs)")
st.caption("Answers are grounded in the provided sources with citations. If evidence is weak, the bot will refuse.")

with st.sidebar:
    st.header("Sources")
    st.markdown("**GOV.UK page**")
    st.write(GOVUK_URL)

    st.markdown("**PDFs** (from repo `data/`)")

    pdf_paths = []
    for p in DEFAULT_PDF_PATHS:
        ok = os.path.exists(p)
        st.write(("✅ " if ok else "❌ ") + p)
        pdf_paths.append(p)

    st.divider()
    st.header("Retrieval settings")
    k = st.slider("Top-K chunks", 4, 15, 8, 1)
    gate = st.slider("Evidence threshold (cosine similarity)", 0.20, 0.70, 0.38, 0.01)
    st.caption("If the best match is below the threshold, the bot will refuse (reduces hallucinations).")

    st.divider()
    st.header("LLM settings")
    use_llm = st.toggle("Use LLM to generate answer (needs API key)", value=True)
    llm_model = st.text_input("Model name", value=DEFAULT_LLM_MODEL)
    verifier = st.toggle("Run verifier pass (2nd call)", value=True)

    st.divider()
    st.header("Index")
    if st.button("Rebuild index"):
        # Clear cached index files by bumping hash via deleting all indices
        try:
            for fn in os.listdir(INDEX_DIR):
                os.remove(os.path.join(INDEX_DIR, fn))
            st.success("Deleted cached index files. Reload to rebuild.")
        except Exception as e:
            st.error(f"Could not clear cache: {e}")


# Build/load index (fast after first time)
with st.spinner("Loading/building index..."):
    index, chunks, source_hash = build_or_load_index(GOVUK_URL, pdf_paths)

st.success(f"Index ready. Chunks: {len(chunks):,}. Index id: {source_hash}")

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me RTT rules questions (clock starts/stops, DNAs, active monitoring, etc.). I will cite the sources I used."}
    ]

# Render messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask a question about RTT rules / recording and reporting...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve evidence
    with st.spinner("Retrieving relevant policy text..."):
        evidence = retrieve(index, chunks, prompt, k=k)

    best_score = evidence[0][1] if evidence else 0.0

    # Evidence display (always)
    with st.expander("Retrieved evidence (top matches)", expanded=False):
        st.write(f"Best similarity: **{best_score:.3f}** (threshold: {gate:.3f})")
        for i, (ch, score) in enumerate(evidence, start=1):
            st.markdown(f"**S{i} — score {score:.3f}** — {ch.citation}")
            if ch.url:
                st.markdown(f"- Source URL: {ch.url}")
            st.markdown(ch.text[:2000] + ("…" if len(ch.text) > 2000 else ""))
            st.divider()

    # Gate on evidence
    if best_score < gate:
        answer = (
            "I can’t find strong enough support for that in the provided sources.\n\n"
            "Try rephrasing (e.g. include whether it’s an RTT pathway, DNA, active monitoring, treatment start, etc.), "
            "or check the retrieved evidence section to see what *is* covered."
        )
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        # Decide whether to generate with LLM
        api_key_present = bool(st.secrets.get("OPENAI_API_KEY", "")) or bool(os.getenv("OPENAI_API_KEY", ""))
        if use_llm and api_key_present:
            with st.spinner("Generating grounded answer..."):
                try:
                    answer = call_llm_answer(prompt, evidence, llm_model)
                except Exception as e:
                    answer = (
                        f"LLM generation failed ({e}).\n\n"
                        "Showing top evidence only. You can still use the retrieved passages above."
                    )

            # Optional verifier
            verification = None
            if verifier:
                with st.spinner("Verifying answer against sources..."):
                    try:
                        verification = call_llm_verify(answer, evidence, llm_model)
                    except Exception as e:
                        verification = {"error": str(e)}

            with st.chat_message("assistant"):
                st.markdown(answer)

                if verification:
                    with st.expander("Verification report", expanded=False):
                        st.json(verification)

            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            # Exuctive fallback: no LLM key
            # Provide a cautious extractive summary by listing the top passages and a small hint
            top = evidence[:5]
            lines = [
                "I’m running in **evidence-only mode** (no LLM API key configured).",
                "Here are the most relevant passages I found — these should contain the answer, with citations:",
                ""
            ]
            for i, (ch, score) in enumerate(top, start=1):
                snippet = ch.text.strip().replace("\n", " ")
                snippet = snippet[:400] + ("…" if len(snippet) > 400 else "")
                lines.append(f"- **[S{i}] {ch.citation}** — {snippet}")

            lines.append("")
            lines.append("If you add an API key in Streamlit secrets, I can generate a full natural-language answer grounded in these sources.")

            answer = "\n".join(lines)

            with st.chat_message("assistant"):
                st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})
