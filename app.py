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
# CONFIG / SECRETS
# =========================
st.set_page_config(page_title="RTT Chatbot", layout="wide")

# Robust secrets -> env (Streamlit Cloud secrets do NOT always appear as env vars)
try:
    key = st.secrets.get("OPENAI_API_KEY", "")
    if key:
        os.environ["OPENAI_API_KEY"] = key
except Exception:
    pass

GOVUK_URL = "https://www.gov.uk/government/publications/right-to-start-consultant-led-treatment-within-18-weeks/referral-to-treatment-consultant-led-waiting-times-rules-suite-october-2022"

DEFAULT_PDF_PATHS = [
    "data/Recording-and-reporting-RTT-waiting-times-guidance-v5.2-Feb25.pdf",
    "data/Recording-and-reporting-RTT-waiting-times-guidance-Accompanying-FAQs-v1.4-Feb25.pdf",
]

CACHE_DIR = ".cache_rtt_bot"
INDEX_DIR = os.path.join(CACHE_DIR, "index")
os.makedirs(INDEX_DIR, exist_ok=True)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gpt-5.2"


# =========================
# DATA STRUCTURES
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
# HELPERS
# =========================
def _clean_text(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
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
# LOADERS
# =========================
@st.cache_data(show_spinner=False)
def fetch_govuk_article(url: str) -> Dict[str, str]:
    r = requests.get(url, timeout=30, headers={"User-Agent": "RTT-Policy-Bot/1.0"})
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    main = soup.find("main") or soup

    current_heading = "Intro"
    sections: Dict[str, List[str]] = {current_heading: []}

    for el in main.find_all(["h2", "h3", "h4", "p", "li"]):
        if el.name in ["h2", "h3", "h4"]:
            current_heading = _clean_text(el.get_text(" "))
            sections.setdefault(current_heading, [])
        else:
            txt = _clean_text(el.get_text(" "))
            if txt:
                sections.setdefault(current_heading, []).append(txt)

    out: Dict[str, str] = {}
    for h, lines in sections.items():
        block = "\n".join(lines).strip()
        if block:
            out[h] = block
    return out


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


def build_chunks(url: str, pdf_paths: List[str]) -> List[Chunk]:
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

    chunks = build_chunks(url, pdf_paths)
    if not chunks:
        raise RuntimeError("No chunks found. Check the URL and PDF paths.")

    texts = [c.text for c in chunks]
    embs = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity because vectors are normalized
    index.add(embs.astype(np.float32))

    faiss.write_index(index, idx_path)
    with open(meta_path, "wb") as f:
        pickle.dump(chunks, f)

    return index, chunks, source_hash


def retrieve(index: faiss.Index, chunks: List[Chunk], query: str, k: int = 8) -> List[Tuple[Chunk, float]]:
    embedder = load_embedder()
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    scores, ids = index.search(q_emb, k)
    out: List[Tuple[Chunk, float]] = []
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx < 0:
            continue
        out.append((chunks[idx], float(score)))
    return out


# =========================
# OPENAI (LLM + VERIFIER)
# =========================
def _openai_client():
    """
    Create an OpenAI client using Streamlit secrets or env var.
    Also surfaces a useful error in the sidebar (without leaking secrets).
    """
    try:
        from openai import OpenAI

        try:
            secret_key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            secret_key = ""

        env_key = os.getenv("OPENAI_API_KEY", "")
        api_key = secret_key or env_key

        if not api_key:
            return None

        return OpenAI(api_key=api_key)
    except Exception as e:
        # Show the reason once (helps debugging in Cloud)
        try:
            st.session_state["_openai_init_error"] = str(e)
        except Exception:
            pass
        return None


def call_llm_answer(question: str, evidence: List[Tuple[Chunk, float]], model: str) -> str:
    client = _openai_client()
    if client is None:
        raise RuntimeError("OpenAI client not available. Check requirements / secrets.")

    ctx_lines = []
    for i, (ch, _score) in enumerate(evidence, start=1):
        ctx_lines.append(f"[S{i}] {ch.citation}")
        if ch.url:
            ctx_lines.append(f"URL: {ch.url}")
        ctx_lines.append(ch.text)
        ctx_lines.append("")
    context = "\n".join(ctx_lines).strip()

    system = (
        "You are a strict policy Q&A assistant for NHS RTT (Referral to Treatment) rules and guidance.\n"
        "You MUST answer ONLY using the provided sources.\n"
        "If the sources do not contain the answer, say you cannot find it in the provided policies.\n"
        "Every key claim MUST include one or more citations in the form [S1], [S2]...\n"
        "Keep the answer concise, operational, and avoid speculation.\n"
        "If the question is ambiguous, ask a single clarifying question unless the sources clearly resolve it.\n"
    )

    user = (
        f"Question: {question}\n\n"
        f"Sources:\n{context}\n\n"
        "Write the answer. Use bullet points if helpful. Add a short 'Sources used' list at the end.\n"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def call_llm_verify(answer: str, evidence: List[Tuple[Chunk, float]], model: str) -> Dict:
    client = _openai_client()
    if client is None:
        raise RuntimeError("OpenAI client not available. Check requirements / secrets.")

    ctx_lines = []
    for i, (ch, _score) in enumerate(evidence, start=1):
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

    payload = {
        "answer": answer,
        "sources": context,
        "instructions": {
            "extract_atomic_claims": True,
            "label_each_claim": ["SUPPORTED", "NOT_SUPPORTED", "AMBIGUOUS"],
            "cite_supporting_sources": True,
            "if_not_supported_suggest_fix": True,
        },
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload)}],
        temperature=0.0,
    )
    txt = resp.choices[0].message.content.strip()

    try:
        return json.loads(txt)
    except Exception:
        return {"parse_error": True, "raw": txt}


# =========================
# UI
# =========================
st.title("RTT Guidance Bot")

def _mask(k: str) -> str:
    if not k:
        return ""
    return k[:6] + "..." + k[-4:]


with st.sidebar:
    st.header("Sources")
    st.markdown("**GOV.UK**")
    st.write(GOVUK_URL)

    st.markdown("**NHSE Guidance Docs**")
    pdf_paths = []
    for p in DEFAULT_PDF_PATHS:
        ok = os.path.exists(p)
        st.write(p)
        pdf_paths.append(p)

    st.divider()

    k = 4
    gate = 0.38
    use_llm = True
    llm_model = DEFAULT_LLM_MODEL
    verifier = True

# Build/load index
with st.spinner("Building index..."):
    index, chunks, source_hash = build_or_load_index(GOVUK_URL, pdf_paths)

st.success(f"Ready")

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me RTT questions and I will answer based on the national guidance.",
        }
    ]

# Render messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask a question about RTT...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Retrieving relevant guidance..."):
        evidence = retrieve(index, chunks, prompt, k=k)

    best_score = evidence[0][1] if evidence else 0.0

    with st.expander("Retrieved guidance (top matches)", expanded=False):
        st.write(f"Best similarity: **{best_score:.3f}** (threshold: {gate:.3f})")
        for i, (ch, score) in enumerate(evidence, start=1):
            st.markdown(f"**S{i} — score {score:.3f}** — {ch.citation}")
            if ch.url:
                st.markdown(f"- Source URL: {ch.url}")
            st.markdown(ch.text[:2000] + ("…" if len(ch.text) > 2000 else ""))
            st.divider()

    if best_score < gate:
        answer = (
            "I can’t find strong enough evidence for that in the guidance.\n\n"
            "Try rephrasing (e.g. include whether it’s an RTT pathway, DNA, active monitoring, treatment start, etc.), "
            "or check the retrieved evidence section to see what *is* covered."
        )
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        api_key_present = bool(os.getenv("OPENAI_API_KEY", "")) or bool(
            (st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else "")
        )

        if use_llm and api_key_present:
            with st.spinner("Generating answer..."):
                try:
                    answer = call_llm_answer(prompt, evidence, llm_model)
                except Exception as e:
                    answer = (
                        f"LLM failed ({e}).\n\n"
                        "Showing top evidence only. You can still use the retrieved passages above."
                    )

            verification = None
            if verifier and "LLM failed" not in answer:
                with st.spinner("Verifying answer..."):
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
            top = evidence[:5]
            lines = [
                "I’m running in **evidence-only mode** (no LLM API key configured).",
                "Here are the most relevant passages I found — these should contain the answer, with citations:",
                "",
            ]
            for i, (ch, _score) in enumerate(top, start=1):
                snippet = ch.text.strip().replace("\n", " ")
                snippet = snippet[:400] + ("…" if len(snippet) > 400 else "")
                lines.append(f"- **[S{i}] {ch.citation}** — {snippet}")

            lines.append("")
            lines.append("If you add an API key in Streamlit secrets, I can generate a full natural-language answer grounded in these sources.")
            answer = "\n".join(lines)

            with st.chat_message("assistant"):
                st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})
