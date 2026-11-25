"""Answering questions about the RTT rules from the published guidance (retrieval-augmented generation).

Sources: the GOV.UK RTT rules suite (scraped), NHS England's recording and reporting guidance and FAQs
(PDFs in data/), and a local access policy (.docx in data/). Text is split into overlapping chunks,
embedded with a sentence-transformers model, and searched with FAISS (cosine similarity). The model
answers from the top chunks only, citing them as [S1], [S2], ...

Nothing here imports Streamlit; app.py adds caching around it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

GOVUK_URL = ("https://www.gov.uk/government/publications/right-to-start-consultant-led-treatment-within-18-weeks/"
             "referral-to-treatment-consultant-led-waiting-times-rules-suite-october-2022")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PDFS = [DATA_DIR / "Recording-and-reporting-RTT-waiting-times-guidance-v5.2-Feb25.pdf",
        DATA_DIR / "Recording-and-reporting-RTT-waiting-times-guidance-Accompanying-FAQs-v1.4-Feb25.pdf"]
DOCXS = [DATA_DIR / "South East London Access Policy.docx"]
NHSE_LINKS = {
    "Recording and reporting RTT waiting times guidance v5.2 (Feb 2025)":
        "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2025/02/"
        "Recording-and-reporting-RTT-waiting-times-guidance-v5.2-Feb25.pdf",
    "Accompanying FAQs v1.4 (Feb 2025)":
        "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2025/02/"
        "Recording-and-reporting-RTT-waiting-times-guidance-Accompanying-FAQs-v1.4-Feb25.pdf",
}

SYSTEM = ("You are an assistant helping an NHS analyst interpret RTT (Referral to Treatment) guidance. Answer using "
          "ONLY the provided sources. If the sources do not contain the answer, say so clearly. Use UK English. "
          "Be precise and practical.")


@dataclass
class Chunk:
    text: str
    source: str
    citation: str
    url: str | None = None
    page: int | None = None
    heading: str | None = None


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


def clean_text(s: str) -> str:
    s = (s or "").replace(" ", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def split_into_chunks(text: str, source: str, citation: str, url: str | None = None, heading: str | None = None,
                      page: int | None = None, max_chars: int = 1800, overlap_chars: int = 200) -> list[Chunk]:
    text = clean_text(text)
    chunks, start = [], 0
    if heading:
        citation = f"{citation} – {heading}"
    if page is not None:
        citation = f"{citation} (p{page})"
    while start < len(text):
        end = min(len(text), start + max_chars)
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(piece, source, citation, url, page, heading))
        if end == len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks


def fetch_govuk_sections(url: str = GOVUK_URL) -> dict[str, str]:
    """GOV.UK rules suite as {heading: text}. Returns {} if the page can't be fetched."""
    import requests
    from bs4 import BeautifulSoup

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except Exception:  # noqa: BLE001 - the other sources still work offline
        return {}
    soup = BeautifulSoup(resp.text, "html.parser")
    content = soup.find("div", class_=re.compile(r"govuk-body")) or soup
    sections: dict[str, list[str]] = {"GOV.UK": []}
    heading = "GOV.UK"
    for el in content.find_all(["h2", "h3", "h4", "p", "li"]):
        txt = clean_text(el.get_text(" ", strip=True))
        if not txt:
            continue
        if el.name in ("h2", "h3", "h4"):
            heading = txt
            sections.setdefault(heading, [])
        else:
            sections.setdefault(heading, []).append(txt)
    return {k: clean_text("\n".join(v)) for k, v in sections.items() if v}


def read_pdf_pages(path: Path) -> list[str]:
    from pypdf import PdfReader

    pages = []
    for page in PdfReader(str(path)).pages:
        try:
            pages.append(clean_text(page.extract_text() or ""))
        except Exception:  # noqa: BLE001
            pages.append("")
    return pages


def read_docx_text(path: Path) -> str:
    try:
        from docx import Document
        doc = Document(str(path))
    except Exception:  # noqa: BLE001
        return ""
    parts = [clean_text(p.text) for p in doc.paragraphs if clean_text(p.text)]
    for table in doc.tables:
        for row in table.rows:
            line = " | ".join(c for c in (clean_text(cell.text) for cell in row.cells) if c)
            if line:
                parts.append(line)
    return clean_text("\n".join(parts))


def build_chunks(include_govuk: bool = True, pdfs: list[Path] = PDFS, docxs: list[Path] = DOCXS) -> list[Chunk]:
    chunks: list[Chunk] = []
    if include_govuk:
        for heading, block in fetch_govuk_sections().items():
            chunks += split_into_chunks(block, "GOVUK", "GOV.UK Rules Suite (Oct 2022)", url=GOVUK_URL, heading=heading)
    for pdf in pdfs:
        if pdf.exists():
            for n, page_text in enumerate(read_pdf_pages(pdf), start=1):
                if page_text:
                    chunks += split_into_chunks(page_text, pdf.name, pdf.name, page=n)
    for docx in docxs:
        if docx.exists() and (txt := read_docx_text(docx)):
            chunks += split_into_chunks(txt, docx.name, docx.name, heading="Access policy")
    return chunks


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, batch_size=64, show_progress_bar=False,
                                            normalize_embeddings=True), dtype=np.float32)


class Index:
    """Cosine-similarity search over chunk embeddings (FAISS inner product on unit vectors)."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder):
        import faiss
        self.chunks, self.embedder = chunks, embedder
        emb = embedder.encode([c.text for c in chunks]).astype(np.float32)
        faiss.normalize_L2(emb)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        import faiss
        q = self.embedder.encode([query]).astype(np.float32)
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, k)
        return [(self.chunks[i], float(s)) for i, s in zip(idxs[0], scores[0]) if 0 <= i < len(self.chunks)]


def cache_key(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.citation.encode())
        h.update(c.text.encode())
    return h.hexdigest()[:16]


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.write_text(json.dumps([asdict(c) for c in chunks], ensure_ascii=False))


def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk(**d) for d in json.loads(path.read_text())]


def build_prompt(question: str, retrieved: list[tuple[Chunk, float]]) -> str:
    context = "\n\n".join(f"[S{n}] {c.citation}\n{c.text}" for n, (c, _) in enumerate(retrieved, start=1))
    return (f"Question:\n{question}\n\nSources context:\n{context}\n\n"
            "Answer in a structured way. Include short bullet points where helpful. "
            "Cite sources by referencing [S1], [S2], etc inline.")
