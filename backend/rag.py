"""Chroma-backed retrieval over backend/docs/*.md, embedded in this process."""
import os
import re
from pathlib import Path

import chromadb

DOCS_DIR = Path(__file__).parent / "docs"
PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "/app/chroma_store")
COLLECTION_NAME = "telco_docs"

_client = None
_collection = None


def _chunk_markdown(text: str, source: str) -> list[dict]:
    """Split on ## headings; each section becomes one retrievable chunk."""
    sections = re.split(r"\n(?=## )", text)
    chunks = []
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        chunks.append({"text": section, "source": f"{source}#{i}"})
    return chunks


def _build_collection():
    global _client, _collection
    _client = chromadb.PersistentClient(path=PERSIST_DIR)
    _collection = _client.get_or_create_collection(COLLECTION_NAME)

    if _collection.count() > 0:
        return

    ids, docs, metadatas = [], [], []
    for path in sorted(DOCS_DIR.glob("*.md")):
        chunks = _chunk_markdown(path.read_text(encoding="utf-8"), path.name)
        for chunk in chunks:
            ids.append(chunk["source"])
            docs.append(chunk["text"])
            metadatas.append({"source": chunk["source"]})

    if docs:
        _collection.add(ids=ids, documents=docs, metadatas=metadatas)


def _get_collection():
    if _collection is None:
        _build_collection()
    return _collection


def retrieve(query: str, k: int = 3) -> list[dict]:
    collection = _get_collection()
    if collection.count() == 0:
        return []
    result = collection.query(query_texts=[query], n_results=min(k, collection.count()))
    hits = []
    for doc, meta in zip(result["documents"][0], result["metadatas"][0]):
        hits.append({"text": doc, "source": meta["source"]})
    return hits
