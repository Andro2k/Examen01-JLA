"""Carga reproducible del corpus SciFact y sus consultas/qrels test."""

import gzip
import json
from pathlib import Path

from examen01_jla.retrieval import Document

CORPUS_PATH = Path(__file__).resolve().parents[2] / "data" / "scifact_docs.jsonl.gz"


def load_scifact(split: str = "test"):
    """Return (documents, queries, qrels) from ir_datasets' BEIR SciFact."""
    import ir_datasets

    dataset = ir_datasets.load(f"beir/scifact/{split}")
    documents = [
        Document(str(doc.doc_id), doc.title or "", doc.text or "")
        for doc in dataset.docs_iter()
    ]
    queries = {str(query.query_id): query.text for query in dataset.queries_iter()}
    qrels: dict[str, dict[str, int]] = {}
    for qrel in dataset.qrels_iter():
        qrels.setdefault(str(qrel.query_id), {})[str(qrel.doc_id)] = int(qrel.relevance)
    return documents, queries, qrels


def export_documents(path: Path = CORPUS_PATH) -> int:
    """Save the SciFact documents so the API can start without downloading anything."""
    documents, _, _ = load_scifact("test")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as file:
        for doc in documents:
            row = {"id": doc.document_id, "title": doc.title, "text": doc.text}
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(documents)


def load_documents(path: Path = CORPUS_PATH) -> list[Document]:
    """Load the bundled corpus (used by the API; no ir_datasets needed)."""
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Genéralo con: uv run python -m examen01_jla.data"
        )
    with gzip.open(path, "rt", encoding="utf-8") as file:
        rows = (json.loads(line) for line in file)
        return [Document(row["id"], row["title"], row["text"]) for row in rows]


if __name__ == "__main__":
    print(f"{export_documents():,} documentos exportados a {CORPUS_PATH}")
