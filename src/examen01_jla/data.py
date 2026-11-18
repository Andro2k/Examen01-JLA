"""Carga reproducible del corpus SciFact y sus consultas/qrels test."""

from examen01_jla.retrieval import Document


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
