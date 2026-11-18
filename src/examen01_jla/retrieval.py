"""Indexación y recuperación léxica para documentos SciFact."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")
_STOPWORDS = frozenset(
    "a about above after again against all am an and any are as at be because been "
    "before being below between both but by can could did do does doing down during "
    "each few for from further had has have having he her here hers herself him "
    "himself his how i if in into is it its itself just me more most my myself no "
    "nor not of off on once only or other our ours ourselves out over own same she "
    "should so some such than that the their theirs them themselves then there these "
    "they this those through to too under until up very was we were what when where "
    "which while who whom why will with would you your yours yourself yourselves".split()
)
_STEMMER = PorterStemmer()


def tokenize(text: str) -> list[str]:
    """Lowercase, remove English stopwords, and apply Porter stemming."""
    return [
        _STEMMER.stem(token)
        for token in _TOKEN.findall(text.lower())
        if token not in _STOPWORDS
    ]


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    text: str

    @property
    def searchable_text(self) -> str:
        return f"{self.title} {self.text}".strip()


class SearchIndex:
    """One vector-space index with independent TF-IDF and BM25 rankings."""

    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise ValueError("El corpus no puede estar vacío")
        self.documents = documents
        self.k1 = k1
        self.b = b
        self._texts = [doc.searchable_text for doc in documents]
        self._tfidf = TfidfVectorizer(analyzer=tokenize, lowercase=False)
        self._tfidf_matrix = self._tfidf.fit_transform(self._texts)
        self._counts = CountVectorizer(analyzer=tokenize, lowercase=False)
        self._count_matrix = self._counts.fit_transform(self._texts).tocsr()
        # CSC stores each term's posting list contiguously for BM25 lookup.
        self._postings = self._count_matrix.tocsc()
        self._lengths = np.asarray(self._count_matrix.sum(axis=1)).ravel()
        self._avg_length = float(self._lengths.mean()) or 1.0
        self._doc_freq = np.asarray((self._count_matrix > 0).sum(axis=0)).ravel()
        self._n_docs = len(documents)

    @property
    def vocabulary_size(self) -> int:
        return len(self._counts.vocabulary_)

    def search(self, query: str, model: str = "bm25", top_k: int = 10) -> list[dict]:
        if model not in {"bm25", "tfidf"}:
            raise ValueError("model debe ser 'bm25' o 'tfidf'")
        if top_k < 1:
            raise ValueError("top_k debe ser mayor que cero")
        if not query.strip():
            return []

        if model == "tfidf":
            query_vector = self._tfidf.transform([query])
            scores = (self._tfidf_matrix @ query_vector.T).toarray().ravel()
        else:
            query_vector = self._counts.transform([query]).tocsr()
            scores = self._bm25_scores(query_vector)

        # Stable descending sort preserves corpus order when scores tie.
        ranked = np.argsort(-scores, kind="stable")
        results = []
        for index in ranked:
            score = float(scores[index])
            if score <= 0:
                continue
            doc = self.documents[int(index)]
            results.append({
                "rank": len(results) + 1,
                "document_id": doc.document_id,
                "score": score,
                "title": doc.title,
                "snippet": self._snippet(doc.text),
            })
            if len(results) == top_k:
                break
        return results

    def _bm25_scores(self, query_vector) -> np.ndarray:
        scores = np.zeros(self._n_docs, dtype=np.float64)
        query_terms = query_vector.indices
        for term_id in query_terms:
            postings = self._postings[:, term_id].tocoo()
            df = int(self._doc_freq[term_id])
            # Robertson-Sparck Jones IDF; it remains positive for SciFact terms.
            idf = math.log1p((self._n_docs - df + 0.5) / (df + 0.5))
            tf = postings.data.astype(np.float64)
            doc_indices = postings.row
            norm = tf + self.k1 * (
                1.0 - self.b + self.b * self._lengths[doc_indices] / self._avg_length
            )
            scores[doc_indices] += idf * tf * (self.k1 + 1.0) / norm
        return scores

    @staticmethod
    def _snippet(text: str, limit: int = 240) -> str:
        compact = " ".join(text.split())
        return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


def retrieval_metrics(ranked_ids: list[str], relevant: dict[str, int], k: int = 10) -> dict[str, float]:
    """Compute macro Precision@k, Recall@k, and graded NDCG@k for one query."""
    top = ranked_ids[:k]
    relevant_ids = {doc_id for doc_id, grade in relevant.items() if grade > 0}
    hits = sum(doc_id in relevant_ids for doc_id in top)
    precision = hits / k
    recall = hits / len(relevant_ids) if relevant_ids else 0.0

    dcg = sum(
        (2 ** relevant.get(doc_id, 0) - 1) / math.log2(rank + 1)
        for rank, doc_id in enumerate(top, start=1)
    )
    ideal_grades = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal_grades, 1))
    return {"precision@10": precision, "recall@10": recall, "ndcg@10": dcg / idcg if idcg else 0.0}


def evaluate(index: SearchIndex, queries: dict[str, str], qrels: dict[str, dict[str, int]], k: int = 10):
    """Evaluate macro metrics over queries that have relevance judgments."""
    output = {}
    for model in ("tfidf", "bm25"):
        per_query = []
        for query_id, relevant in qrels.items():
            query = queries.get(query_id)
            if query is None:
                continue
            result_ids = [r["document_id"] for r in index.search(query, model, k)]
            per_query.append(retrieval_metrics(result_ids, relevant, k))
        if not per_query:
            raise ValueError("No hay consultas evaluables con qrels")
        output[model] = {
            metric: float(np.mean([row[metric] for row in per_query]))
            for metric in per_query[0]
        }
    return output
