"""
c2_catalog_retriever.py  –  MedShop-RAG Component C2: Catalog Retriever
Hybrid: BM25 (rank_bm25) + TF-IDF cosine (pure NumPy, no sklearn), fused with RRF k=60.
NOTE: sentence-transformers unavailable in sandbox; TF-IDF cosine used as
      dense-retrieval substitute. Flagged as TFIDF_SUBSTITUTE_FOR_BIENCODER.
NOTE: sklearn hangs on import in sandbox; pure NumPy TF-IDF used instead.
SIMULATION_FLAG: SYNTHETIC_DATA
"""

import json, re, math
from collections import Counter
from rank_bm25 import BM25Okapi
import numpy as np

RRF_K = 60  # locked in Stage 4 engineering spec

def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())

class TFIDFVectorizer:
    """Minimal TF-IDF using pure NumPy (no sklearn dependency)."""
    def __init__(self):
        self.vocab = {}
        self.idf = None
        self.doc_vecs = None

    def fit_transform(self, docs):
        tokenized = [tokenize(d) for d in docs]
        # Build vocab
        all_words = set()
        for toks in tokenized: all_words.update(toks)
        self.vocab = {w: i for i, w in enumerate(sorted(all_words))}
        V = len(self.vocab)
        N = len(docs)
        # TF matrix
        tf = np.zeros((N, V), dtype=np.float32)
        for i, toks in enumerate(tokenized):
            counts = Counter(toks)
            total = sum(counts.values()) + 1e-9
            for w, c in counts.items():
                if w in self.vocab:
                    tf[i, self.vocab[w]] = c / total
        # IDF
        df = np.sum(tf > 0, axis=0).astype(np.float32)
        self.idf = np.log((N + 1) / (df + 1)) + 1.0
        self.doc_vecs = tf * self.idf
        # L2 normalize
        norms = np.linalg.norm(self.doc_vecs, axis=1, keepdims=True) + 1e-9
        self.doc_vecs = self.doc_vecs / norms
        return self.doc_vecs

    def transform_query(self, query: str) -> np.ndarray:
        toks = tokenize(query)
        counts = Counter(toks)
        total = sum(counts.values()) + 1e-9
        V = len(self.vocab)
        qvec = np.zeros(V, dtype=np.float32)
        for w, c in counts.items():
            if w in self.vocab:
                qvec[self.vocab[w]] = (c / total) * self.idf[self.vocab[w]]
        norm = np.linalg.norm(qvec) + 1e-9
        return qvec / norm

class CatalogRetriever:
    def __init__(self, products: list):
        self.products = products
        self.ids = [p["product_id"] for p in products]
        self.doc_texts = [
            f"{p['name']} {p['active_ingredient']} {p['indications']} {p['category']}"
            for p in products
        ]
        # BM25
        tokenized = [tokenize(d) for d in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized)
        # TF-IDF (dense substitute, pure numpy)
        self.tfidf = TFIDFVectorizer()
        self.tfidf_matrix = self.tfidf.fit_transform(self.doc_texts)

    def _bm25_rank(self, query: str):
        scores = self.bm25.get_scores(tokenize(query))
        ranked = np.argsort(scores)[::-1]
        return {self.ids[i]: rank+1 for rank, i in enumerate(ranked)}

    def _tfidf_rank(self, query: str):
        qvec = self.tfidf.transform_query(query)
        scores = self.tfidf_matrix @ qvec
        ranked = np.argsort(scores)[::-1]
        return {self.ids[i]: rank+1 for rank, i in enumerate(ranked)}

    def rrf_fuse(self, rank_bm25: dict, rank_tfidf: dict):
        pids = set(rank_bm25.keys()) | set(rank_tfidf.keys())
        scores = {}
        for pid in pids:
            r1 = rank_bm25.get(pid, len(self.products)+1)
            r2 = rank_tfidf.get(pid, len(self.products)+1)
            scores[pid] = 1.0/(RRF_K + r1) + 1.0/(RRF_K + r2)
        return dict(sorted(scores.items(), key=lambda x: -x[1]))

    def retrieve(self, query: str, top_k: int = 10):
        rb = self._bm25_rank(query)
        rt = self._tfidf_rank(query)
        fused = self.rrf_fuse(rb, rt)
        ranked_ids = list(fused.keys())[:top_k]
        pid_to_product = {p["product_id"]: p for p in self.products}
        return [{"rank": rank+1, "product_id": pid, "name": pid_to_product[pid]["name"],
                 "rrf_score": round(fused[pid], 6),
                 "bm25_rank": rb.get(pid), "tfidf_rank": rt.get(pid)}
                for rank, pid in enumerate(ranked_ids)]

    def retrieve_bm25_only(self, query: str, top_k: int = 10):
        rb = self._bm25_rank(query)
        ranked_ids = sorted(rb.keys(), key=lambda x: rb[x])[:top_k]
        pid_to_product = {p["product_id"]: p for p in self.products}
        return [{"rank": rank+1, "product_id": pid, "name": pid_to_product[pid]["name"],
                 "bm25_rank": rb[pid], "SIMULATION_FLAG":"SYNTHETIC_DATA"}
                for rank, pid in enumerate(ranked_ids)]

    def retrieve_tfidf_only(self, query: str, top_k: int = 10):
        rt = self._tfidf_rank(query)
        ranked_ids = sorted(rt.keys(), key=lambda x: rt[x])[:top_k]
        pid_to_product = {p["product_id"]: p for p in self.products}
        return [{"rank": rank+1, "product_id": pid, "name": pid_to_product[pid]["name"],
                 "tfidf_rank": rt[pid],
                 "note": "TFIDF_SUBSTITUTE_FOR_BIENCODER",
                 "SIMULATION_FLAG":"SYNTHETIC_DATA"}
                for rank, pid in enumerate(ranked_ids)]


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    with open("../data/products_catalog.json") as f:
        products = json.load(f)
    retriever = CatalogRetriever(products)
    results = retriever.retrieve("pain reliever headache", top_k=5)
    for r in results:
        print(f"  #{r['rank']} {r['product_id']} {r['name']}")
