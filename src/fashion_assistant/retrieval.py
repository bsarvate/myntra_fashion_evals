"""Local BM25, cosine baseline and reciprocal-rank fusion with common filters."""
from collections import Counter
import math
import re

from .catalog import matches, product_text


def tokenize(text):
    return re.findall(r"\w+", text.casefold())


class BM25:
    """BM25 with positive Robertson IDF: log(1 + (N-df+.5)/(df+.5))."""
    def __init__(self, products, k1=1.5, b=0.75):
        self.products = {p["product_id"]: p for p in products}
        self.docs = {pid: Counter(tokenize(product_text(p))) for pid, p in self.products.items()}
        self.lengths = {pid: sum(doc.values()) for pid, doc in self.docs.items()}
        self.avgdl = sum(self.lengths.values()) / max(len(self.docs), 1)
        self.df = Counter(term for doc in self.docs.values() for term in doc)
        self.k1, self.b = k1, b

    def search(self, query, filters=None, k=10):
        if k <= 0:
            raise ValueError("k must be positive")
        filters = filters or {}
        if not query.strip():
            # Filter-only browsing has no keyword relevance score.
            return [{"product_id": pid, "score": 0.0}
                    for pid in sorted(self.products)
                    if matches(self.products[pid], filters)][:k]
        rows = []
        for pid, doc in self.docs.items():
            if not matches(self.products[pid], filters):
                continue
            score = 0.0
            for term in set(tokenize(query)):
                tf = doc[term]
                if tf:
                    idf = math.log(1 + (len(self.docs) - self.df[term] + .5)/(self.df[term] + .5))
                    score += idf * tf * (self.k1+1)/(tf+self.k1*(1-self.b+self.b*self.lengths[pid]/self.avgdl))
            if score > 0:
                rows.append({"product_id": pid, "score": score})
        return sorted(rows, key=lambda r: (-r["score"], r["product_id"]))[:k]


def fuse(*rankings, k=10, constant=60):
    if k <= 0 or constant <= 0:
        raise ValueError("k and constant must be positive")
    scores = Counter()
    for ranking in rankings:
        ids = [r["product_id"] for r in ranking]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate IDs in a ranking")
        for position, pid in enumerate(ids, start=1):
            scores[pid] += 1/(constant+position)
    return [{"product_id": pid, "score": score}
            for pid, score in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]]


class LocalVectors:
    """Exact cosine baseline for debugging embeddings before Pinecone upload."""
    def __init__(self, products, ids, vectors):
        import numpy as np
        self.products = {p["product_id"]: p for p in products}
        self.ids = list(ids)
        matrix = np.asarray(vectors, dtype=float)
        if matrix.ndim != 2 or len(matrix) != len(ids) or len(set(ids)) != len(ids):
            raise ValueError("Unique IDs and a matching vector matrix required")
        if not set(ids) <= self.products.keys() or not np.isfinite(matrix).all():
            raise ValueError("Unknown ID or nonfinite vector")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("Zero vector")
        self.vectors = matrix / norms

    def search_vector(self, vector, filters=None, k=10):
        import numpy as np
        query = np.asarray(vector, dtype=float)
        if query.shape != (self.vectors.shape[1],) or not np.isfinite(query).all() or np.linalg.norm(query) == 0:
            raise ValueError("Invalid query vector or embedding dimension")
        scores = self.vectors @ (query / np.linalg.norm(query))
        rows = [{"product_id": pid, "score": float(score)} for pid, score in zip(self.ids, scores)
                if matches(self.products[pid], filters or {})]
        return sorted(rows, key=lambda r: (-r["score"], r["product_id"]))[:k]


class HybridSearch:
    """Inject local/Pinecone stores. Image paths originate from UI, never the LLM."""
    def __init__(self, bm25, text_encoder=None, text_store=None, clip=None, image_store=None):
        self.bm25, self.text_encoder, self.text_store = bm25, text_encoder, text_store
        self.clip, self.image_store = clip, image_store

    def search(self, query, filters=None, k=10, mode="bm25", image_path=None):
        filters = filters or {}
        if mode not in {"bm25", "dense", "hybrid", "image", "multimodal"}:
            raise ValueError("Unknown retrieval mode")
        if mode in {"bm25", "dense", "hybrid"} and not query.strip():
            return self.bm25.search(query, filters, k)
        rankings = []
        if mode in {"bm25", "hybrid", "multimodal"} and query.strip():
            rankings.append(self.bm25.search(query, filters, max(k, 30)))
        if mode in {"dense", "hybrid", "multimodal"} and query.strip():
            if self.text_encoder is None or self.text_store is None:
                raise ValueError("Text embeddings are not configured")
            rankings.append(self.text_store.search_vector(self.text_encoder.encode([query])[0], filters, max(k, 30)))
        if mode in {"image", "multimodal"}:
            if self.clip is None or self.image_store is None or not image_path:
                raise ValueError("An image query, CLIP and image index are required")
            rankings.append(self.image_store.search_vector(self.clip.encode_images([image_path])[0], filters, max(k, 30)))
        return fuse(*rankings, k=k) if len(rankings) > 1 else (rankings[0][:k] if rankings else [])
