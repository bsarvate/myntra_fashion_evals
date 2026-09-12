"""Bring-your-own-vector adapter; no remote actions during module import."""
import math
import os

from .catalog import FILTER_FIELDS, fingerprint


def index_namespace(catalog_hash, model_id, revision, dimension):
    # Prevent mixing data snapshots, model spaces or revisions in one namespace.
    return "v1-" + fingerprint([catalog_hash, model_id, revision, dimension])[:24]


def pinecone_filter(filters):
    if set(filters) - FILTER_FIELDS:
        raise ValueError("Unsupported metadata filter")
    return {key: {"$eq": str(value).strip().casefold()} for key, value in filters.items()}


class PineconeVectors:
    def __init__(self, name, namespace, dimension, client=None):
        if client is None:
            from pinecone import Pinecone
            client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.client, self.name, self.namespace, self.dimension = client, name, namespace, dimension
        self.index = None

    def create_if_missing(self, cloud="aws", region="us-east-1"):
        from pinecone import ServerlessSpec
        if self.name not in self.client.list_indexes().names():
            self.client.create_index(name=self.name, dimension=self.dimension, metric="cosine",
                                     spec=ServerlessSpec(cloud=cloud, region=region))
        return self.client.describe_index(self.name)

    def connect(self):
        description = self.client.describe_index(self.name)
        if description.dimension != self.dimension or description.metric != "cosine":
            raise ValueError("Index dimension/metric differs from embedding configuration")
        if not description.status["ready"]:
            raise RuntimeError("Index is not ready; rerun the status cell later")
        self.index = self.client.Index(host=description.host)

    def upsert(self, products, ids, vectors, batch_size=100):
        if self.index is None:
            raise RuntimeError("Connect first")
        catalog = {p["product_id"]: p for p in products}
        if len(ids) != len(vectors) or len(set(ids)) != len(ids):
            raise ValueError("IDs must be unique and align with vectors")
        rows = []
        for pid, vector in zip(ids, vectors):
            values = [float(v) for v in vector]
            if len(values) != self.dimension or not all(math.isfinite(v) for v in values) or not any(values):
                raise ValueError("Invalid embedding")
            p = catalog[pid]
            rows.append({"id": pid, "values": values,
                         "metadata": {key: str(p.get(key, "")).strip().casefold() for key in FILTER_FIELDS}})
        for start in range(0, len(rows), batch_size):
            self.index.upsert(vectors=rows[start:start+batch_size], namespace=self.namespace)
        return len(rows)

    def search_vector(self, vector, filters=None, k=10):
        if self.index is None:
            raise RuntimeError("Connect first")
        values = [float(v) for v in vector]
        if len(values) != self.dimension or not all(math.isfinite(v) for v in values) or not any(values):
            raise ValueError("Invalid query embedding")
        kwargs = {"namespace": self.namespace, "vector": values, "top_k": k, "include_metadata": False}
        if filters:
            kwargs["filter"] = pinecone_filter(filters)
        result = self.index.query(**kwargs)
        return [{"product_id": match.id, "score": match.score} for match in result.matches]
