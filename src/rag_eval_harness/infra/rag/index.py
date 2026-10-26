"""Vector store Qdrant (local mode, cosine), embedder bge-m3, reranker FlashRank — SPEC-001 Vòng 5–6."""

import uuid

from qdrant_client import QdrantClient, models

from .pipeline import Hit


class QdrantIndex:
    def __init__(self, client: QdrantClient, collection: str, embedder):
        self.client = client
        self.collection = collection
        self.embedder = embedder

    def recreate(self, dim: int) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
        )

    def upsert(self, points: list[dict]) -> None:
        self.client.upsert(
            self.collection,
            points=[
                models.PointStruct(id=str(uuid.UUID(hex=p["id"])), vector=p["vector"], payload=p["payload"])
                for p in points
            ],
        )

    def search(self, query: str, limit: int) -> list[Hit]:
        vector = self.embedder.embed([query])[0]
        res = self.client.query_points(self.collection, query=vector, limit=limit, with_payload=True)
        return [Hit(p.score, p.payload["parent_id"], p.payload["parent_text"]) for p in res.points]


class SentenceTransformerEmbedder:
    """Tải model lần đầu dùng (~2GB với bge-m3) — import chậm nên để lazy."""

    def __init__(self, model_name: str, dim: int = 1024):
        self.model_name = model_name
        self.dim = dim
        self._model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model.encode(texts, normalize_embeddings=True).tolist()


class FlashRankReranker:
    def __init__(self, model_name: str, cache_dir: str):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._ranker = None

    def rerank(self, query: str, passages: list[str]) -> list[str]:
        from flashrank import Ranker, RerankRequest

        if self._ranker is None:
            self._ranker = Ranker(model_name=self.model_name, cache_dir=self.cache_dir)
        ranked = self._ranker.rerank(
            RerankRequest(query=query, passages=[{"id": i, "text": p} for i, p in enumerate(passages)])
        )
        return [passages[r["id"]] for r in ranked]
