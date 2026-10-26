"""Qdrant ở chế độ :memory: — không cần server, không tải model."""

from qdrant_client import QdrantClient

from rag_eval_harness.infra.rag.index import QdrantIndex
from tests.fakes import FakeEmbedder


def _point(i, text, parent="p", domain="legal"):
    return {
        "id": f"{i:032x}",
        "vector": FakeEmbedder().embed([text])[0],
        "payload": {"parent_id": parent, "parent_text": f"parent of {text}", "text": text, "domain": domain},
    }


def test_upsert_then_search_returns_hits_with_parent():
    index = QdrantIndex(QdrantClient(":memory:"), "corpus", FakeEmbedder())
    index.recreate(dim=4)
    index.upsert([_point(1, "nước sạch", "p1"), _point(2, "scada", "p2")])

    hits = index.search("nước sạch", limit=1)

    assert len(hits) == 1
    assert (hits[0].parent_id, hits[0].parent_text) == ("p1", "parent of nước sạch")
    assert hits[0].score > 0.99  # cùng vector, cosine ≈ 1


def test_recreate_drops_old_points():
    index = QdrantIndex(QdrantClient(":memory:"), "corpus", FakeEmbedder())
    index.recreate(dim=4)
    index.upsert([_point(1, "a")])

    index.recreate(dim=4)

    assert index.search("a", limit=5) == []
