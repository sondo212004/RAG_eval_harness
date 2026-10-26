from rag_eval_harness.app.ingest import Document, ingest
from rag_eval_harness.infra.rag.chunker import ParentChildChunker
from tests.fakes import FakeEmbedder, FakeIndex

TEXT = "Điều 1. Nước sạch phải đạt QCVN. " * 300


def _chunker():
    return ParentChildChunker(parent_size=3000, parent_overlap=200, child_size=500, child_overlap=50)


def _docs():
    return [Document("qcvn-01.html", "legal", TEXT), Document("scada.pdf", "scada", TEXT[:900])]


# AC15
def test_ingest_reproducible_chunk_ids():
    first, second = FakeIndex(), FakeIndex()

    ingest(_docs(), _chunker(), FakeEmbedder(), first)
    ingest(_docs(), _chunker(), FakeEmbedder(), second)

    assert first.ids() == second.ids()
    assert first.recreated == 1


def test_payload_keeps_domain_and_parent_for_retrieval():
    index = FakeIndex()

    stats = ingest(_docs(), _chunker(), FakeEmbedder(), index, batch_size=4)

    point = index.points[0]
    assert point["payload"]["domain"] == "legal"
    assert point["payload"]["doc_id"] == "qcvn-01.html"
    assert point["payload"]["text"] in point["payload"]["parent_text"]
    assert stats == {"documents": 2, "chunks": len(index.points)}
    assert max(index.batch_sizes) <= 4


def test_empty_documents_skipped():
    index = FakeIndex()

    stats = ingest([Document("blank.pdf", "om", "   ")], _chunker(), FakeEmbedder(), index)

    assert stats == {"documents": 0, "chunks": 0}
