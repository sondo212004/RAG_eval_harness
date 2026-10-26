from rag_eval_harness.infra.rag.chunker import ParentChildChunker

TEXT = ("Điều 1. Phạm vi điều chỉnh. " * 40 + "\n\n" + "Điều 2. Đối tượng áp dụng. " * 40) * 3


def _chunker():
    return ParentChildChunker(parent_size=3000, parent_overlap=200, child_size=500, child_overlap=50)


# AC15
def test_ingest_reproducible_chunk_ids():
    first = _chunker().split("doc-a", TEXT)
    second = _chunker().split("doc-a", TEXT)

    assert [c.id for c in first] == [c.id for c in second]
    assert len(first) > 3


def test_ids_unique_and_differ_across_documents():
    a = _chunker().split("doc-a", TEXT)
    b = _chunker().split("doc-b", TEXT)

    assert len({c.id for c in a}) == len(a)
    assert not {c.id for c in a} & {c.id for c in b}


def test_child_within_parent_and_sizes_respected():
    chunks = _chunker().split("doc-a", TEXT)

    for c in chunks:
        assert c.text in c.parent_text
        assert len(c.text) <= 500
        assert len(c.parent_text) <= 3000
