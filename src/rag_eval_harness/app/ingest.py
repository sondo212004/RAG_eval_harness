"""Use case `ingest` (AC15): chunk → embed → nạp lại vector store từ đầu, id chunk tất định."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Document:
    doc_id: str
    domain: str
    text: str


class Chunker(Protocol):
    def split(self, doc_id: str, text: str) -> list: ...


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorIndex(Protocol):
    def recreate(self, dim: int) -> None: ...
    def upsert(self, points: list[dict]) -> None: ...


def ingest(
    documents: list[Document], chunker: Chunker, embedder: Embedder, index: VectorIndex, batch_size: int = 64
) -> dict:
    index.recreate(embedder.dim)  # nạp lại toàn bộ → kết quả chỉ phụ thuộc corpus + config
    used_docs, total = 0, 0
    for doc in documents:
        chunks = chunker.split(doc.doc_id, doc.text) if doc.text.strip() else []
        if not chunks:
            continue
        used_docs += 1
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = embedder.embed([c.text for c in batch])
            index.upsert(
                [
                    {
                        "id": c.id,
                        "vector": v,
                        "payload": {
                            "doc_id": doc.doc_id,
                            "domain": doc.domain,
                            "parent_id": c.parent_id,
                            "parent_text": c.parent_text,
                            "text": c.text,
                        },
                    }
                    for c, v in zip(batch, vectors, strict=True)
                ]
            )
        total += len(chunks)
    return {"documents": used_docs, "chunks": total}
