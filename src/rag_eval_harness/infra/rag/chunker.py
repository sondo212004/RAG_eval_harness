"""Chunk cha/con theo ký tự (SPEC-001 Vòng 5): tìm trên chunk con, trả chunk cha làm context."""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ...domain.corpus import chunk_id


@dataclass(frozen=True)
class Chunk:
    id: str
    parent_id: str
    doc_id: str
    text: str
    parent_text: str


class ParentChildChunker:
    def __init__(self, parent_size: int, parent_overlap: int, child_size: int, child_overlap: int):
        self.parents = RecursiveCharacterTextSplitter(chunk_size=parent_size, chunk_overlap=parent_overlap)
        self.children = RecursiveCharacterTextSplitter(chunk_size=child_size, chunk_overlap=child_overlap)

    def split(self, doc_id: str, text: str) -> list[Chunk]:
        chunks = []
        for pi, parent in enumerate(self.parents.split_text(text)):
            pid = chunk_id(doc_id, f"p{pi}", parent)
            for ci, child in enumerate(self.children.split_text(parent)):
                chunks.append(Chunk(chunk_id(doc_id, f"p{pi}c{ci}", child), pid, doc_id, child, parent))
        return chunks
