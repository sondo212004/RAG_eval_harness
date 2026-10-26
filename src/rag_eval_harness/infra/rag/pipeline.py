"""RAG tham chiếu — đối tượng đo mặc định, cài đặt ChatAdapter.

router LLM → (rag) search chunk con fetch_k → lọc score ≥ ngưỡng → gộp theo chunk cha
→ rerank → top_k → sinh câu trả lời.
"""

from dataclasses import dataclass
from typing import Protocol

from ...app.ports import ChatAnswer, Usage
from ..llm import LLMResult
from .router import ROUTER_SYSTEM, parse_route

RAG_SYSTEM = """Trả lời bằng tiếng Việt, CHỈ dựa trên các đoạn tài liệu được cung cấp.
Nếu tài liệu không đủ thông tin, nói rõ là không tìm thấy. Nêu tên văn bản/điều khoản khi có."""
DIRECT_SYSTEM = "Trả lời ngắn gọn bằng tiếng Việt."


@dataclass(frozen=True)
class Hit:
    score: float
    parent_id: str
    parent_text: str


class LLM(Protocol):
    def complete(self, system: str, user: str) -> LLMResult: ...


class Retriever(Protocol):
    def search(self, query: str, limit: int) -> list[Hit]: ...


class Reranker(Protocol):
    def rerank(self, query: str, passages: list[str]) -> list[str]: ...


def _sum_usage(*results: LLMResult) -> Usage | None:
    usages = [r.usage for r in results if r.usage]
    if not usages:
        return None
    return Usage(sum(u.input_tokens for u in usages), sum(u.output_tokens for u in usages))


class ReferenceRAG:
    def __init__(
        self, llm: LLM, retriever: Retriever, reranker: Reranker, *, fetch_k: int, score_threshold: float, top_k: int
    ):
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.fetch_k = fetch_k
        self.score_threshold = score_threshold
        self.top_k = top_k

    def _contexts(self, question: str) -> list[str]:
        parents: dict[str, str] = {}
        for hit in self.retriever.search(question, self.fetch_k):
            if hit.score >= self.score_threshold:
                parents.setdefault(hit.parent_id, hit.parent_text)
        if not parents:
            return []
        return self.reranker.rerank(question, list(parents.values()))[: self.top_k]

    def ask(self, question: str, request_id: str) -> ChatAnswer:
        routed = self.llm.complete(ROUTER_SYSTEM, question)
        route = parse_route(routed.text)

        if route == "direct_answer":
            contexts: list[str] = []
            reply = self.llm.complete(DIRECT_SYSTEM, question)
        else:
            contexts = self._contexts(question)
            docs = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, start=1)) or "(không có tài liệu phù hợp)"
            reply = self.llm.complete(RAG_SYSTEM, f"Tài liệu:\n{docs}\n\nCâu hỏi: {question}")

        return ChatAnswer(answer=reply.text, contexts=contexts, route=route, usage=_sum_usage(routed, reply))
