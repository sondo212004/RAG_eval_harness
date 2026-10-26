from rag_eval_harness.app.ports import Usage
from rag_eval_harness.infra.llm import LLMResult
from rag_eval_harness.infra.rag.pipeline import Hit, ReferenceRAG
from rag_eval_harness.infra.rag.router import ROUTER_SYSTEM


class FakeLLM:
    def __init__(self, route_reply):
        self.route_reply = route_reply
        self.prompts = []

    def complete(self, system, user):
        self.prompts.append((system, user))
        if system == ROUTER_SYSTEM:
            return LLMResult(self.route_reply, Usage(10, 1))
        return LLMResult("câu trả lời", Usage(100, 20))


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, query, limit):
        self.calls.append((query, limit))
        return self.hits


class ReverseReranker:
    def rerank(self, query, passages):
        return list(reversed(passages))


def _rag(llm, retriever):
    return ReferenceRAG(llm, retriever, ReverseReranker(), fetch_k=30, score_threshold=0.7, top_k=2)


def test_direct_route_skips_retrieval():
    llm, retriever = FakeLLM('{"route": "direct_answer"}'), FakeRetriever([])

    reply = _rag(llm, retriever).ask("Xin chào", "req-1")

    assert reply.route == "direct_answer"
    assert reply.contexts == []
    assert retriever.calls == []
    assert reply.usage == Usage(110, 21)  # router + sinh câu trả lời


def test_rag_route_filters_dedupes_reranks_and_cuts_top_k():
    hits = [
        Hit(0.95, "p1", "parent 1"),
        Hit(0.90, "p1", "parent 1"),  # cùng parent → gộp
        Hit(0.85, "p2", "parent 2"),
        Hit(0.80, "p3", "parent 3"),
        Hit(0.65, "p4", "parent 4"),  # dưới ngưỡng 0.7
    ]
    llm, retriever = FakeLLM('```json\n{"route": "rag_retrieval"}\n```'), FakeRetriever(hits)

    reply = _rag(llm, retriever).ask("QCVN nào?", "req-1")

    assert retriever.calls == [("QCVN nào?", 30)]
    assert reply.route == "rag_retrieval"
    assert reply.contexts == ["parent 3", "parent 2"]
    assert "parent 3" in llm.prompts[-1][1]


def test_unparseable_router_reply_falls_back_to_rag():
    reply = _rag(FakeLLM("hmm"), FakeRetriever([Hit(0.9, "p1", "parent 1")])).ask("Q", "r")

    assert reply.route == "rag_retrieval"
