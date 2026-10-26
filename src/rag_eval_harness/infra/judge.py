"""Judge RAGAS: faithfulness, answer_relevancy, context_recall (SPEC-001 AC3, Vòng 9)."""

import math

from ..domain.errors import AppError

NEEDS_CONTEXT = {"faithfulness", "context_recall"}


def _inputs(metric: str, question: str, answer: str, contexts: list[str], ground_truth: str) -> dict:
    if metric == "faithfulness":
        return {"user_input": question, "response": answer, "retrieved_contexts": contexts}
    if metric == "answer_relevancy":
        return {"user_input": question, "response": answer}
    if metric == "context_recall":
        return {"user_input": question, "retrieved_contexts": contexts, "reference": ground_truth}
    raise AppError("CONFIG_INVALID", f"Metric không hỗ trợ: {metric}")


class RagasJudge:
    def __init__(self, model: str, metrics: dict):
        self.model = model
        self.metrics = metrics

    def score(self, question: str, answer: str, contexts: list[str], ground_truth: str) -> dict[str, float]:
        scores = {}
        for name, metric in self.metrics.items():
            if name in NEEDS_CONTEXT and not contexts:  # không truy xuất → không có gì để bám/nhớ lại
                scores[name] = 0.0
                continue
            try:
                value = metric.score(**_inputs(name, question, answer, contexts, ground_truth)).value
            except AppError:
                raise
            except Exception as e:  # lỗi gọi judge / parse output → coi là tạm thời, để app/ retry
                raise AppError("PROVIDER_ERROR", f"judge {name}: {type(e).__name__}") from e
            if value is None or math.isnan(value):
                raise AppError("PROVIDER_ERROR", f"judge {name} trả NaN")
            scores[name] = float(value)
        return scores


def build_ragas_judge(llm, embed_model: str) -> RagasJudge:
    """llm: OpenAICompatLLM — tái dùng client openai trỏ tới provider đã chọn."""
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, ContextRecall, Faithfulness

    judge_llm = llm_factory(llm.model, provider="openai", client=llm.client, temperature=llm.temperature)
    embeddings = HuggingFaceEmbeddings(model=embed_model)
    return RagasJudge(
        llm.model,
        {
            "faithfulness": Faithfulness(llm=judge_llm),
            "answer_relevancy": AnswerRelevancy(llm=judge_llm, embeddings=embeddings),
            "context_recall": ContextRecall(llm=judge_llm),
        },
    )
