import math
from types import SimpleNamespace

import pytest

from rag_eval_harness.domain.errors import AppError
from rag_eval_harness.infra.judge import RagasJudge


class FakeMetric:
    def __init__(self, value):
        self.value = value
        self.kwargs = None

    def score(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(value=self.value)


def test_maps_inputs_to_each_metric():
    f, a, c = FakeMetric(0.9), FakeMetric(0.8), FakeMetric(0.7)
    judge = RagasJudge("judge-model", {"faithfulness": f, "answer_relevancy": a, "context_recall": c})

    scores = judge.score("Q", "A", ["ctx"], "GT")

    assert scores == {"faithfulness": 0.9, "answer_relevancy": 0.8, "context_recall": 0.7}
    assert f.kwargs == {"user_input": "Q", "response": "A", "retrieved_contexts": ["ctx"]}
    assert a.kwargs == {"user_input": "Q", "response": "A"}
    assert c.kwargs == {"user_input": "Q", "retrieved_contexts": ["ctx"], "reference": "GT"}
    assert judge.model == "judge-model"


def test_nan_score_is_provider_error_so_it_gets_retried():
    judge = RagasJudge("m", {"faithfulness": FakeMetric(math.nan)})

    with pytest.raises(AppError) as exc:
        judge.score("Q", "A", ["ctx"], "GT")

    assert exc.value.code == "PROVIDER_ERROR"


def test_empty_contexts_score_zero_without_calling_context_metrics():
    f, c = FakeMetric(0.9), FakeMetric(0.7)
    judge = RagasJudge("m", {"faithfulness": f, "context_recall": c})

    assert judge.score("Q", "A", [], "GT") == {"faithfulness": 0.0, "context_recall": 0.0}
    assert f.kwargs is None and c.kwargs is None
