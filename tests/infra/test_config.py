from pathlib import Path

import pytest

from rag_eval_harness.domain.errors import AppError
from rag_eval_harness.infra.config import load_settings

ROOT = Path(__file__).resolve().parents[2]


def test_committed_config_matches_spec():
    s = load_settings(ROOT / "config" / "harness.yaml")

    assert s.provider == "gemini"
    assert (s.models.generator, s.models.judge) == ("gemini-2.5-flash", "gemini-2.5-flash")
    assert s.temperature == 0
    assert (s.pricing.input_per_1m, s.pricing.output_per_1m) == (0.30, 2.50)
    assert (s.chunking.parent_size, s.chunking.parent_overlap) == (3000, 200)
    assert (s.chunking.child_size, s.chunking.child_overlap) == (500, 50)
    assert (s.retrieval.fetch_k, s.retrieval.score_threshold, s.retrieval.top_k) == (30, 0.5, 5)
    assert s.embedding.model == "BAAI/bge-m3"
    assert s.retrieval.reranker == "ms-marco-TinyBERT-L-2-v2"


def test_unknown_provider_config_invalid(tmp_path):
    bad = tmp_path / "h.yaml"
    text = (ROOT / "config" / "harness.yaml").read_text("utf-8").replace("provider: gemini", "provider: x")
    bad.write_text(text, encoding="utf-8")

    with pytest.raises(AppError) as exc:
        load_settings(bad)

    assert exc.value.code == "CONFIG_INVALID"
    assert exc.value.details[0]["field"] == "provider"


def test_missing_file_config_invalid(tmp_path):
    with pytest.raises(AppError) as exc:
        load_settings(tmp_path / "nope.yaml")

    assert exc.value.code == "CONFIG_INVALID"
