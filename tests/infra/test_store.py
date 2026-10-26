import json

from rag_eval_harness.infra.store import FileResultStore


def test_paths_follow_spec_layout(tmp_path):
    store = FileResultStore(tmp_path)

    store.save("r1", "eval", {"a": "điểm"})
    store.save("r1", "bench", {"b": 1})

    assert json.loads((tmp_path / "r1.json").read_text(encoding="utf-8")) == {"a": "điểm"}
    assert (tmp_path / "r1.bench.json").exists()


def test_roundtrip_exists_delete(tmp_path):
    store = FileResultStore(tmp_path / "results")

    assert store.load("baseline", "eval") is None
    store.save("baseline", "eval", {"x": 1})
    assert store.exists("baseline", "eval")
    assert store.load("baseline", "eval") == {"x": 1}

    store.delete("baseline", "eval")
    store.delete("baseline", "eval")  # xoá 2 lần không lỗi
    assert not store.exists("baseline", "eval")


def test_output_is_stable_for_git_diff(tmp_path):
    store = FileResultStore(tmp_path)

    store.save("r1", "eval", {"b": 1, "a": 2})

    assert (tmp_path / "r1.json").read_bytes() == b'{\n  "a": 2,\n  "b": 1\n}\n'


def test_latest_run_id_ignores_baseline_and_bench(tmp_path):
    store = FileResultStore(tmp_path)
    for name in ("20261025-0900-aaaa", "20261026-0900-bbbb", "baseline"):
        store.save(name, "eval", {})
    store.save("20261027-0900-cccc", "bench", {})

    assert store.latest_run_id() == "20261026-0900-bbbb"
