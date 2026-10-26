import pytest

from rag_eval_harness.domain.dataset import dataset_version, filter_reviewed, parse_dataset
from rag_eval_harness.domain.errors import AppError


def _item(**overrides):
    item = {
        "id": "legal-001",
        "question": "Nước sạch phải đạt quy chuẩn nào?",
        "ground_truth": "QCVN 01-1:2024/BYT.",
        "gold_contexts": ["Nước sạch sử dụng cho mục đích sinh hoạt phải đáp ứng QCVN 01-1:2024/BYT."],
        "domain": "legal",
        "source": "QCVN 01-1:2024/BYT, Điều 1",
        "reviewed": True,
    }
    item.update(overrides)
    return item


# AC1
def test_load_valid():
    items = parse_dataset([_item(id="a"), _item(id="b", domain="scada")])

    assert [i.id for i in items] == ["a", "b"]
    assert items[1].domain == "scada"


def test_missing_field():
    record = _item()
    del record["ground_truth"]

    with pytest.raises(AppError) as exc:
        parse_dataset([record])

    assert exc.value.code == "DATASET_INVALID"
    assert {"item": "legal-001", "field": "ground_truth", "rule": "required"} in exc.value.details


def test_unknown_domain():
    with pytest.raises(AppError) as exc:
        parse_dataset([_item(domain="finance")])

    assert exc.value.code == "DATASET_INVALID"
    assert {"item": "legal-001", "field": "domain", "rule": "enum"} in exc.value.details


def test_empty_dataset_invalid():
    with pytest.raises(AppError) as exc:
        parse_dataset([])

    assert exc.value.code == "DATASET_INVALID"


def test_duplicate_id():
    with pytest.raises(AppError) as exc:
        parse_dataset([_item(id="x"), _item(id="y"), _item(id="x")])

    assert exc.value.code == "DATASET_DUPLICATE_ID"
    assert exc.value.details == [{"item": "x", "field": "id", "rule": "unique"}]


# AC2
def test_unreviewed_excluded_and_counted():
    items = parse_dataset([_item(id="a"), _item(id="b", reviewed=False), _item(id="c", reviewed=False)])

    kept, excluded = filter_reviewed(items)

    assert [i.id for i in kept] == ["a"]
    assert excluded == 2


def test_dataset_version_is_sha256_prefix_and_stable():
    v1 = dataset_version(b'[{"id": "a"}]')

    assert v1 == dataset_version(b'[{"id": "a"}]')
    assert len(v1) == 12
    assert v1 != dataset_version(b'[{"id": "b"}]')
