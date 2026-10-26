"""Đọc config/harness.yaml → Settings đã validate (lỗi → CONFIG_INVALID)."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from ..domain.errors import AppError


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Models(_Strict):
    generator: str
    judge: str


class Pricing(_Strict):
    input_per_1m: float
    output_per_1m: float


class Embedding(_Strict):
    model: str
    dim: int


class Chunking(_Strict):
    parent_size: int
    parent_overlap: int
    child_size: int
    child_overlap: int


class Retrieval(_Strict):
    fetch_k: int
    score_threshold: float
    top_k: int
    reranker: str


class HttpAdapter(_Strict):
    base_url: str = ""
    timeout_s: float = 60.0


class Paths(_Strict):
    qa_dataset: Path
    routing_dataset: Path
    manifest: Path
    corpus_dir: Path
    results_dir: Path
    qdrant_dir: Path
    cache_dir: Path


class Settings(_Strict):
    provider: Literal["gemini", "openai", "ollama"]
    models: Models
    temperature: float
    pricing: Pricing
    embedding: Embedding
    chunking: Chunking
    retrieval: Retrieval
    adapter: Literal["reference", "http"]
    http: HttpAdapter
    collection: str
    paths: Paths


def load_settings(path: Path) -> Settings:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise AppError("CONFIG_INVALID", f"Không đọc được config {path}") from e
    try:
        return Settings.model_validate(raw)
    except ValidationError as e:
        details = [{"field": ".".join(map(str, err["loc"])), "rule": err["type"]} for err in e.errors()]
        raise AppError("CONFIG_INVALID", f"Config {path} không hợp lệ", details) from e
