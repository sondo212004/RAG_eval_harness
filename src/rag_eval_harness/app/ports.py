"""Interface mà app/ phụ thuộc — infra/ cài đặt, test dùng fake."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

Kind = Literal["eval", "bench"]


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class ChatAnswer:
    """Khớp `AskResponse` trong contracts/chat-adapter.openapi.yaml."""

    answer: str
    contexts: list[str]
    route: str | None = None
    usage: Usage | None = None


class Clock(Protocol):
    def now(self) -> datetime: ...


class IdGenerator(Protocol):
    def new(self) -> str: ...


class Timer(Protocol):
    def monotonic_ms(self) -> float: ...


class ChatAdapter(Protocol):
    """Đối tượng được đo: RAG tham chiếu hoặc chatbot ngoài qua HTTP. Lỗi tạm thời → AppError PROVIDER_ERROR."""

    def ask(self, question: str, request_id: str) -> ChatAnswer: ...


class Judge(Protocol):
    model: str

    def score(self, question: str, answer: str, contexts: list[str], ground_truth: str) -> dict[str, float]: ...


class ResultStore(Protocol):
    """name = run_id hoặc "baseline"; kind phân biệt file eval / bench."""

    def exists(self, name: str, kind: Kind) -> bool: ...
    def load(self, name: str, kind: Kind) -> dict | None: ...
    def save(self, name: str, kind: Kind, payload: dict) -> None: ...
    def delete(self, name: str, kind: Kind) -> None: ...


class Tracer(Protocol):
    def trace(self, name: str, payload: dict) -> None: ...


class Downloader(Protocol):
    def get(self, url: str) -> bytes: ...


class CorpusSink(Protocol):
    def save(self, name: str, content: bytes) -> None: ...
