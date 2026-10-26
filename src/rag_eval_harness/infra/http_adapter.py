"""Adapter đo chatbot bên ngoài theo contracts/chat-adapter.openapi.yaml."""

from typing import Literal

import httpx
from pydantic import BaseModel, ValidationError

from ..app.ports import ChatAnswer, Usage
from ..domain.errors import AppError

PATH = "/api/v1/ask"


class _Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class _AskResponse(BaseModel):
    answer: str
    contexts: list[str]
    route: Literal["rag_retrieval", "direct_answer"] | None = None
    usage: _Usage | None = None


class HttpChatAdapter:
    def __init__(self, client: httpx.Client):
        self.client = client

    def ask(self, question: str, request_id: str) -> ChatAnswer:
        try:
            resp = self.client.post(PATH, json={"question": question}, headers={"X-Request-ID": request_id})
        except httpx.TransportError as e:  # gồm timeout, mất kết nối
            raise AppError("PROVIDER_ERROR", f"{type(e).__name__} khi gọi {PATH}") from e

        if resp.status_code >= 500 or resp.status_code == 429:
            raise AppError("PROVIDER_ERROR", f"HTTP {resp.status_code} từ chatbot")
        if resp.status_code >= 400:
            raise AppError("PROVIDER_REJECTED", f"HTTP {resp.status_code} từ chatbot — kiểm tra contract/config")

        try:
            body = _AskResponse.model_validate(resp.json())
        except (ValidationError, ValueError) as e:
            raise AppError("CONTRACT_VIOLATION", "Response không khớp AskResponse") from e

        usage = Usage(body.usage.input_tokens, body.usage.output_tokens) if body.usage else None
        return ChatAnswer(answer=body.answer, contexts=body.contexts, route=body.route, usage=usage)


def build_http_adapter(base_url: str, timeout_s: float) -> HttpChatAdapter:
    return HttpChatAdapter(httpx.Client(base_url=base_url, timeout=timeout_s))
