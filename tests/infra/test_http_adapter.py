"""Adapter HTTP theo contracts/chat-adapter.openapi.yaml."""

import json

import httpx
import pytest

from rag_eval_harness.app.ports import Usage
from rag_eval_harness.domain.errors import AppError
from rag_eval_harness.infra.http_adapter import HttpChatAdapter


def _adapter(handler):
    client = httpx.Client(base_url="https://bot.example", transport=httpx.MockTransport(handler))
    return HttpChatAdapter(client)


def test_request_and_response_follow_contract():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["rid"] = request.headers["X-Request-ID"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "answer": "A",
                "contexts": ["c1"],
                "route": "rag_retrieval",
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        )

    reply = _adapter(handler).ask("Q?", "req-1")

    assert seen == {"path": "/api/v1/ask", "rid": "req-1", "body": {"question": "Q?"}}
    assert (reply.answer, reply.contexts, reply.route, reply.usage) == ("A", ["c1"], "rag_retrieval", Usage(10, 2))


def test_optional_fields_absent():
    reply = _adapter(lambda r: httpx.Response(200, json={"answer": "A", "contexts": []})).ask("Q", "r")

    assert (reply.route, reply.usage) == (None, None)


@pytest.mark.parametrize(
    ("handler", "code"),
    [
        (lambda r: httpx.Response(503, json={"error": {"code": "X", "message": "down"}}), "PROVIDER_ERROR"),
        (lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("slow", request=r)), "PROVIDER_ERROR"),
        (lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused", request=r)), "PROVIDER_ERROR"),
        (
            lambda r: httpx.Response(400, json={"error": {"code": "VALIDATION_ERROR", "message": "x"}}),
            "PROVIDER_REJECTED",
        ),
        (lambda r: httpx.Response(200, json={"answer": "A"}), "CONTRACT_VIOLATION"),
        (lambda r: httpx.Response(200, json={"answer": "A", "contexts": [], "route": "web"}), "CONTRACT_VIOLATION"),
    ],
)
def test_errors(handler, code):
    with pytest.raises(AppError) as exc:
        _adapter(handler).ask("Q", "r")

    assert exc.value.code == code
