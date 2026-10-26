from types import SimpleNamespace

import httpx
import openai
import pytest

from rag_eval_harness.domain.errors import AppError
from rag_eval_harness.infra.llm import OpenAICompatLLM, build_llm


# AC13
@pytest.mark.parametrize(
    ("provider", "env", "base_url"),
    [
        ("gemini", {"GEMINI_API_KEY": "k"}, "https://generativelanguage.googleapis.com/v1beta/openai/"),
        ("openai", {"OPENAI_API_KEY": "k"}, "https://api.openai.com/v1/"),
        ("ollama", {}, "http://localhost:11434/v1/"),
        ("ollama", {"OLLAMA_BASE_URL": "http://gpu:11434/v1"}, "http://gpu:11434/v1/"),
    ],
)
def test_selects_adapter(provider, env, base_url):
    llm = build_llm(provider, "m", env)

    assert isinstance(llm, OpenAICompatLLM)
    assert llm.provider == provider
    assert str(llm.client.base_url) == base_url


def test_unknown_provider_config_invalid():
    with pytest.raises(AppError) as exc:
        build_llm("claude", "m", {})

    assert exc.value.code == "CONFIG_INVALID"


def test_missing_key_config_missing():
    with pytest.raises(AppError) as exc:
        build_llm("gemini", "m", {})

    assert exc.value.code == "CONFIG_MISSING"
    assert exc.value.details == [{"field": "GEMINI_API_KEY", "rule": "required"}]


class _FakeCompletions:
    def __init__(self, result):
        self.result = result
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _llm(result):
    completions = _FakeCompletions(result)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions), base_url="x")
    return OpenAICompatLLM("gemini", "gemini-2.5-flash", client, temperature=0), completions


def test_complete_returns_text_and_usage():
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="xin chào"))],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3),
    )
    llm, completions = _llm(resp)

    out = llm.complete("sys", "user")

    assert (out.text, out.usage.input_tokens, out.usage.output_tokens) == ("xin chào", 12, 3)
    assert completions.kwargs["temperature"] == 0
    assert completions.kwargs["messages"][0] == {"role": "system", "content": "sys"}


_REQ = httpx.Request("POST", "https://x")


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (openai.APITimeoutError(request=_REQ), "PROVIDER_ERROR"),
        (openai.InternalServerError("boom", response=httpx.Response(503, request=_REQ), body=None), "PROVIDER_ERROR"),
        (openai.RateLimitError("slow", response=httpx.Response(429, request=_REQ), body=None), "PROVIDER_ERROR"),
        (
            openai.AuthenticationError("bad key", response=httpx.Response(401, request=_REQ), body=None),
            "PROVIDER_REJECTED",
        ),
    ],
)
def test_errors_mapped_retryable_or_not(error, code):
    llm, _ = _llm(error)

    with pytest.raises(AppError) as exc:
        llm.complete("sys", "user")

    assert exc.value.code == code
