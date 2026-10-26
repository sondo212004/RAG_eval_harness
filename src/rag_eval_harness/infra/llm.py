"""Client LLM chung cho gemini / openai / ollama qua endpoint OpenAI-compatible (AC13, SPEC-001 Vòng 9)."""

from collections.abc import Mapping
from dataclasses import dataclass

import openai

from ..app.ports import Usage
from ..domain.errors import AppError

# provider → (base_url mặc định, biến môi trường chứa key; None = không cần key)
PROVIDERS: dict[str, tuple[str, str | None]] = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}
TIMEOUT_S = 60.0


@dataclass(frozen=True)
class LLMResult:
    text: str
    usage: Usage | None


class OpenAICompatLLM:
    def __init__(self, provider: str, model: str, client, temperature: float = 0.0):
        self.provider = provider
        self.model = model
        self.client = client
        self.temperature = temperature

    def complete(self, system: str, user: str) -> LLMResult:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
        except (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError) as e:
            raise AppError("PROVIDER_ERROR", f"{self.provider}: {type(e).__name__}") from e
        except openai.APIStatusError as e:
            raise AppError("PROVIDER_REJECTED", f"{self.provider}: HTTP {e.status_code}") from e

        u = resp.usage
        usage = Usage(u.prompt_tokens, u.completion_tokens) if u else None
        return LLMResult(resp.choices[0].message.content or "", usage)


def build_llm(provider: str, model: str, env: Mapping[str, str], temperature: float = 0.0) -> OpenAICompatLLM:
    if provider not in PROVIDERS:
        raise AppError("CONFIG_INVALID", f"Provider không hỗ trợ: {provider}", [{"field": "provider", "rule": "enum"}])
    base_url, key_var = PROVIDERS[provider]
    if provider == "ollama":
        base_url = env.get("OLLAMA_BASE_URL", base_url)
    api_key = env.get(key_var) if key_var else "ollama"
    if not api_key:
        raise AppError("CONFIG_MISSING", f"Thiếu biến môi trường {key_var}", [{"field": key_var, "rule": "required"}])

    client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=TIMEOUT_S, max_retries=0)  # retry ở app/
    return OpenAICompatLLM(provider, model, client, temperature)
