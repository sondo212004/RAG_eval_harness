"""Fake cho các port của app/ — biên ngoài (LLM, file, thời gian) đều giả, domain dùng đồ thật."""

import hashlib
from datetime import UTC, datetime

from rag_eval_harness.app.ports import ChatAnswer
from rag_eval_harness.domain.errors import AppError


class FakeClock:
    def __init__(self, now=datetime(2026, 10, 26, 9, 0, tzinfo=UTC)):
        self._now = now

    def now(self):
        return self._now


class FakeIds:
    def __init__(self, prefix="req"):
        self._n = 0
        self._prefix = prefix

    def new(self):
        self._n += 1
        return f"{self._prefix}-{self._n}"


class FakeTimer:
    """Mỗi lần gọi trả mốc kế tiếp; cặp (start, end) cho ra latency."""

    def __init__(self, latencies_ms):
        self._ticks = []
        t = 0.0
        for lat in latencies_ms:
            self._ticks += [t, t + lat]
            t += lat + 1
        self._i = 0

    def monotonic_ms(self):
        v = self._ticks[self._i]
        self._i += 1
        return v


class FakeChat:
    """answers: question → ChatAnswer | list[AppError | ChatAnswer] (lần lượt theo lần gọi)."""

    def __init__(self, answers):
        self._answers = answers
        self.calls = []

    def ask(self, question, request_id):
        self.calls.append((question, request_id))
        a = self._answers[question]
        if isinstance(a, list):
            a = a.pop(0) if len(a) > 1 else a[0]
        if isinstance(a, AppError):
            raise a
        return a


def answer(text="ok", contexts=("ctx",), route=None, usage=None):
    return ChatAnswer(answer=text, contexts=list(contexts), route=route, usage=usage)


def provider_error():
    return AppError("PROVIDER_ERROR", "timeout")


class FakeJudge:
    def __init__(self, scores_by_question, model="judge-model"):
        self._scores = scores_by_question
        self.model = model

    def score(self, question, answer, contexts, ground_truth):
        return self._scores[question]


class FakeStore:
    def __init__(self, data=None):
        self.data = dict(data or {})

    def exists(self, name, kind):
        return (name, kind) in self.data

    def load(self, name, kind):
        return self.data.get((name, kind))

    def save(self, name, kind, payload):
        self.data[(name, kind)] = payload

    def delete(self, name, kind):
        self.data.pop((name, kind), None)


class FakeTracer:
    def __init__(self):
        self.events = []

    def trace(self, name, payload):
        self.events.append((name, payload))


class FakeDownloader:
    """content: url → bytes | AppError."""

    def __init__(self, content):
        self._content = content

    def get(self, url):
        c = self._content[url]
        if isinstance(c, AppError):
            raise c
        return c


class FakeSink:
    def __init__(self):
        self.saved = {}

    def save(self, name, content):
        self.saved[name] = content


class FakeEmbedder:
    dim = 4

    def embed(self, texts):
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            out.append([b / 255 for b in h[: self.dim]])
        return out


class FakeIndex:
    def __init__(self):
        self.points = []
        self.recreated = 0
        self.batch_sizes = []

    def recreate(self, dim):
        self.recreated += 1
        self.points = []

    def upsert(self, points):
        self.batch_sizes.append(len(points))
        self.points += points

    def ids(self):
        return [p["id"] for p in self.points]
