"""Router LLM (SPEC-001 Vòng 8): chọn rag_retrieval | direct_answer."""

import json
import re

from ...domain.routing import ROUTES

FALLBACK_ROUTE = "rag_retrieval"  # không đọc được ý router → tra tài liệu cho chắc

ROUTER_SYSTEM = """Bạn là bộ định tuyến của chatbot ngành nước (luật/quy chuẩn, vận hành thiết bị, SCADA).
Chọn đúng một route:
- "rag_retrieval": câu hỏi cần tra cứu tài liệu chuyên ngành (điều khoản, chỉ tiêu, thông số, quy trình).
- "direct_answer": chào hỏi, kiến thức phổ thông, tính toán đơn giản, không cần tài liệu.
Chỉ trả JSON: {"route": "<rag_retrieval|direct_answer>"}"""

_JSON = re.compile(r"\{.*?\}", re.DOTALL)


def parse_route(text: str) -> str:
    match = _JSON.search(text)
    if not match:
        return FALLBACK_ROUTE
    try:
        route = json.loads(match.group()).get("route")
    except (json.JSONDecodeError, AttributeError):
        return FALLBACK_ROUTE
    return route if route in ROUTES else FALLBACK_ROUTE
