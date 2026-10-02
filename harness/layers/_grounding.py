"""Các phép đo grounding dùng chung cho `citation_checker` và `critic`.

Cùng một cách so khớp với `arena.scorer._supports`: chuẩn hoá NFC +
casefold + gộp khoảng trắng, và một claim chỉ được một tài liệu "đỡ" khi
nó nằm gọn trong MỘT DÒNG của tài liệu đó. Không đọc `Doc.tags`.
"""

from __future__ import annotations

import re
import unicodedata

_WS_RE = re.compile(r"\s+")

#: Ngắn hơn ngưỡng này thì scorer không coi là trích dẫn (MIN_SUPPORT_CHARS).
MIN_SUPPORT_CHARS = 12


def norm(text) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


def _doc_lines(ctx, doc) -> tuple:
    cache = ctx.state.setdefault("_grounding_lines", {})
    lines = cache.get(doc.doc_id)
    if lines is None:
        lines = tuple(line for line in (norm(raw) for raw in doc.body.splitlines()) if line)
        cache[doc.doc_id] = lines
    return lines


def supports(ctx, doc, normalised_claim: str) -> bool:
    """Tài liệu có câu này như một trích dẫn nguyên văn trong một dòng không?"""
    if doc is None or len(normalised_claim) < MIN_SUPPORT_CHARS:
        return False
    return any(normalised_claim in line for line in _doc_lines(ctx, doc))


def seen_docs(ctx) -> list:
    """Tài liệu lượt chạy này đã thực sự thấy, tài liệu fetch nguyên vẹn trước.

    Một tài liệu được tính là đã thấy khi toàn văn của nó nằm trong quan
    sát (fetch sạch), hoặc mã của nó xuất hiện trong quan sát (kết quả
    search — scorer cũng tính các tài liệu đó là `retrieved`).
    """
    corpus = ctx.corpus
    if corpus is None:
        return []
    observed = ctx.observed_text
    fetched, listed = [], []
    for doc in corpus.docs:
        if doc.body and doc.body in observed:
            fetched.append(doc)
        elif doc.doc_id in observed:
            listed.append(doc)
    return fetched + listed


def observed(ctx, normalised_claim: str) -> bool:
    """Câu này có nằm nguyên văn trong bằng chứng agent đã đọc không?"""
    cache = ctx.state.get("_grounding_observed")
    if cache is None or cache[0] != len(ctx.observations):
        cache = (len(ctx.observations), norm(ctx.observed_text))
        ctx.state["_grounding_observed"] = cache
    return bool(normalised_claim) and normalised_claim in cache[1]


def source_for(ctx, text, current_doc_id=None, load=None):
    """Mã tài liệu đã thấy thực sự chứa `text`, hoặc None.

    Giữ `current_doc_id` nếu nó đã đúng; nếu không, chọn tài liệu đã thấy
    đang gánh ít claim nhất (scorer chỉ chấm tối đa 4 claim mỗi tài liệu).
    """
    n = norm(text)
    if not observed(ctx, n):
        return None
    candidates = [doc for doc in seen_docs(ctx) if supports(ctx, doc, n)]
    if not candidates:
        return None
    ids = [doc.doc_id for doc in candidates]
    if current_doc_id in ids:
        return current_doc_id
    load = load or {}
    return min(ids, key=lambda doc_id: (load.get(doc_id, 0), ids.index(doc_id)))


def citations_of(claims: list) -> list:
    return sorted({c["doc_id"] for c in claims if isinstance(c.get("doc_id"), str) and c["doc_id"]})
