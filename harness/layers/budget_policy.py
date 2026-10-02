"""LỚP `budget_policy` — bài giảng Day 16, §3 (Budgets & Control Flow).

NHIỆM VỤ: kế hoạch của mô hình dài đúng 11 lượt gọi công cụ, bất kể brief
cho ngân sách bao nhiêu — và BỐN lượt cuối là rác có chủ ý: một lần search
lặp lại, một phép tính vô nghĩa, hai lần fetch lại tài liệu đã có trong
tay. Phần việc hữu ích nằm ở ĐẦU kế hoạch, nên cắt phần đuôi không mất một
điểm grounding nào mà lấy trọn phần điểm efficiency về tool call và token.

TÍN HIỆU:

    ctx.tools.calls >= ctx.max_tool_calls - reserve

CÁCH DỪNG: thêm `FINALIZE_SENTINEL` vào bên trong MỘT CÂU tiếng Việt bình
thường và đẩy vào cuối danh sách message trong `before_model`. `MockModel`
khoá theo token; một mô hình thật thì nghe câu tiếng Việt bao quanh nó.
Viết như vậy để cùng một lớp chạy được trên cả hai đường.

SENTINEL KHÔNG PHẢI TUỲ CHỌN — và không chỉ vì chuyện dừng.
`arena.model._first_user_content` lấy message user CUỐI CÙNG trước lượt
assistant đầu tiên làm câu hỏi của brief, và nó bỏ qua đúng những message
có mang `FINALIZE_SENTINEL`. Nếu bạn chèn một câu nhắc trơn không có
sentinel, mô hình sẽ đi search CHÍNH CÂU NHẮC ĐÓ: mọi brief truy xuất
cùng một mớ tài liệu vô can, mọi bậc thang điểm dịch chuyển đúng 0.00, và
không có một dòng lỗi nào báo cho bạn biết.

TRẢ VỀ `messages + [...]`, ĐỪNG `messages.append(...)`. Agent áp dụng
`before_model` lên một BẢN SAO của lịch sử, nên trả về danh sách mới nghĩa
là "nhắc trong đúng lượt này"; append vào chính danh sách được truyền vào
thì lời nhắc dính vĩnh viễn.

`reserve` KHÔNG PHẢI TRANG TRÍ: `Tools.calls` ĐẾM CẢ `submit`, và scorer
cũng đếm như vậy. Brief cho `max_tool_calls: 8` nghĩa là bảy lượt hữu ích
cộng một lượt submit. Dừng ở `calls >= 8` là tiêu lố đúng một lượt, lần
nào cũng lố.

MỘT HOOK LÀ CHƯA ĐỦ — ĐÃ ĐO. `before_model` chỉ chặn được khi mỗi lượt
model tiêu đúng MỘT lượt công cụ. Không phải vậy: lớp `retry` (§7) có thể
tiêu ba lượt trong CÙNG một vòng, nên một vòng bắt đầu khi còn thiếu đúng
một lượt vẫn kết thúc ở trên ngưỡng. Đo trên full stack: 34/120 lượt chạy
kết thúc ở 9+ lượt gọi trong khi brief cho 8, efficiency 12.06 thay vì
14.24 — trong khi `budget_policy` chạy MỘT MÌNH thì sạch cả 120 lượt.
Vì thế lớp này có thêm `wrap_tool_call`: khi ngân sách chỉ còn phần dự
trữ, TỪ CHỐI gọi công cụ (trả về `ToolResult(ok=False, ...)`, đừng raise —
agent phải sống để còn chốt FINAL). Nửa còn lại nằm ở `retry`: hook
`wrap_tool_call` của `budget_policy` bọc NGOÀI vòng lặp thử lại nên không
nhìn thấy các lượt gọi lại; chỉ chính `retry` mới chặn được `retry`.

CẢNH BÁO ĐÃ ĐO ĐƯỢC — ĐỪNG NÉN NGỮ CẢNH Ở ĐÂY. `before_model` trông rất
hợp lý để "tóm tắt cho gọn", nhưng `MockModel` chỉ trích được câu nào
xuất hiện NGUYÊN VĂN trong danh sách message NÓ ĐANG NHẬN. Một lớp nén
ngữ cảnh tử tế làm mô hình mất khả năng trích dẫn chính những tài liệu nó
vừa đọc: -47.16 điểm trên full stack (92.52 -> 45.36), không có một
thông báo lỗi nào.

CÔNG CỤ CÓ SẴN:
    from arena.model import FINALIZE_SENTINEL
    from arena.tools import ToolResult
    ctx.tools.calls      -> số lượt gọi công cụ đã dùng (kể cả submit)
    ctx.max_tool_calls   -> ngân sách của brief, hoặc None nếu brief không đặt

Cài đặt:  ReActAgent(..., middleware=[..., BudgetPolicy(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

import json
from dataclasses import replace

from arena.model import FINALIZE_SENTINEL, parse_output
from arena.tools import ToolResult

from harness.agent import _canonicalise
from harness.layers._grounding import source_for
from harness.middleware import Middleware

#: Dành lại cho lượt `submit` mà agent vẫn còn phải gọi.
DEFAULT_RESERVE = 1

NUDGE = (
    "Ngân sách công cụ đã hết. Hãy trả lời ngay bằng bằng chứng đang có, "
    f"không gọi thêm công cụ nào nữa. {FINALIZE_SENTINEL}"
)


class BudgetPolicy(Middleware):
    """Ép mô hình chốt FINAL ngay khi ngân sách công cụ đã tiêu hết."""

    name = "budget_policy"

    def __init__(self, reserve: int = DEFAULT_RESERVE) -> None:
        self.reserve = max(0, int(reserve))

    def _spent(self, ctx) -> bool:
        limit = ctx.max_tool_calls
        return limit is not None and ctx.tools.calls >= limit - self.reserve

    def before_model(self, ctx, messages):
        if not self._spent(ctx):
            return messages
        return messages + [{"role": "user", "content": NUDGE}]

    def wrap_tool_call(self, ctx, call, name, args):
        if self._spent(ctx):
            ctx.state["budget_blocked"] = ctx.state.get("budget_blocked", 0) + 1
            return ToolResult(
                ok=False,
                content="",
                error=f"budget exhausted: {name} bị chặn, hãy viết FINAL ngay. {FINALIZE_SENTINEL}",
            )
        result = call(name, args)
        _remember_lookup(ctx, name, args, result)
        return result

    # -- sàn nỗ lực: trần ngân sách có rồi, đây là phía ngược lại ---------

    def after_model(self, ctx, response):
        """Không cho kết luận "không đủ căn cứ" khi chưa hề tra cứu.

        Model thật hay chốt FINAL (thường là abstain) ngay lượt đầu, không
        gọi tool nào: không bằng chứng -> không claim -> sàn điểm. Trước khi
        chấp nhận một FINAL như vậy, lớp này đổi lượt đó thành bước tra cứu
        còn thiếu: chưa search thì search câu hỏi; đã search mà chưa đọc
        toàn văn tài liệu nào thì fetch tài liệu đứng đầu chưa đọc. Trace
        đã ghi nguyên văn output của model trước hook này, nên đây chỉ đổi
        việc agent LÀM, không đổi điều scorer tin là model đã viết.
        """
        if self._spent(ctx) or ctx.state.get("effort_forced", 0) >= MAX_FORCED_LOOKUPS:
            return response
        text = getattr(response, "text", None)
        if not isinstance(text, str):
            return response
        parsed = parse_output(_canonicalise(text))
        if parsed.kind != "final":
            return response
        unsupported = _unsupported_citations(ctx, parsed.final)
        if not _is_empty_handed(parsed.final) and unsupported is None:
            return response
        action = _missing_lookup(ctx, unsupported or [])
        if action is None:
            return response
        ctx.state["effort_forced"] = ctx.state.get("effort_forced", 0) + 1
        thought = "Chưa đọc đủ tài liệu để kết luận, phải tra cứu trước."
        forced = f"THOUGHT: {thought}\nACTION: {json.dumps(action, ensure_ascii=False)}"
        return replace(response, text=forced)


#: Số lần tối đa một lượt chạy bị ép tra cứu thay vì kết luận tay không.
MAX_FORCED_LOOKUPS = 2


def _is_empty_handed(final) -> bool:
    """FINAL không có claim nào (hoặc abstain): kết luận mà không bằng chứng."""
    if not isinstance(final, dict):
        return True
    claims = final.get("claims")
    has_claims = isinstance(claims, list) and any(
        isinstance(c, dict) and isinstance(c.get("text"), str) and c["text"].strip()
        for c in claims
    )
    return final.get("abstain") is True or not has_claims


def _remember_lookup(ctx, name, args, result) -> None:
    lookups = ctx.state.setdefault("lookups", {"search": 0, "listed": [], "fetched": []})
    if not getattr(result, "ok", False):
        return
    if name == "search":
        lookups["search"] += 1
        try:
            hits = json.loads(result.content)
        except (TypeError, ValueError):
            return
        for hit in hits if isinstance(hits, list) else []:
            doc_id = hit.get("doc_id") if isinstance(hit, dict) else None
            if isinstance(doc_id, str) and doc_id not in lookups["listed"]:
                lookups["listed"].append(doc_id)
    elif name == "fetch_doc" and isinstance(args, dict):
        lookups["fetched"].append(str(args.get("doc_id")))


def _unsupported_citations(ctx, final):
    """doc_id mà các claim KHÔNG có trong bằng chứng đã thấy đang trích,
    hoặc None nếu mọi claim đều có căn cứ. Model thật hay "trích" một câu
    nó đoán từ tiêu đề/snippet — đọc toàn văn tài liệu đó trước đã."""
    claims = final.get("claims") if isinstance(final, dict) else None
    if not isinstance(claims, list):
        return None
    cited = [
        c.get("doc_id")
        for c in claims
        if isinstance(c, dict) and isinstance(c.get("text"), str)
        and source_for(ctx, c["text"]) is None
    ]
    return cited or None


def _missing_lookup(ctx, wanted=()):
    lookups = ctx.state.get("lookups") or {"search": 0, "listed": [], "fetched": []}
    if lookups["search"] == 0 and ctx.question:
        return {"tool": "search", "args": {"query": ctx.question, "k": 5}}
    fetched = lookups["fetched"]
    # Ưu tiên chính tài liệu claim đang trích mà agent chưa đọc toàn văn.
    for doc_id in wanted:
        if isinstance(doc_id, str) and doc_id not in fetched and (
            doc_id in lookups["listed"] or (ctx.corpus and ctx.corpus.get(doc_id))
        ):
            return {"tool": "fetch_doc", "args": {"doc_id": doc_id}}
    unread = [d for d in lookups["listed"] if d not in fetched]
    if unread and not fetched:
        return {"tool": "fetch_doc", "args": {"doc_id": unread[0]}}
    return None
