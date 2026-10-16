"""M5 污点引擎：切片级污点表示与传播。

TaintString 数据结构定义在 core/types.py，本模块提供传播辅助函数与
"模型输出派生污点"标注策略。
"""

from __future__ import annotations

import json

from .types import Sens, Span, TaintString, Trust


def taint(value: str, src_id: str, trust: Trust, sens: Sens = Sens.L1) -> TaintString:
    return TaintString(value, [Span(0, len(value), src_id, trust, sens)])


def concat(parts: list[TaintString | str]) -> TaintString:
    out = TaintString("")
    for p in parts:
        out = out + (p if isinstance(p, TaintString) else TaintString(p))
    return out


def replace(src: TaintString, old: str, new: str | TaintString) -> TaintString:
    """替换：未命中区间保留原 span，命中区间采用 new 的来源。"""
    new_ts = new if isinstance(new, TaintString) else TaintString(
        new, [Span(0, len(new), "literal", Trust.SYSTEM, Sens.L1)])
    pieces: list[TaintString] = []
    idx = 0
    while True:
        pos = src.value.find(old, idx)
        if pos < 0:
            pieces.append(src[idx:])
            break
        pieces.append(src[idx:pos])
        pieces.append(new_ts)
        idx = pos + len(old)
    return concat(pieces)


def to_json(obj, base: TaintString | None = None) -> TaintString:
    """JSON 序列化：视为字符串拼接，污点跟随来源。"""
    s = json.dumps(obj, ensure_ascii=False)
    if base is None:
        return TaintString(s)
    return TaintString(s, list(base.spans))


def extract(src: TaintString, start: int, end: int) -> TaintString:
    """正则提取等价：取子区间并裁剪 span。"""
    return src[start:end]


def derive_from_context(output: str,
                        ctx_spans: list[Span],
                        strict: bool = True) -> TaintString:
    """模型输出的派生污点标注。

    严格模式：上下文中出现过不可信内容，则整段输出标记为 DERIVED_TAINTED。
    宽松模式：仅当输出与污染来源有字面重叠时才标注。
    """
    if not ctx_spans:
        return TaintString(output)
    untrusted = [s for s in ctx_spans if s.trust <= Trust.SEMI]
    if not untrusted:
        return TaintString(output)

    if strict:
        worst_trust = min(s.trust for s in untrusted)
        worst_sens = max(s.sens for s in untrusted)
        return TaintString(output, [Span(0, len(output), "llm:derived",
                                         worst_trust, worst_sens, "DERIVED_TAINTED")])

    spans: list[Span] = []
    for s in untrusted:
        probe = s.note or ""
        if probe and probe in output:
            i = output.find(probe)
            spans.append(Span(i, i + len(probe), "llm:derived", s.trust, s.sens))
    return TaintString(output, spans)
