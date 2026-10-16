"""全局类型定义。

本文件只依赖 Python 标准库，是整个工程的类型契约。
所有跨模块数据交换都使用这里定义的结构，修改前请先同步 tests/。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any


class Trust(IntEnum):
    """来源可信级。数值越低越不可信。"""

    UNTRUSTED = 0   # 外部抓取：网页 / 附件 / 邮件正文 / OCR 结果
    SEMI = 1        # 半可信：校内知识库条目（可能被师生提交内容污染）
    USER = 2        # 当前会话用户的直接输入
    SYSTEM = 3      # 系统提示词 / 管理员配置


class Sens(IntEnum):
    """数据敏感级，对齐《教育数据分类分级指南》JY/T 0661-2025 的 L1-L5。"""

    L1 = 1   # 公开
    L2 = 2   # 一般
    L3 = 3   # 一般（含敏感个人信息，需重点保护）
    L4 = 4   # 重要数据
    L5 = 5   # 核心数据


class Action(str, Enum):
    """检测层三态决策。"""

    BLOCK = "block"           # 阻断
    QUARANTINE = "quarantine" # 隔离：降级为纯数据放行
    PASS = "pass"             # 放行


class Verdict(str, Enum):
    """仲裁层动作。"""

    ALLOW = "allow"       # 允许
    SANITIZE = "sanitize" # 脱敏后允许
    CONFIRM = "confirm"   # 人工确认
    DENY = "deny"         # 拒绝


class Capability(str, Enum):
    """工具能力（按副作用划分）。"""

    READ = "read"      # 只查询
    WRITE = "write"    # 改变校内系统状态
    EGRESS = "egress"  # 数据出校


class Cause(str, Enum):
    """存活变异体归因类别。"""

    STRUCT_BLIND = "struct_blind"    # 结构通道没命中
    SEMANTIC_BLIND = "semantic_blind"# 两通道都没感觉
    FUSION_WEIGHT = "fusion_weight"  # 规则命中但融合没判成
    THRESHOLD = "threshold"          # 阈值不当
    OOD_UNCERTAIN = "ood_uncertain"  # 语义已认出，但分布外不确定度把它压成隔离


# --------------------------------------------------------------------------
# 污点模型
# --------------------------------------------------------------------------


@dataclass
class Span:
    """字符串上的一段来源标注。区间为左闭右开 [start, end)。"""

    start: int
    end: int
    src_id: str
    trust: Trust
    sens: Sens
    note: str = ""

    def shifted(self, offset: int) -> "Span":
        return Span(self.start + offset, self.end + offset,
                    self.src_id, self.trust, self.sens, self.note)

    def cropped(self, lo: int, hi: int, new_base: int = 0) -> "Span | None":
        s, e = max(self.start, lo), min(self.end, hi)
        if s >= e:
            return None
        return Span(s - lo + new_base, e - lo + new_base,
                    self.src_id, self.trust, self.sens, self.note)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start, "end": self.end, "src_id": self.src_id,
            "trust": int(self.trust), "sens": int(self.sens), "note": self.note,
        }


@dataclass
class TaintString:
    """带污点标注的字符串：值 + 来源片段列表（有序、不重叠）。"""

    value: str
    spans: list[Span] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.spans = TaintString._normalize(self.spans, len(self.value))

    @staticmethod
    def _normalize(spans: list[Span], length: int) -> list[Span]:
        """裁剪越界 + 按起点排序 + 合并相邻且同属性的区间。"""
        clipped: list[Span] = []
        for sp in spans:
            s, e = max(0, sp.start), min(length, sp.end)
            if s < e:
                clipped.append(Span(s, e, sp.src_id, sp.trust, sp.sens, sp.note))
        clipped.sort(key=lambda x: (x.start, x.end))
        merged: list[Span] = []
        for sp in clipped:
            if merged:
                prev = merged[-1]
                if (sp.start <= prev.end and prev.src_id == sp.src_id
                        and prev.trust == sp.trust and prev.sens == sp.sens):
                    prev.end = max(prev.end, sp.end)
                    continue
            merged.append(sp)
        return merged

    # ---- 传播操作 ----

    def __add__(self, other: "TaintString | str") -> "TaintString":
        o = other if isinstance(other, TaintString) else TaintString(other)
        return TaintString(self.value + o.value,
                           self.spans + [s.shifted(len(self.value)) for s in o.spans])

    def __getitem__(self, sl: slice) -> "TaintString":
        if not isinstance(sl, slice):
            sl = slice(sl, sl + 1)
        start, stop, _ = sl.indices(len(self.value))
        new_val = self.value[start:stop]
        new_spans = [s for s in (sp.cropped(start, stop, 0) for sp in self.spans) if s]
        return TaintString(new_val, new_spans)

    def with_span(self, src_id: str, trust: Trust, sens: Sens) -> "TaintString":
        return TaintString(self.value, [Span(0, len(self.value), src_id, trust, sens)])

    def max_sens(self) -> Sens:
        return max((s.sens for s in self.spans), default=Sens.L1)

    def min_trust(self) -> Trust:
        return min((s.trust for s in self.spans), default=Trust.SYSTEM)

    def covered_ratio(self, pred) -> float:
        """满足 pred(span) 的片段覆盖的字符比例。"""
        if not self.value:
            return 0.0
        n = sum(s.end - s.start for s in self.spans if pred(s))
        return n / len(self.value)

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "spans": [s.to_dict() for s in self.spans]}


# --------------------------------------------------------------------------
# 检测结果
# --------------------------------------------------------------------------


@dataclass
class Evidence:
    feature: str
    detail: str
    positions: list[int] = field(default_factory=list)
    strength: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"feature": self.feature, "detail": self.detail,
                "positions": self.positions[:20], "strength": round(self.strength, 4)}


@dataclass
class BPA:
    """Dempster-Shafer 基本概率分配：m(注入) / m(正常) / m(不确定)。"""

    inj: float
    ben: float
    theta: float

    def to_dict(self) -> dict[str, Any]:
        return {"inj": round(self.inj, 4), "ben": round(self.ben, 4),
                "theta": round(self.theta, 4)}


@dataclass
class Decision:
    action: Action
    bpa: BPA
    reason: str
    evidence: list[Evidence] = field(default_factory=list)
    struct_score: float = 0.0
    semantic_score: float = 0.0
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "bpa": self.bpa.to_dict(),
            "reason": self.reason,
            "evidence": [e.to_dict() for e in self.evidence],
            "struct_score": round(self.struct_score, 4),
            "semantic_score": round(self.semantic_score, 4),
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


@dataclass
class ArbiterVerdict:
    verdict: Verdict
    tool: str
    capability: Capability
    reason: str
    param_findings: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value, "tool": self.tool,
            "capability": self.capability.value, "reason": self.reason,
            "param_findings": self.param_findings,
        }
