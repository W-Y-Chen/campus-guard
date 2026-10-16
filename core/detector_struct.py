"""M2 结构 / 符号通道（S 通道）。

六类特征，全部为确定性计算，不依赖任何模型：
  f1 隐形字符密度
  f2 同形字密度
  f3 伪分隔符 / 伪角色标签
  f4 指令动词突发比（核心特征）
  f5 编码混淆
  f6 位置可疑度
"""

from __future__ import annotations

import math
import re

from .lexicon import FAKE_DELIM_PATTERNS, IMPERATIVE_EN, IMPERATIVE_ZH, INSTRUCTION_WORDS
from .normalizer import NormText
from .types import Evidence, Trust

FEATURE_WEIGHTS_DEFAULT: dict[str, float] = {
    "f1_invisible": 0.15,
    "f2_homoglyph": 0.15,
    "f3_fake_delim": 0.25,
    "f4_imperative": 0.20,
    "f5_encoding": 0.15,
    "f6_position": 0.10,
}

WINDOW = 50                 # 指令密度滑动窗口（字符数）
BURST_BASELINE = 3.0        # 峰值达基线 3 倍时 f4 ≈ 0.5
_EN_WORDS = set(w.lower() for w in IMPERATIVE_EN)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def _token_count(text: str) -> int:
    """粗略 token 数：中日韩按字计，拉丁按词计。"""
    n = 0
    for ch in text:
        if ord(ch) > 0x2E80:
            n += 1
    n += len(re.findall(r"[A-Za-z]+", text))
    return max(1, n)


def _imperative_hits(text: str) -> int:
    low = text.lower()
    n = sum(text.count(w) for w in IMPERATIVE_ZH)
    n += sum(low.count(w) for w in _EN_WORDS)
    return n


def f1_invisible(nt: NormText) -> float:
    n = nt.flags.get("invisible", 0)
    denom = max(1.0, 0.002 * max(1, len(nt.raw)))
    return _clamp01(n / denom)


def f2_homoglyph(nt: NormText) -> float:
    n = nt.flags.get("homoglyph", 0)
    denom = max(1.0, 0.004 * max(1, len(nt.raw)))
    return _clamp01(n / denom)


def f3_fake_delim(nt: NormText) -> tuple[float, list[Evidence]]:
    ev: list[Evidence] = []
    hits = 0
    for name, pat in FAKE_DELIM_PATTERNS:
        for m in re.finditer(pat, nt.text):
            hits += 1
            ev.append(Evidence("f3_fake_delim", f"{name}: {m.group(0)[:40]}",
                               [m.start()], 0.5))
    return _clamp01(hits / 2.0), ev


def f4_imperative_burst(nt: NormText) -> tuple[float, list[Evidence]]:
    """滑动窗口内指令动词密度峰值 / 全文基线密度。

    这是相对关键词黑名单的核心优势：注入 payload 的共性不是"含某个词"，
    而是"在长文档的某个局部突然出现指令密度尖峰"，对上下文稀释类攻击有效。
    """
    text = nt.text
    if not text:
        return 0.0, []
    base_hits = _imperative_hits(text)
    base_density = base_hits / _token_count(text)
    if base_hits == 0:
        return 0.0, []

    peaks: list[tuple[int, int]] = []
    step = max(1, WINDOW // 2)
    for i in range(0, max(1, len(text)), step):
        seg = text[i:i + WINDOW]
        h = _imperative_hits(seg)
        if h:
            peaks.append((h / max(1, _token_count(seg)), i))
    if not peaks:
        return 0.0, []

    rho_max, pos = max(peaks)
    ratio = rho_max / (base_density + 1e-6)
    score = _sigmoid(math.log(ratio + 1e-9) - math.log(BURST_BASELINE))

    ev = [Evidence("f4_imperative",
                   f"窗口指令密度 {rho_max:.3f} / 基线 {base_density:.3f} = {ratio:.2f}x",
                   [pos], score)]
    return _clamp01(score), ev


def f5_encoding(nt: NormText) -> tuple[float, list[Evidence]]:
    if not nt.decode_hits:
        return 0.0, []
    ev = [Evidence("f5_encoding", f"{h.kind} 解码: {h.decoded[:60]}", [h.start], 1.0)
          for h in nt.decode_hits[:5]]
    return 1.0, ev


def f6_position(nt: NormText) -> float:
    """外部内容缺少来源分隔标记，或位于长文本中后段，则更可疑。"""
    score = 0.0
    trust = nt.spans[0].trust if nt.spans else Trust.USER
    if trust == Trust.UNTRUSTED and not re.search(
            r"(以下内容|以下为|来自|来源|摘录|引用)", nt.text[:60]):
        score += 0.6
    if len(nt.text) > 400 and nt.spans and nt.spans[0].trust == Trust.UNTRUSTED:
        score += 0.4
    return _clamp01(score)


def compute_features(nt: NormText) -> tuple[dict[str, float], list[Evidence]]:
    evidence: list[Evidence] = []

    v1 = f1_invisible(nt)
    if v1 > 0:
        evidence.append(Evidence("f1_invisible",
                                 f"剥离 {nt.flags.get('invisible', 0)} 个隐形字符",
                                 nt.removed[:20], v1))
    v2 = f2_homoglyph(nt)
    if v2 > 0:
        evidence.append(Evidence("f2_homoglyph",
                                 f"归一 {nt.flags.get('homoglyph', 0)} 个同形字",
                                 nt.homoglyph_positions[:20], v2))
    v3, ev3 = f3_fake_delim(nt)
    evidence += ev3
    v4, ev4 = f4_imperative_burst(nt)
    evidence += ev4
    v5, ev5 = f5_encoding(nt)
    evidence += ev5
    v6 = f6_position(nt)
    if v6 > 0:
        evidence.append(Evidence("f6_position", "外部内容位置/来源标记可疑", [], v6))

    feats = {
        "f1_invisible": v1, "f2_homoglyph": v2, "f3_fake_delim": v3,
        "f4_imperative": v4, "f5_encoding": v5, "f6_position": v6,
    }
    return feats, evidence


def struct_score(nt: NormText,
                 weights: dict[str, float] | None = None) -> tuple[float, dict, list[Evidence]]:
    w = weights or FEATURE_WEIGHTS_DEFAULT
    feats, evidence = compute_features(nt)
    s = sum(w.get(k, 0.0) * v for k, v in feats.items())
    return _clamp01(s), feats, evidence


def uncertainty_S(nt: NormText, feats: dict[str, float]) -> float:
    """结构通道的不确定度：来自特征之间的冲突。"""
    u = 0.0
    trust = nt.spans[0].trust if nt.spans else Trust.USER
    if feats["f1_invisible"] > 0.3 and feats["f4_imperative"] < 0.2:
        u += 0.25                      # 隐形字符多但没有指令词：可能是复制粘贴噪声
    if feats["f3_fake_delim"] > 0.5 and trust == Trust.SYSTEM:
        u += 0.30                      # 系统提示词本来就有角色标签
    if feats["f4_imperative"] > 0.6 and trust == Trust.USER:
        u += 0.20                      # 用户本来就可能下达强指令
    return min(0.6, u)


__all__ = [
    "FEATURE_WEIGHTS_DEFAULT", "struct_score", "compute_features",
    "uncertainty_S", "INSTRUCTION_WORDS",
]
