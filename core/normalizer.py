"""M1 文本归一化与结构指纹提取。

职责：把"看起来正常、实际藏了东西"的文本还原成可判定形式，同时保留
归一化后位置 -> 原文位置的映射，供控制台高亮与证据回溯使用。

纯标准库实现。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .types import Sens, Span, Trust

# --------------------------------------------------------------------------
# 隐形 / 格式字符集合（U+00AD 软连字符、U+200B-200F、U+202A-202E、
# U+2060-2064、U+FEFF、U+180E 蒙古文元音分隔符）
# --------------------------------------------------------------------------
INVISIBLE_RANGES = [
    (0x00AD, 0x00AD),
    (0x180E, 0x180E),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x2064),
    (0xFEFF, 0xFEFF),
]


def is_invisible(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in INVISIBLE_RANGES)


# --------------------------------------------------------------------------
# 同形字映射表：人工裁剪的子集（西里尔 / 希腊 / 全角 / 数学字母符号 -> 拉丁）
# 说明：不直接整表引入 Unicode confusables（上万条会误伤中文与生僻字），
#       这里只保留教学与演示场景中最常出现的混淆字符。
# --------------------------------------------------------------------------
_HOMOGLYPH_RAW = {
    # 西里尔（含扩展字符，如 і U+0456 / ј U+0458 / ѕ U+0455）
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d", "һ": "h", "ӏ": "l",
    "ԛ": "q", "ԝ": "w", "в": "b", "п": "n", "г": "r", "м": "m", "т": "t",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",
    # 希腊
    "α": "a", "ο": "o", "ν": "v", "τ": "t", "ρ": "p", "Α": "A", "Β": "B",
    "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N",
    "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    # 全角 ASCII（只取字母与 @：，。：；（）等是**中文正文的标准标点**，
    # 若一并计入同形字，会让任何中文文本的结构分虚高，属于误报来源）
    "＠": "@",
    "ａ": "a", "ｂ": "b", "ｃ": "c", "ｄ": "d", "ｅ": "e", "ｆ": "f", "ｇ": "g",
    "ｈ": "h", "ｉ": "i", "ｊ": "j", "ｋ": "k", "ｌ": "l", "ｍ": "m", "ｎ": "n",
    "ｏ": "o", "ｐ": "p", "ｑ": "q", "ｒ": "r", "ｓ": "s", "ｔ": "t", "ｕ": "u",
    "ｖ": "v", "ｗ": "w", "ｘ": "x", "ｙ": "y", "ｚ": "z",
    "Ａ": "A", "Ｂ": "B", "Ｃ": "C", "Ｄ": "D", "Ｅ": "E", "Ｆ": "F", "Ｇ": "G",
    "Ｈ": "H", "Ｉ": "I", "Ｊ": "J", "Ｋ": "K", "Ｌ": "L", "Ｍ": "M", "Ｎ": "N",
    "Ｏ": "O", "Ｐ": "P", "Ｑ": "Q", "Ｒ": "R", "Ｓ": "S", "Ｔ": "T", "Ｕ": "U",
    "Ｖ": "V", "Ｗ": "W", "Ｘ": "X", "Ｙ": "Y", "Ｚ": "Z", "＠": "@", "．": ".",
}
HOMOGLYPH_MAP: dict[str, str] = _HOMOGLYPH_RAW

# 编码混淆扫描
_B64_RE = re.compile(r"[A-Za-z0-9+/=]{16,}")
_URLENC_RE = re.compile(r"(?:%[0-9A-Fa-f]{2}){3,}")
_HEX_RE = re.compile(r"(?:0x)?[0-9A-Fa-f]{32,}")


@dataclass
class DecodeHit:
    kind: str          # base64 / rot13 / url / hex
    raw: str
    decoded: str
    start: int


@dataclass
class NormText:
    """归一化结果。"""

    text: str                              # 归一化后文本
    raw: str                               # 原始文本
    pos_map: list[int]                     # norm[i] -> 原文下标
    flags: dict[str, int] = field(default_factory=dict)
    removed: list[int] = field(default_factory=list)   # 被剥离字符的原文下标
    decode_hits: list[DecodeHit] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)
    homoglyph_positions: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "flags": self.flags,
            "removed_count": len(self.removed),
            "decode_hits": [
                {"kind": h.kind, "decoded": h.decoded[:120]} for h in self.decode_hits
            ],
        }


def _rot13(s: str) -> str:
    out = []
    for ch in s:
        if "a" <= ch <= "z":
            out.append(chr((ord(ch) - 97 + 13) % 26 + 97))
        elif "A" <= ch <= "Z":
            out.append(chr((ord(ch) - 65 + 13) % 26 + 65))
        else:
            out.append(ch)
    return "".join(out)


def _try_b64(seg: str) -> str | None:
    import base64
    try:
        pad = "=" * (-len(seg) % 4)
        data = base64.b64decode(seg + pad, validate=False)
        return data.decode("utf-8", errors="strict")
    except Exception:
        return None


def _try_urlenc(seg: str) -> str | None:
    from urllib.parse import unquote
    try:
        return unquote(seg)
    except Exception:
        return None


def scan_encodings(text: str, instruction_words: list[str] | None = None) -> list[DecodeHit]:
    """扫描 Base64 / ROT13 / URL 编码 / 长十六进制，解码后若含指令词则记为命中。"""
    words = instruction_words or []
    hits: list[DecodeHit] = []

    for m in _B64_RE.finditer(text):
        dec = _try_b64(m.group(0))
        if dec and any(ord(c) > 127 or c.isalnum() or c.isspace() for c in dec):
            if not words or any(w in dec for w in words):
                hits.append(DecodeHit("base64", m.group(0), dec, m.start()))

    for m in _URLENC_RE.finditer(text):
        dec = _try_urlenc(m.group(0))
        if dec and (not words or any(w in dec for w in words)):
            hits.append(DecodeHit("url", m.group(0), dec, m.start()))

    # ROT13：整串试解后看是否命中指令词
    r = _rot13(text)
    if r != text and words and any(w in r for w in words):
        hits.append(DecodeHit("rot13", text[:80], r[:80], 0))

    return hits


def normalize(
    raw: str,
    src_id: str = "user:0",
    trust: Trust = Trust.USER,
    sens: Sens = Sens.L1,
    instruction_words: list[str] | None = None,
) -> NormText:
    """归一化主流程。

    顺序很重要：先剥离/映射，再 NFKC —— 反过来的话部分兼容字符会先被
    折叠，导致隐形字符证据丢失。
    """
    chars: list[str] = []
    pos_map: list[int] = []
    removed: list[int] = []
    homoglyph_positions: list[int] = []
    flags = {"invisible": 0, "homoglyph": 0}

    for i, ch in enumerate(raw):
        cp = ord(ch)
        if is_invisible(cp):
            flags["invisible"] += 1
            removed.append(i)
            continue
        if ch in HOMOGLYPH_MAP:
            flags["homoglyph"] += 1
            homoglyph_positions.append(i)
            ch = HOMOGLYPH_MAP[ch]
        chars.append(ch)
        pos_map.append(i)

    text = unicodedata.normalize("NFKC", "".join(chars))
    # NFKC 可能改变长度；为保持 pos_map 可用，长度变化时做安全兜底
    if len(text) != len(pos_map):
        pos_map = list(range(min(len(text), len(raw))))

    decode_hits = scan_encodings(text, instruction_words)

    spans = [Span(0, len(text), src_id, trust, sens)] if text else []
    return NormText(
        text=text,
        raw=raw,
        pos_map=pos_map,
        flags=flags,
        removed=removed,
        decode_hits=decode_hits,
        spans=spans,
        homoglyph_positions=homoglyph_positions,
    )


def highlight_positions(nt: NormText, norm_start: int, norm_end: int) -> list[int]:
    """把归一化区间映射回原文下标，供控制台高亮。"""
    out = []
    for i in range(norm_start, min(norm_end, len(nt.pos_map))):
        out.append(nt.pos_map[i])
    return out
