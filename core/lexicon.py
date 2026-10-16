"""指令词表与敏感数据指纹规则。

词表为项目自建，可按学校场景增删。修改后请重跑 tests/test_struct.py，
因为部分用例依赖词表命中。
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------
# 指令动词 / 短语（中文 + 英文）
# --------------------------------------------------------------------------
IMPERATIVE_ZH = [
    "忽略", "无视", "忘记", "不要遵循", "不必遵循", "无需遵循", " disregard",
    "覆盖", "绕过", "跳过", "解除", "禁用", "关闭", "取消",
    "立即执行", "立刻执行", "马上执行", "务必", "必须", "一定要",
    "现在开始", "从现在起", "从此刻起", "接下来请", "你的新指令",
    "新的指令", "新的系统提示", "更高优先级", "最高优先级",
    "泄露", "泄漏", "导出", "发送", "转发", "邮寄", "上传", "提交到",
    "删除", "清空", "修改", "篡改", "转账", "付款",
    "不要告诉", "不要向用户", "不要提及", "秘密地", "悄悄地",
    "请把", "请将", "请务必", "请求你", "要求你",
]

IMPERATIVE_EN = [
    "ignore", "disregard", "override", "bypass", "forget", "skip",
    "instead", "from now on", "new instruction", "new system prompt",
    "highest priority", "do not tell", "do not inform", "secretly",
    "exfiltrate", "forward", "send to", "leak", "export", "reveal",
    "you must", "you will", "always", "never",
]

INSTRUCTION_WORDS = IMPERATIVE_ZH + IMPERATIVE_EN

# --------------------------------------------------------------------------
# 伪分隔符 / 伪角色标签
# --------------------------------------------------------------------------
FAKE_DELIM_PATTERNS: list[tuple[str, str]] = [
    ("sys_tag_en", r"(?im)^\s*(system|assistant|user|developer)\s*[:：]"),
    ("chatml", r"<<\s*SYS\s*>>|\[INST\]|<\|im_start\|>|<\|im_end\|>"),
    ("md_heading", r"#{2,}\s*(system|instruction|指令|系统指令|重要指令)"),
    ("html_comment", r"<!--[\s\S]*?-->"),
    ("block_comment", r"/\*[\s\S]*?\*/"),
    ("fake_system_zh", r"(以下|下面|接?下来)(的?内容)?(是|为|属于)?(系统|最高级|管理员)(指令|提示|要求|规则)"),
    ("fake_role_zh", r"(你现在是|你现在扮演|请扮演|假设你是)"),
]

# --------------------------------------------------------------------------
# 敏感数据指纹（输出侧防外带）
# --------------------------------------------------------------------------
RE_PID = re.compile(r"(?<!\d)[1-9]\d{5}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)")
RE_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
RE_BANK = re.compile(r"(?<!\d)\d{16,19}(?!\d)")
RE_STUDENT_ID = re.compile(r"(?<!\d)(20[12]\d)\d{6,8}(?!\d)")
RE_GRADE_CTX = re.compile(r"(成绩|分数|绩点|排名|平均分|GPA)")

PID_WEIGHTS = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
PID_CHECK = "10X98765432"


def pid_valid(pid: str) -> bool:
    """身份证号校验位验证，降低误报。"""
    if len(pid) != 18:
        return False
    try:
        s = sum(int(pid[i]) * PID_WEIGHTS[i] for i in range(17))
    except ValueError:
        return False
    return PID_CHECK[s % 11] == pid[17].upper()


def luhn_valid(num: str) -> bool:
    """Luhn 校验，用于银行卡号降噪。"""
    total, alt = 0, False
    for ch in reversed(num):
        if not ch.isdigit():
            return False
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


FINGERPRINT_RULES = [
    ("pid", RE_PID, pid_valid),
    ("phone", RE_PHONE, None),
    ("email", RE_EMAIL, None),
    ("bank", RE_BANK, luhn_valid),
    ("student_id", RE_STUDENT_ID, None),
]


def fingerprint_scan(text: str) -> list[str]:
    """返回命中的敏感数据类型列表。"""
    hits: list[str] = []
    for name, rx, validator in FINGERPRINT_RULES:
        for m in rx.finditer(text):
            if validator is None or validator(m.group(0)):
                hits.append(name)
                break
    if RE_GRADE_CTX.search(text) and re.search(r"\d{1,3}(\.\d+)?\s*(分|名)", text):
        hits.append("grade")
    return hits
