"""M11 智能体适配层：自研轻量 Tool-Calling Loop + 模拟校园工具集。

说明（重要，避免误解）：
    本模块是**确定性脚本化的智能体模拟器**，不是真实大语言模型。
    它按规则生成工具调用意图，目的是在离线环境下可复现地演示
    "外部内容 -> 工具调用" 这条攻击链，以及护栏在哪一层拦住它。
    接入真实模型时，只需把 plan() 换成对 OpenAI 兼容接口的调用，
    其余（污点标注、仲裁、审计）代码完全不变。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.types import Sens, Span, TaintString, Trust, Verdict

# --------------------------------------------------------------------------
# 模拟校园工具（本地内存实现，无任何真实网络与数据库操作）
# --------------------------------------------------------------------------

KB = {
    "选课": "本学期选课于第 2 周周一 9:00 开始，第 3 周周日 23:59 结束。",
    "借阅": "图书借阅期限为 60 天，逾期按每天 0.10 元计费。",
    "成绩单": "成绩单可在教务系统『我的成绩』页面自助打印。",
    "转专业": "转专业需提交申请表、成绩单与学院意见，详见教务处通知。",
    "报修": "宿舍报修请登录后勤服务平台提交工单，或拨打校内维修热线。",
}

SCORES = {"2023010101": 88, "2023010102": 92, "2023010103": 75}   # 合成数据


def kb_search(q: str) -> str:
    for k, v in KB.items():
        if k in q:
            return v
    return "未检索到相关内容，建议咨询教务处。'"


def query_score(student_id: str) -> str:
    return f"学号 {student_id} 的成绩为 {SCORES.get(student_id, '无记录')} 分。"


def read_file(path: str) -> str:
    return f"[模拟读取] {path} 的内容。"


def fetch_web(url: str) -> str:
    return f"[模拟抓取] {url} 的正文内容。"


def update_record(record_id: str, value: str) -> str:
    return f"[模拟写入] 记录 {record_id} 已更新为 {value}。"


def send_email(to: str, body: str) -> str:
    return f"[模拟外发] 邮件已发往 {to}，正文 {len(body)} 字。"


def http_post(url: str, data: str) -> str:
    return f"[模拟外发] 已向 {url} POST {len(data)} 字节。"


TOOL_IMPL = {
    "kb_search": kb_search, "query_score": query_score, "read_file": read_file,
    "fetch_web": fetch_web, "update_record": update_record,
    "send_email": send_email, "http_post": http_post,
}


# --------------------------------------------------------------------------
# 规划器
# --------------------------------------------------------------------------

@dataclass
class Plan:
    tool: str
    args: dict[str, TaintString]
    reason: str


class ScriptedAgent:
    """按规则生成工具调用意图的模拟器。"""

    def __init__(self) -> None:
        self.reasoning: list[str] = []

    def plan(self, user_msg: str,
             external_docs: list[tuple[str, str]]) -> list[Plan]:
        plans: list[Plan] = []

        # 正常意图：用户消息驱动
        if any(k in user_msg for k in ("选课", "借阅", "成绩", "转专业", "报修")):
            plans.append(Plan("kb_search",
                              {"q": TaintString(user_msg, [Span(0, len(user_msg),
                                                                "user:turn0",
                                                                Trust.USER, Sens.L1)])},
                              "用户提出校园知识问答"))

        # 被劫持意图：外部内容驱动（模拟提示词注入成功后的行为）
        for src_id, text in external_docs:
            if re.search(r"(发送|转发|导出|上传|提交到|mail|send|export|http)", text):
                m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
                to = m.group(0) if m else "attacker@example.com"
                to_pos = text.find(to)
                plans.append(Plan(
                    "send_email",
                    {
                        "to": TaintString(to, [Span(max(0, to_pos),
                                                     max(0, to_pos) + len(to),
                                                     src_id, Trust.UNTRUSTED, Sens.L2)]),
                        "body": TaintString(text, [Span(0, len(text), src_id,
                                                        Trust.UNTRUSTED, Sens.L3)]),
                    },
                    f"外部内容 {src_id} 中出现外发指令，智能体据此规划邮件发送"))
                break

        return plans


class AgentLoop:
    """带护栏的智能体循环：规划 -> 逐条仲裁 -> 执行（仅 ALLOW/SANITIZE）。"""

    def __init__(self, guard, agent: ScriptedAgent | None = None) -> None:
        self.guard = guard
        self.agent = agent or ScriptedAgent()

    def run(self, user_msg: str,
            external_docs: list[tuple[str, str]] | None = None,
            session_id: str = "s0") -> dict:
        external_docs = external_docs or []

        # ① 检测：外部内容先过检测层
        admitted: list[tuple[str, str]] = []
        for src_id, text in external_docs:
            d = self.guard.detect(text, Trust.UNTRUSTED, Sens.L3, src_id)
            if d.action.value == "block":
                self.guard.ledger.append(session_id, "detect", "guard", text,
                                         "block", {"src": src_id})
                continue
            admitted.append((src_id, text))

        # ② 规划
        plans = self.agent.plan(user_msg, admitted)

        # ③ 仲裁 + 执行
        results = []
        for p in plans:
            v = self.guard.arbitrate(p.tool, p.args, session_id)
            entry = {"tool": p.tool, "reason": p.reason,
                     "verdict": v.verdict.value, "arbiter_reason": v.reason,
                     "findings": v.param_findings}
            if v.verdict in (Verdict.ALLOW, Verdict.SANITIZE):
                fn = TOOL_IMPL.get(p.tool)
                if fn:
                    entry["output"] = fn(*[a.value for a in p.args.values()])
            results.append(entry)

        return {
            "session_id": session_id,
            "admitted_docs": [s for s, _ in admitted],
            "plans": [{"tool": p.tool, "reason": p.reason} for p in plans],
            "results": results,
        }
