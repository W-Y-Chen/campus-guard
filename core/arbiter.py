"""M6 工具仲裁：污点感知的最小权限判定。

不含任何概率判断 —— 判定是纯查表 + 参数遍历，因此在"工具集封闭、策略表
正确实现"的前提下，越权拦截是**可穷举验证的不变式**，而不是统计结论。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .lexicon import fingerprint_scan
from .types import (ArbiterVerdict, Capability, Sens, TaintString, Trust, Verdict)

# --------------------------------------------------------------------------
# 能力矩阵：(trust, capability) -> 默认动作
# --------------------------------------------------------------------------
DEFAULT_MATRIX: dict[tuple[Trust, Capability], Verdict] = {
    (Trust.SYSTEM, Capability.READ):    Verdict.ALLOW,
    (Trust.USER, Capability.READ):      Verdict.ALLOW,
    (Trust.SEMI, Capability.READ):      Verdict.ALLOW,
    (Trust.UNTRUSTED, Capability.READ): Verdict.ALLOW,

    (Trust.SYSTEM, Capability.WRITE):    Verdict.CONFIRM,
    (Trust.USER, Capability.WRITE):      Verdict.CONFIRM,
    (Trust.SEMI, Capability.WRITE):      Verdict.DENY,
    (Trust.UNTRUSTED, Capability.WRITE): Verdict.DENY,

    (Trust.SYSTEM, Capability.EGRESS):    Verdict.DENY,
    (Trust.USER, Capability.EGRESS):      Verdict.CONFIRM,
    (Trust.SEMI, Capability.EGRESS):      Verdict.DENY,
    (Trust.UNTRUSTED, Capability.EGRESS): Verdict.DENY,
}

SEVERITY = {Verdict.ALLOW: 0, Verdict.SANITIZE: 1, Verdict.CONFIRM: 2, Verdict.DENY: 3}

# 寻址类参数（收件人/抄送/目标地址）不参与敏感指纹扫描
ADDRESS_PARAMS = {"to", "recipient", "cc", "bcc", "addr", "address", "url", "target"}


@dataclass
class ToolSpec:
    name: str
    capability: Capability
    description: str = ""
    max_sens: Sens = Sens.L3      # 允许接触的最高敏感级


@dataclass
class ToolCall:
    name: str
    args: dict[str, TaintString] = field(default_factory=dict)


class Arbiter:
    def __init__(self,
                 tools: dict[str, ToolSpec] | None = None,
                 matrix: dict[tuple[Trust, Capability], Verdict] | None = None) -> None:
        self.tools = tools or default_tools()
        self.matrix = matrix or dict(DEFAULT_MATRIX)

    # ---- 核心仲裁 ----

    def arbitrate(self, call: ToolCall) -> ArbiterVerdict:
        spec = self.tools.get(call.name)
        if spec is None:
            return ArbiterVerdict(Verdict.DENY, call.name, Capability.READ,
                                  "未登记的工具，默认拒绝")

        findings: list[dict] = []

        # ① 敏感级硬门槛
        for pname, arg in call.args.items():
            for sp in arg.spans:
                if sp.sens > spec.max_sens and spec.capability != Capability.READ:
                    findings.append({"param": pname, "issue": "sens",
                                     "sens": int(sp.sens), "src": sp.src_id})
                    return ArbiterVerdict(
                        Verdict.DENY, call.name, spec.capability,
                        f"参数 {pname} 含 L{int(sp.sens)} 级数据，超过工具允许的 "
                        f"L{int(spec.max_sens)} 级", findings)

        # ② 污点硬门槛（核心不变式）
        for pname, arg in call.args.items():
            for sp in arg.spans:
                if sp.trust <= Trust.UNTRUSTED and spec.capability in (
                        Capability.WRITE, Capability.EGRESS):
                    findings.append({"param": pname, "issue": "taint",
                                     "trust": int(sp.trust), "src": sp.src_id})
                    return ArbiterVerdict(
                        Verdict.DENY, call.name, spec.capability,
                        f"参数 {pname} 源自不可信内容（{sp.src_id}），"
                        f"禁止 {spec.capability.value} 动作", findings)

        # ③ 查能力矩阵，取最严格
        all_spans = [sp for arg in call.args.values() for sp in arg.spans]
        if all_spans:
            worst = min(all_spans, key=lambda s: s.trust)
            action = self.matrix.get((worst.trust, spec.capability), Verdict.DENY)
        else:
            action = self.matrix.get((Trust.USER, spec.capability), Verdict.CONFIRM)

        # ④ 输出侧防外带
        # 只对"载荷"参数做指纹扫描：收件人地址本身就是邮箱格式，
        # 若一并扫描会让任何合法邮件都被拒。
        if spec.capability == Capability.EGRESS:
            blob = " ".join(a.value for name, a in call.args.items()
                            if name.lower() not in ADDRESS_PARAMS)
            hits = fingerprint_scan(blob)
            if hits:
                findings.append({"param": "*", "issue": "fingerprint", "types": hits})
                return ArbiterVerdict(
                    Verdict.DENY, call.name, spec.capability,
                    f"检出敏感数据指纹：{sorted(set(hits))}", findings)

        reason = f"来源可信级 {int(worst.trust) if all_spans else 2} × 能力 " \
                 f"{spec.capability.value} -> {action.value}"
        return ArbiterVerdict(action, call.name, spec.capability, reason, findings)


# --------------------------------------------------------------------------
# 默认校园工具集（封闭，便于不变式测试穷举）
# --------------------------------------------------------------------------

def default_tools() -> dict[str, ToolSpec]:
    return {
        "kb_search": ToolSpec("kb_search", Capability.READ,
                              "校园知识库检索", Sens.L3),
        "query_score": ToolSpec("query_score", Capability.READ,
                                "成绩查询", Sens.L4),
        "read_file": ToolSpec("read_file", Capability.READ,
                              "读取本地文件", Sens.L3),
        "fetch_web": ToolSpec("fetch_web", Capability.READ,
                              "抓取网页", Sens.L2),
        "update_record": ToolSpec("update_record", Capability.WRITE,
                                  "修改教务记录", Sens.L3),
        "send_email": ToolSpec("send_email", Capability.EGRESS,
                               "发送邮件（数据出校）", Sens.L2),
        "http_post": ToolSpec("http_post", Capability.EGRESS,
                              "向外部地址 POST", Sens.L2),
    }
