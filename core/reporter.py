"""M10 合规报告生成（纯标准库，输出 Markdown + JSON）。

三类报告分别对齐：
  1. 安全事件处置报告  ->《生成式人工智能服务管理暂行办法》第 14 条
     （发现违法内容应及时停止生成、消除处置并向主管部门报告）
  2. 教育数据分类分级使用报告 -> JY/T 0661-2025（六类数据 / L1-L5 分级）
  3. 个人信息保护合规审计摘要 -> 年度合规审计与报送要求
"""

from __future__ import annotations

import time
from collections import Counter

from .types import Sens, Trust


def _pct(n: int, total: int) -> str:
    return f"{(n / total * 100):.1f}%" if total else "0.0%"


class Reporter:
    def __init__(self, ledger) -> None:
        self.ledger = ledger

    def _records(self, session_id: str | None = None):
        if session_id:
            return self.ledger.session(session_id)
        return self.ledger.records

    # ---------------- 1. 安全事件处置报告 ----------------

    def security_incident(self, session_id: str | None = None) -> str:
        recs = self._records(session_id)
        total = len(recs)
        by_decision = Counter(r.decision for r in recs)
        by_stage = Counter(r.stage for r in recs)
        blocked = [r for r in recs if r.decision in ("block", "deny")]

        lines = [
            "# 安全事件处置报告",
            "",
            f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- 会话标识：{session_id or '全部会话'}",
            f"- 审计记录数：{total}",
            f"- 审计链校验：{'通过' if self.ledger.verify()['ok'] else '未通过'}",
            "",
            "## 一、处置概况",
            "",
            "| 处置动作 | 次数 | 占比 |",
            "|---|---|---|",
        ]
        for k, v in by_decision.most_common():
            lines.append(f"| {k} | {v} | {_pct(v, total)} |")
        lines += ["", "## 二、触发环节分布", "", "| 环节 | 次数 |", "|---|---|"]
        for k, v in by_stage.most_common():
            lines.append(f"| {k} | {v} |")
        lines += [
            "",
            "## 三、处置措施（对齐《生成式人工智能服务管理暂行办法》第 14 条）",
            "",
            "已采取的处置措施：",
            "1. 对判定为注入的内容执行停止生成与阻断；",
            "2. 对可疑内容执行降级隔离，仅作为数据处理，不作为指令执行；",
            "3. 对涉及写操作与数据出校的工具调用执行拒绝；",
            "4. 全过程留痕，记录哈希上链以支持事后追溯与向主管部门报告。",
            "",
            "## 四、证据清单（前 20 条）",
            "",
            "| 序号 | 时间 | 环节 | 处置 | 证据哈希 |",
            "|---|---|---|---|---|",
        ]
        for r in blocked[:20]:
            lines.append(f"| {r.seq} | {r.ts} | {r.stage} | {r.decision} | {r.hash[:16]} |")
        return "\n".join(lines)

    # ---------------- 2. 教育数据分类分级使用报告 ----------------

    def data_classification(self, session_id: str | None = None) -> str:
        recs = self._records(session_id)
        sens_counter: Counter = Counter()
        trust_counter: Counter = Counter()
        for r in recs:
            for ev in (r.evidence or {}).get("spans", []):
                sens_counter[ev.get("sens", 1)] += 1
                trust_counter[ev.get("trust", 3)] += 1
        total_s = sum(sens_counter.values()) or 1

        lines = [
            "# 教育数据分类分级使用报告",
            "",
            "分级依据：《教育数据分类分级指南》JY/T 0661-2025（L1 公开 / L2 一般 / "
            "L3 一般含敏感个人信息 / L4 重要数据 / L5 核心数据，就高从严）。",
            "",
            "## 一、敏感级分布",
            "",
            "| 敏感级 | 片段数 | 占比 | 保护要求 |",
            "|---|---|---|---|",
        ]
        desc = {
            1: "可公开", 2: "一般保护", 3: "重点保护（含敏感个人信息）",
            4: "重要数据保护", 5: "核心数据保护",
        }
        for lvl in range(1, 6):
            n = sens_counter.get(lvl, 0)
            lines.append(f"| L{lvl} | {n} | {_pct(n, total_s)} | {desc[lvl]} |")
        lines += ["", "## 二、来源可信级分布", "", "| 可信级 | 片段数 |", "|---|---|"]
        tdesc = {0: "外部不可信", 1: "半可信", 2: "用户输入", 3: "系统"}
        for t in range(0, 4):
            lines.append(f"| {t}（{tdesc[t]}） | {trust_counter.get(t, 0)} |")
        lines += [
            "",
            "## 三、结论与建议",
            "",
            "1. 对 L3 及以上数据，禁止经由写操作与出校通道流转；",
            "2. 对来源可信级为 0 的外部内容，默认隔离并禁止其驱动任何工具调用；",
            "3. 数据业务属性或使用场景发生变化时，应重新分级并按流程报批（动态调整原则）。",
        ]
        return "\n".join(lines)

    # ---------------- 3. 个人信息保护合规审计摘要 ----------------

    def privacy_audit(self, session_id: str | None = None) -> str:
        recs = self._records(session_id)
        fp_counter: Counter = Counter()
        denied = 0
        for r in recs:
            fps = (r.evidence or {}).get("fingerprints", [])
            for f in fps:
                fp_counter[f] += 1
            if r.decision == "deny":
                denied += 1
        lines = [
            "# 个人信息保护合规审计摘要",
            "",
            f"- 审计周期：全部记录（{len(recs)} 条）",
            f"- 敏感数据指纹命中：{sum(fp_counter.values())} 次",
            f"- 工具调用拒绝次数：{denied}",
            "",
            "## 一、敏感数据接触情况",
            "",
            "| 类型 | 命中次数 |",
            "|---|---|",
        ]
        names = {"pid": "身份证号", "phone": "手机号", "email": "邮箱",
                 "bank": "银行卡号", "student_id": "学号", "grade": "成绩/排名"}
        for k, v in fp_counter.most_common():
            lines.append(f"| {names.get(k, k)} | {v} |")
        if not fp_counter:
            lines.append("| （无） | 0 |")
        lines += [
            "",
            "## 二、控制有效性",
            "",
            "1. 输出侧指纹检测覆盖身份标识、联系方式、金融账号、学号与成绩；",
            "2. 命中指纹且目标工具具出校能力时一律拒绝；",
            "3. 审计记录仅保存原文哈希，不留存明文，降低日志本身成为泄露源的风险；",
            "4. 审计链为只追加结构，任何删改均可被校验发现。",
            "",
            "## 三、年度合规审计建议",
            "",
            "建议按年度对上述控制措施执行有效性复核，并保留本报告与审计链导出文件"
            "作为报送材料附件。",
        ]
        return "\n".join(lines)

    def all(self, session_id: str | None = None) -> dict[str, str]:
        return {
            "security_incident": self.security_incident(session_id),
            "data_classification": self.data_classification(session_id),
            "privacy_audit": self.privacy_audit(session_id),
        }
