"""M4 Dempster-Shafer 证据融合与三态决策。

辨识框架 Ω = {inj, ben}，幂集元素 {inj}、{ben}、Θ={inj,ben}（不确定）。
两个通道的输出统一表达为基本概率分配（BPA），用 Dempster 组合规则合成，
再按三态规则决策。

相比加权平均的核心价值：能显式表达"两通道冲突 / 都不确定"，把拿不准的
样本送进隔离区而不是硬判，从而压低误报率。
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import Action, BPA, Decision

TAU_HI = 0.30     # 注入 vs 正常的优势阈值
TAU_UNC = 0.35    # 不确定度阈值
K_LIMIT = 0.95    # 冲突系数上限，超过即判定为不确定


def bpa_from(scores: tuple[float, float], u: float) -> BPA:
    """由 (正向分, 负向分) 与不确定度构造 BPA。

    仅适用于**判别式**通道（同时在两类中训练过，能给出"是正常"的证据），
    例如语义通道。签名式通道请用 bpa_from_detector。
    """
    a, b = scores
    a = max(0.0, min(1.0, a))
    b = max(0.0, min(1.0, b))
    u = max(0.0, min(0.99, u))
    return BPA(a * (1 - u), b * (1 - u), u)


def bpa_from_detector(a: float, u: float = 0.0) -> BPA:
    """**签名式**通道的 BPA：只能作攻击证词，不能作正常证词。

        m({inj}) = a·(1-u)
        m({ben}) = 0
        m(Θ)     = 1 - a·(1-u)

    为什么必须这样：结构通道是"找攻击特征"的签名检测器。得分低只意味着
    **没有找到攻击证据**，绝不意味着"这是正常内容"——把证据缺失当成缺失的
    证据，是基本概率分配里最典型的建模错误。

    后果有多严重（实测）：若按 bpa_from((a, 1-a), u) 构造，拼音/逐字分隔
    变异体上语义通道给出 p=1.000（确信是注入），而结构通道因为没匹配到
    指令词给出 m({ben})=0.96；Dempster 组合后 m(Θ)=0.96×0=0，不确定度被
    一个"自信的正常证词"抹平，最终判 PASS —— 高确信的攻击证据被低信息量的
    通道单方面否决。这就是 D-S 组合的 Zadeh 悖论。

    改为签名式 BPA 后，结构通道"没找到"只能贡献不确定度，无法否决语义证据。
    """
    a = max(0.0, min(1.0, a)) * (1.0 - max(0.0, min(0.99, u)))
    return BPA(a, 0.0, max(0.0, 1.0 - a))


def combine(m1: BPA, m2: BPA) -> BPA:
    """Dempster 组合规则。"""
    k = m1.inj * m2.ben + m1.ben * m2.inj          # 冲突系数（交集为空的项）
    if k >= K_LIMIT:
        return BPA(0.0, 0.0, 1.0)                  # 严重冲突 -> 完全不确定
    norm = 1.0 - k
    inj = (m1.inj * m2.inj + m1.inj * m2.theta + m1.theta * m2.inj) / norm
    ben = (m1.ben * m2.ben + m1.ben * m2.theta + m1.theta * m2.ben) / norm
    theta = (m1.theta * m2.theta) / norm
    s = inj + ben + theta
    if s > 0:
        inj, ben, theta = inj / s, ben / s, theta / s
    return BPA(inj, ben, theta)


def decide(m: BPA, tau_hi: float = TAU_HI, tau_unc: float = TAU_UNC) -> tuple[Action, str]:
    if m.inj - m.ben > tau_hi:
        return Action.BLOCK, "注入证据占优"
    if m.theta > tau_unc:
        return Action.QUARANTINE, "证据冲突或不足，降级为纯数据"
    if m.ben - m.inj > tau_hi:
        return Action.PASS, "正常证据占优"
    return Action.QUARANTINE, "落入灰区，降级为纯数据"


@dataclass
class FusionConfig:
    tau_hi: float = TAU_HI
    tau_unc: float = TAU_UNC
    struct_weights: dict | None = None


def weighted_average_baseline(s: float, p: float, w: float = 0.5) -> Action:
    """对照实现（消融实验 D 组）：无不确定度、无隔离态。

    用于证明 D-S 融合 + 三态决策相对朴素加权平均的增益。
    """
    return Action.BLOCK if (w * s + (1 - w) * p) > 0.5 else Action.PASS


def make_decision(m: BPA, cfg: FusionConfig, reason: str = "", **kw) -> Decision:
    action, why = decide(m, cfg.tau_hi, cfg.tau_unc)
    return Decision(action=action, bpa=m, reason=reason or why, **kw)
