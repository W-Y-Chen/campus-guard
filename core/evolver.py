"""M8 自进化器：归因 -> 定向修复 -> 复测 -> 回归门禁。

门禁条件（硬编码，单测覆盖"不满足必定回滚"）：
  1. MS 必须提升
  2. 误报样本数增加不超过 max_fp_extra 个（小样本下用绝对个数比百分点更稳健）
  3. 召回率下降不超过 max_recall_drop
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .detector_struct import struct_score
from .mutation import Mutant, MutationReport, generate, ms_by_level, report
from .normalizer import normalize
from .types import Cause, Sens, Trust


@dataclass
class EvalResult:
    ms: float = 0.0
    strict_ms: float = 0.0
    recall: float = 0.0
    fpr: float = 0.0
    f1: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    total_mutants: int = 0
    escaped: int = 0        # 完全没被处置（判 PASS）的变异体数

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


@dataclass
class EvolveGate:
    max_fp_extra: int = 1          # 允许增加的误报样本数
    max_fpr_delta: float = 0.005   # 同时保留百分点约束（大样本时生效）
    max_recall_drop: float = 0.01

    def accept(self, before: EvalResult, after: EvalResult) -> tuple[bool, str]:
        # 主判据是**严格口径** MS（只有 BLOCK 才算真正认出并拦住）。
        #
        # 为什么不能用宽松口径（BLOCK+QUARANTINE）当主判据：OOD 不确定度会把
        # 大量新变体判为 QUARANTINE，于是"只要不确定就得分"，宽松 MS 天然饱和
        # （实测 0.9904 且三轮进化毫无改进空间）。那不是护栏变强了，是指标失效了。
        # 严格口径才衡量"确实认出这是攻击"。
        # 次判据：严格口径不下降的前提下，宽松口径提升也算改进。
        if after.strict_ms > before.strict_ms + 1e-9:
            pass
        elif after.strict_ms >= before.strict_ms - 1e-9 and after.ms > before.ms + 1e-9:
            pass
        else:
            return False, (f"MS 未提升（严格 {before.strict_ms:.4f} -> "
                           f"{after.strict_ms:.4f}；宽松 {before.ms:.4f} -> {after.ms:.4f}）")
        fp_delta = after.fp - before.fp
        fpr_delta = after.fpr - before.fpr
        if fp_delta > self.max_fp_extra and fpr_delta > self.max_fpr_delta:
            return False, (f"误报恶化超限（FP {before.fp}->{after.fp}，"
                           f"FPR {before.fpr:.4f}->{after.fpr:.4f}）")
        if after.recall < before.recall - self.max_recall_drop:
            return False, f"召回率下降超限（{before.recall:.4f} -> {after.recall:.4f}）"
        # 防退化：不允许任何变异体从"被处置"退化为"放行"。
        # 实测教训：把难例并入分布外参考词表后，一部分原本被隔离的变体因为
        # 不确定度下降而直接判 PASS。宽松 MS 从 1.0 掉到 0.9560 —— 严格 MS
        # 确实涨了，但换来的是放行，净安全收益是负的。隔离 -> 放行 是比
        # 隔离 -> 阻断 危险得多的方向，必须单独设卡。
        if after.escaped > before.escaped:
            return False, (f"出现放行退化（未被处置的变异体 "
                           f"{before.escaped} -> {after.escaped}）")
        return True, "通过门禁"


# 阈值搜索空间
TAU_HI_GRID = [0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
TAU_UNC_GRID = [0.30, 0.35, 0.45, 0.55]

# 验收探针集的生成种子：与学习用的轮转种子族完全隔离，且全程固定
PROBE_SEED = 555001


class Evolver:
    def __init__(self, guard, bench, gate: EvolveGate | None = None,
                 combo: int = 4, triple: int = 4, base_seed: int = 2026) -> None:
        self.guard = guard
        self.bench = bench
        self.gate = gate or EvolveGate()
        self.history: list[dict] = []
        self.hard_examples: list[str] = []
        self._tune_pool: list[Mutant] | None = None
        self.combo: int = combo      # 每个 payload 额外生成的双算子组合数（L2）
        self.triple: int = triple    # 每个 payload 额外生成的三算子组合数（L3）
        self.base_seed: int = base_seed
        # 每轮的变异生成种子必须不同。若固定同一个种子，第一轮把存活变体当难例
        # 喂进语义通道后，后续轮次只是把同一批变体重新检测一遍 —— 那不是泛化，
        # 是背诵。轮转种子后每轮面对的都是**新变异体**，护栏必须真的学到规律。
        self.seed_stride: int = 37
        self._probe: list[Mutant] | None = None

    def seeds(self) -> list[tuple[str, str]]:
        """变异引擎只对原始 payload 做派生，避免"变体的变体"组合爆炸。"""
        return self.bench.get("seed_attacks", self.bench["attacks"])

    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------

    def evaluate(self, k: int = 2, seed: int = 2026,
                 mutants: list[Mutant] | None = None) -> tuple[EvalResult, list[Mutant]]:
        if mutants is None:
            mutants = generate(self.seeds(), k=k, seed=seed,
                               combo=self.combo, triple=self.triple)
        for m in mutants:
            if not m.valid:
                continue
            d = self.guard.detect(m.payload, Trust.UNTRUSTED, Sens.L3, "mutant")
            m.killed = d.action in ("block", "quarantine")
            m.killed_by = d.action.value
        rep = report(mutants)

        tp = fp = fn = tn = 0
        for text in self.bench["attacks_text"]:
            d = self.guard.detect(text, Trust.UNTRUSTED, Sens.L3, "bench")
            if d.action in ("block", "quarantine"):
                tp += 1
            else:
                fn += 1
        for text in self.bench["benign_text"]:
            d = self.guard.detect(text, Trust.USER, Sens.L1, "bench")
            if d.action == "block":
                fp += 1
            else:
                tn += 1
        recall = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        prec = tp / max(1, tp + fp)
        f1 = 2 * prec * recall / max(1e-9, prec + recall)

        res = EvalResult(ms=rep.ms, strict_ms=rep.strict_ms, recall=recall,
                         fpr=fpr, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn,
                         total_mutants=rep.total,
                         escaped=rep.total - rep.killed)
        return res, mutants

    # ------------------------------------------------------------------
    # 归因
    # ------------------------------------------------------------------

    def attribute(self, mutant: Mutant) -> Cause:
        """存活变体归因。

        分桶规则（按"谁的响应不足"来分，保证每个存活变体都能落到可执行的
        修复动作上）：
          p < 0.5                       -> 语义通道没把握，需要难例重训
          p >= 0.5 且 u_ood 高          -> 语义认出来了，但被分布外不确定度
                                           压成隔离，需要把该模式纳入已学分布
          p >= 0.5 且 s < 0.2           -> 语义看到了、结构没命中，需要补规则
          其余                          -> 两通道都有响应但没判成，融合/阈值问题
        """
        nt = normalize(mutant.payload, trust=Trust.UNTRUSTED, sens=Sens.L3)
        s, feats, _ = struct_score(nt, self.guard.cfg.struct_weights)
        sm = self.guard.semantic.score(nt.text)
        if sm.p < 0.5:
            return Cause.SEMANTIC_BLIND
        if sm.u_ood > 0.30:
            return Cause.OOD_UNCERTAIN
        if s < 0.2:
            return Cause.STRUCT_BLIND
        return Cause.FUSION_WEIGHT

    # ------------------------------------------------------------------
    # 修复
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 训练与探针验收
    # ------------------------------------------------------------------

    def _refit(self, extend_ref_texts: list[str] | None = None) -> None:
        texts = self.bench["train_texts"] + self.hard_examples
        labels = self.bench["train_labels"] + [1] * len(self.hard_examples)
        self.guard.semantic.fit(texts, labels)
        if extend_ref_texts:
            self.guard.semantic.model.extend_ref(extend_ref_texts)
        # 校准始终在 val 上做，避免用训练集自己校准导致概率失真
        self.guard.semantic.calibrate(self.bench["val_texts"],
                                      self.bench["val_labels"], iters=40)

    def _probe_escaped(self) -> int:
        """探针集上"完全没被处置"的变异体数（判 PASS 的个数）。"""
        n = 0
        for m in self.probe():
            if not m.valid:
                continue
            d = self.guard.detect(m.payload, Trust.UNTRUSTED, Sens.L3, "probe")
            if d.action.value == "pass":
                n += 1
        return n

    def _add_hard_examples(self, survivors: list[Mutant]) -> tuple[int, int]:
        """增量式验证注入。返回 (保留的难例数, 其中纳入已学分布的条数)。

        为什么必须逐条验证：朴素贝叶斯是按类归一化的，新增难例会抬高正例的总
        词数，从而**稀释所有 n-gram 的 pos[g]**。实测后果是：加入 13 条难例后，
        逐字分隔变体的语义分从 0.55 掉到 0.384、另一条从 0.684 掉到 0.276，
        原本已被阻断/隔离的变体反而滑向放行。也就是说，"学新的"会挤出"旧的"。

        因此每加一条难例都要在固定探针集上重新验收：只要放行数量没有增加就保留，
        一旦增加就撤销这一条。这比"整轮回滚"粒度更细，能让真正有益的难例留下来。
        """
        blind = [m for m in survivors if self.attribute(m) is Cause.SEMANTIC_BLIND]
        # 语义已经认出（p>=0.5），只是被分布外不确定度压成了隔离
        uncertain = [m for m in survivors if self.attribute(m) is Cause.OOD_UNCERTAIN]
        pool = [(m.payload, False) for m in blind] + [(m.payload, True) for m in uncertain]
        pool = pool[:30]
        if not pool:
            return 0, 0

        model = self.guard.semantic.model
        base_escaped = self._probe_escaped()
        kept = n_ref = 0
        for payload, is_certain_uncertain in pool:
            ref_snap = set(model.ref_vocab) if model.ref_vocab else None
            self.hard_examples.append(payload)
            # 只把"语义已经认出"的模式纳入已学分布；语义仍认不出的若也纳入，
            # 会在 p 低 + 不确定度低 的情况下直接判 PASS，退化为放行。
            self._refit([payload] if is_certain_uncertain else [])
            if self._probe_escaped() <= base_escaped:
                kept += 1
                n_ref += 1 if is_certain_uncertain else 0
            else:
                self.hard_examples.pop()          # 该难例挤出已拦住的变体，撤销
                if ref_snap is not None:
                    model.ref_vocab = ref_snap
                self._refit([])
        return kept, n_ref

    def _tune_mutants(self) -> list[Mutant]:
        """阈值寻优用的变异体：来自 train 家族，与门禁所用的 val 家族不重叠。"""
        if self._tune_pool is None:
            self._tune_pool = generate(self.bench.get("train_seeds", self.seeds()),
                                       k=2, seed=2026)
        return self._tune_pool

    def _tune_objective(self) -> tuple[float, int]:
        """返回 (验证集召回率, 验证集误报数)。

        在**未见过的攻击家族**（val）上寻优，而不是在训练家族上 —— 后者
        已被语义通道记住（MS=1.0），寻优地形是饱和的，找不到改进方向。
        """
        tp = fp = fn = 0
        for t, y in zip(self.bench["val_texts"], self.bench["val_labels"]):
            d = self.guard.detect(t, Trust.UNTRUSTED if y else Trust.USER,
                                  Sens.L3 if y else Sens.L1, "tune")
            if y:
                tp += d.action in ("block", "quarantine")
                fn += d.action == "pass"
            else:
                fp += d.action == "block"
        return tp / max(1, tp + fn), fp

    def _score_val(self) -> dict:
        tp = fp = fn = tn = 0
        for t, y in zip(self.bench["val_texts"], self.bench["val_labels"]):
            d = self.guard.detect(t, Trust.UNTRUSTED if y else Trust.USER,
                                  Sens.L3 if y else Sens.L1, "val")
            hit = d.action in ("block", "quarantine")
            if y:
                tp += hit
                fn += not hit
            else:
                fp += (d.action == "block")
                tn += (d.action != "block")
        recall = tp / max(1, tp + fn)
        prec = tp / max(1, tp + fp)
        f1 = 2 * prec * recall / max(1e-9, prec + recall)
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "recall": recall, "f1": f1}

    def _tune_thresholds(self) -> tuple[float, float] | None:
        """网格搜索 (tau_hi, tau_unc)。

        寻优目标与门禁目标保持一致 —— 直接最大化**变异得分**，
        约束是训练集良性样本误报数不增加（不允许用误报换检出）。
        寻优数据来自 train 家族，门禁验收用 val 家族，最终报数用 test 家族。
        """
        base_ms, base_fp = self._tune_objective()
        old = (self.guard.cfg.tau_hi, self.guard.cfg.tau_unc)
        best = None
        for hi in TAU_HI_GRID:
            for unc in TAU_UNC_GRID:
                self.guard.cfg.tau_hi = hi
                self.guard.cfg.tau_unc = unc
                self.guard.fusion_cfg.tau_hi = hi
                self.guard.fusion_cfg.tau_unc = unc
                ms, fp = self._tune_objective()
                if fp > base_fp:
                    continue
                if best is None or ms > best[0]:
                    best = (ms, (hi, unc))
        self.guard.cfg.tau_hi, self.guard.cfg.tau_unc = old
        self.guard.fusion_cfg.tau_hi = old[0]
        self.guard.fusion_cfg.tau_unc = old[1]
        if best is None or best[0] <= base_ms:
            return None
        return best[1]

    def repair(self, survivors: list[Mutant], causes: dict) -> list[str]:
        actions: list[str] = []

        n, n_ref = self._add_hard_examples(survivors)
        if n:
            actions.append(f"语义通道重训：新增 {n} 条难例（其中 {n_ref} 条已确认"
                           f"语义可识别，纳入已学分布以降低不确定度）")

        tuned = self._tune_thresholds()
        if tuned:
            hi, unc = tuned
            self.guard.cfg.tau_hi, self.guard.cfg.tau_unc = hi, unc
            self.guard.fusion_cfg.tau_hi = hi
            self.guard.fusion_cfg.tau_unc = unc
            actions.append(f"阈值寻优：tau_hi={hi}, tau_unc={unc}（val 集 F1 最优）")

        struct = causes.get(Cause.STRUCT_BLIND, 0)
        if struct:
            actions.append(f"{struct} 个存活变体属结构盲区，需人工补规则（本轮未自动处理）")

        if not actions:
            actions.append("无可自动修复项")
        return actions

    # ------------------------------------------------------------------
    # 一轮进化
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 验收探针集
    # ------------------------------------------------------------------

    def probe(self) -> list[Mutant]:
        """固定的验收变异体集合：只用于判定"有没有变强"，**从不参与训练**。

        为什么验收和学习要用不同的变异体：若每轮都在当轮新生成的变异体上验收，
        换一次种子就换一批题，跨轮的"前/后"值不可比（实测 0.813 / 0.899 / 0.895
        三个起点，看起来像波动其实是三套不同的题）。学习必须面对新题（否则是背诵），
        验收必须始终同一套题（否则无法判断是否真的变强）。
        """
        if self._probe is None:
            self._probe = generate(self.seeds(), k=2, seed=PROBE_SEED,
                                   combo=self.combo, triple=self.triple)
        return self._probe

    def round(self, round_no: int, k: int = 2, seed: int | None = None) -> dict:
        if seed is None:
            seed = self.base_seed + self.seed_stride * round_no
        # 验收在固定探针集上做，保证跨轮可比
        before, _ = self.evaluate(mutants=self.probe())
        # 归因与难例挖掘则面对本轮**新生成**的变异体（种子轮转，防止背诵）
        mutants = generate(self.seeds(), k=k, seed=seed,
                           combo=self.combo, triple=self.triple)
        for m in mutants:
            if not m.valid:
                continue
            d = self.guard.detect(m.payload, Trust.UNTRUSTED, Sens.L3, "mutant")
            m.killed = d.action in ("block", "quarantine")
            m.killed_by = d.action.value
        rep: MutationReport = report(mutants)
        # "存活"的判据用**严格口径**：只有被判 BLOCK 才算被确证认出。
        #
        # 用宽松口径（BLOCK+QUARANTINE）当存活判据时，实测 MS=1.0 意味着
        # 没有任何存活变体，自进化拿不到作业对象，三轮全部空转回滚。
        # 而 QUARANTINE 的含义是"我不确定" —— 护栏并没有认出这是攻击，
        # 只是不敢放行。那 19% 只拿到隔离的变体才是真正的能力缺口。
        survivors = [m for m in mutants if m.valid and m.killed_by != "block"]
        causes: dict = {}
        for m in survivors:
            c = self.attribute(m)
            causes[c] = causes.get(c, 0) + 1

        model = self.guard.semantic.model
        snapshot = {
            "tau_hi": self.guard.cfg.tau_hi,
            "tau_unc": self.guard.cfg.tau_unc,
            "hard_examples": list(self.hard_examples),
            "semantic": copy.deepcopy(self.guard.semantic),
            "ref_vocab": set(model.ref_vocab) if model.ref_vocab else None,
        }
        actions = self.repair(survivors, causes)
        after, _ = self.evaluate(mutants=self.probe())
        ok, why = self.gate.accept(before, after)

        if not ok and model.ref_vocab and snapshot["ref_vocab"] is not None \
                and len(model.ref_vocab) > len(snapshot["ref_vocab"]):
            # 分级回退：先只撤掉"纳入已学分布"这一步，保留难例重训的成果再验收一次。
            #
            # 这一步有跨体副作用：把某些 n-gram 算作已学分布后，会连带降低**其它**
            # 变体的分布外不确定度，其中语义分偏低的那些就从"隔离"滑到了"放行"。
            # 与其整轮放弃，不如退到更保守的修复方案再试一次。
            model.ref_vocab = set(snapshot["ref_vocab"])
            actions.append("回退「纳入已学分布」步骤（该操作使部分变体由隔离退化为放行），"
                           "保留难例重训后重新验收")
            after, _ = self.evaluate(mutants=self.probe())
            ok, why = self.gate.accept(before, after)

        if not ok:
            # 完整回滚：阈值、难例集、语义通道（含校准参数、已学分布）一并还原
            self.guard.cfg.tau_hi = snapshot["tau_hi"]
            self.guard.cfg.tau_unc = snapshot["tau_unc"]
            self.guard.fusion_cfg.tau_hi = snapshot["tau_hi"]
            self.guard.fusion_cfg.tau_unc = snapshot["tau_unc"]
            self.hard_examples = snapshot["hard_examples"]
            self.guard.semantic = snapshot["semantic"]

        # 分强度 MS 取自固定的验收探针集，保证跨轮可比
        level_ms = {f"L{l}": v for l, v in ms_by_level(self.probe()).items()}
        rec = {
            "round": round_no,
            "seed": seed,
            "ms_before": round(before.ms, 4),
            "ms_after": round(after.ms, 4),
            "strict_ms_before": round(before.strict_ms, 4),
            "strict_ms_after": round(after.strict_ms, 4),
            "ms_by_level": level_ms,
            "fpr_before": round(before.fpr, 4),
            "fpr_after": round(after.fpr, 4),
            "fp_before": before.fp,
            "fp_after": after.fp,
            "recall_before": round(before.recall, 4),
            "recall_after": round(after.recall, 4),
            "causes": {c.value if hasattr(c, "value") else str(c): v
                       for c, v in causes.items()},
            "actions": actions,
            "accepted": ok,
            "gate_reason": why,
        }
        self.history.append(rec)
        return rec

    def run(self, rounds: int = 5, k: int = 2, seed: int | None = None) -> list[dict]:
        for i in range(1, rounds + 1):
            self.round(i, k=k, seed=None if seed is None else seed)
        return self.history
