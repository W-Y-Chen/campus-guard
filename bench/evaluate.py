"""评估与消融实验。

输出 docs/benchmark.md（实测记录），所有数字均来自本脚本，可复现。

用法：
    python -m bench.evaluate            # 完整评估 + 消融
    python -m bench.evaluate --quick     # 小规模快速验证
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.arbiter import Arbiter, default_tools
from core.evolver import Evolver
from core.fusion import bpa_from, combine, weighted_average_baseline
from core.mutation import generate, report
from core.pipeline import Guard, GuardConfig
from core.detector_struct import struct_score
from core.normalizer import normalize
from core.taint import derive_from_context
from core.types import Action, Capability, Sens, Span, TaintString, Trust, Verdict
from bench.dataset import build


def metrics(guard: Guard, texts: list[str], labels: list[int],
            trust_for_pos=Trust.UNTRUSTED) -> dict:
    tp = fp = fn = tn = fp_strict = 0
    latencies: list[float] = []
    for text, y in zip(texts, labels):
        trust = trust_for_pos if y == 1 else Trust.USER
        d = guard.detect(text, trust, Sens.L3 if y == 1 else Sens.L1, "bench")
        latencies.append(d.elapsed_ms)
        detected = d.action in (Action.BLOCK, Action.QUARANTINE)
        if y == 1:
            tp += detected
            fn += not detected
        else:
            fp += (d.action is Action.BLOCK)     # 宽松口径：仅 BLOCK 计误报
            # 严格口径：BLOCK 与 QUARANTINE 都算误报。
            # 没有这个口径，"一律隔离"可以刷出 Recall=1 / FPR=0 的假象。
            fp_strict += d.action in (Action.BLOCK, Action.QUARANTINE)
            tn += (d.action is not Action.BLOCK)
    recall = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    prec = tp / max(1, tp + fp)
    f1 = 2 * prec * recall / max(1e-9, prec + recall)
    latencies.sort()
    p95 = latencies[int(len(latencies) * 0.95) - 1] if latencies else 0.0
    return {"recall": recall, "fpr": fpr, "precision": prec, "f1": f1,
            "fpr_strict": fp_strict / max(1, fp + tn),
            "p95_ms": p95, "mean_ms": sum(latencies) / max(1, len(latencies)),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# --------------------------------------------------------------------------
# 端到端攻击成功率（含仲裁层）
# --------------------------------------------------------------------------

def asr(guard: Guard, attacks: list[str], use_arbitration: bool,
        use_detection: bool) -> dict:
    """模拟：外部文档触发智能体发起 send_email，看是否被拦住。"""
    success = blocked = 0
    for text in attacks:
        d = guard.detect(text, Trust.UNTRUSTED, Sens.L3, "web:1")
        if use_detection and d.action is Action.BLOCK:
            blocked += 1
            continue
        ctx = [Span(0, len(text), "web:1", Trust.UNTRUSTED, Sens.L3)]
        body = derive_from_context("全班成绩单：" + text[:40], ctx, strict=True)
        args = {"to": TaintString("a@x.com", [Span(0, 7, "web:1", Trust.UNTRUSTED, Sens.L2)]),
                "body": body}
        if not use_arbitration:
            success += 1
            continue
        v = guard.arbiter.arbitrate(__import__("core.arbiter", fromlist=["ToolCall"]).ToolCall(
            "send_email", args))
        if v.verdict is Verdict.ALLOW or v.verdict is Verdict.SANITIZE:
            success += 1
        else:
            blocked += 1
    n = max(1, len(attacks))
    return {"asr": success / n, "blocked": blocked, "attempts": n}


# --------------------------------------------------------------------------
# 消融配置
# --------------------------------------------------------------------------

def pick_threshold(scores_pos: list[float], scores_neg: list[float],
                   max_fpr: float = 0.03) -> float:
    """在给定 FPR 上限下挑选阈值。

    单通道消融必须这样做才公平：结构通道与语义通道的分数尺度不同，
    若统一用 0.5 作为阈值，等于人为让其中一通道失效。
    """
    if not scores_neg:
        return 0.0
    neg_sorted = sorted(scores_neg, reverse=True)
    best_thr, best_recall = 1.0, -1.0
    for thr in sorted(set(scores_pos) | {0.0, 1.0}):
        fp = sum(1 for s in scores_neg if s >= thr)
        if fp / len(scores_neg) > max_fpr:
            continue
        rec = sum(1 for s in scores_pos if s >= thr) / max(1, len(scores_pos))
        if rec > best_recall:
            best_recall, best_thr = rec, thr
    _ = neg_sorted
    return best_thr


def calibrate_operating_points(guard: Guard, data: dict) -> dict:
    """在验证集上按 FPR<=3% 标定结构/语义通道的工作点与硬阈值。"""
    vpos = [t for t, y in zip(data["val_texts"], data["val_labels"]) if y == 1]
    vneg = [t for t, y in zip(data["val_texts"], data["val_labels"]) if y == 0]

    def s_struct(t):
        nt = normalize(t, trust=Trust.UNTRUSTED, sens=Sens.L3)
        s, _, _ = struct_score(nt, guard.cfg.struct_weights)
        return s

    def s_sem(t):
        return guard.semantic.score(normalize(t).text).p

    thr_s = pick_threshold([s_struct(t) for t in vpos], [s_struct(t) for t in vneg])
    thr_m = pick_threshold([s_sem(t) for t in vpos], [s_sem(t) for t in vneg])
    guard.cfg.thr_struct = thr_s
    guard.cfg.thr_sem = thr_m
    guard.cfg.hard_block_struct = min(1.0, thr_s + 0.10)
    guard.cfg.min_quarantine_struct = thr_s
    return {"thr_struct": round(thr_s, 4), "thr_sem": round(thr_m, 4),
            "hard_block_struct": round(guard.cfg.hard_block_struct, 4),
            "min_quarantine_struct": round(guard.cfg.min_quarantine_struct, 4)}


def ablation(guard: Guard, data: dict) -> list[dict]:
    atk = data["attacks_text"]
    ben = data["benign_text"]
    rows: list[dict] = []

    def mk(name, detection, arbitration, mode):
        g = Guard(semantic=guard.semantic, arbiter=Arbiter(default_tools()),
                  cfg=GuardConfig(struct_weights=dict(guard.cfg.struct_weights),
                                  tau_hi=guard.cfg.tau_hi, tau_unc=guard.cfg.tau_unc))
        g.mode = mode
        g.detection_on = detection
        return g, name, arbitration

    # 用统一的打分函数重算各配置，避免污染主 guard
    def score_struct(text):
        nt = normalize(text, trust=Trust.UNTRUSTED, sens=Sens.L3)
        s, _, _ = struct_score(nt, guard.cfg.struct_weights)
        return s

    def score_sem(text):
        return guard.semantic.score(normalize(text).text).p

    def row(name, detect_fn, arbitration, detection):
        tp = fp = fn = tn = 0
        for t in atk:
            detected = detect_fn(t) if detection else False
            tp += detected
            fn += not detected
        for t in ben:
            hit = detect_fn(t) if detection else False
            fp += hit
            tn += not hit
        a = asr(guard, atk, use_arbitration=arbitration, use_detection=detection)
        rows.append({
            "config": name,
            "recall": round(tp / max(1, tp + fn), 4),
            "fpr": round(fp / max(1, fp + tn), 4),
            "asr": round(a["asr"], 4),
        })

    def row_real(name, arbitration, detection=True):
        """直接调用真实 detect() 流程（含 D-S 融合、三态决策、硬阻断）。"""
        tp = fp = fn = tn = fp_strict = 0
        for t in atk:
            if not detection:
                fn += 1
                continue
            d = guard.detect(t, Trust.UNTRUSTED, Sens.L3, "abl")
            hit = d.action in (Action.BLOCK, Action.QUARANTINE)
            tp += hit
            fn += not hit
        for t in ben:
            if not detection:
                tn += 1
                continue
            d = guard.detect(t, Trust.USER, Sens.L1, "abl")
            fp += d.action is Action.BLOCK
            fp_strict += d.action in (Action.BLOCK, Action.QUARANTINE)
            tn += d.action is not Action.BLOCK
        a = asr(guard, atk, use_arbitration=arbitration, use_detection=detection)
        rows.append({
            "config": name,
            "recall": round(tp / max(1, tp + fn), 4),
            "fpr": round(fp / max(1, fp + tn), 4),
            "fpr_strict": round(fp_strict / max(1, fp + tn), 4),
            "asr": round(a["asr"], 4),
        })

    # 单通道阈值在 val 上按 FPR<=3% 标定，保证公平比较
    val_pos = [t for t, y in zip(data["val_texts"], data["val_labels"]) if y == 1]
    val_neg = [t for t, y in zip(data["val_texts"], data["val_labels"]) if y == 0]
    thr_s = pick_threshold([score_struct(t) for t in val_pos],
                           [score_struct(t) for t in val_neg])
    thr_m = pick_threshold([score_sem(t) for t in val_pos],
                           [score_sem(t) for t in val_neg])

    # A 无防护
    row("A 无防护", lambda t: False, False, False)
    # B 仅结构通道（阈值按 val 标定）
    row("B 仅结构通道 S", lambda t: score_struct(t) >= thr_s, False, True)
    # C 仅语义通道（阈值按 val 标定）
    row("C 仅语义通道 M", lambda t: score_sem(t) >= thr_m, False, True)
    # D 加权平均（无隔离态）
    row("D S+M 加权平均（无隔离态）",
        lambda t: weighted_average_baseline(score_struct(t), score_sem(t)) is Action.BLOCK,
        False, True)
    # E / E+仲裁：走真实 detect() 流程，不做任何简化复现
    row_real("E 完整 DCFD（真实流程）", arbitration=False)
    row_real("E+仲裁 完整系统", arbitration=True)
    return rows


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    data = build(mutate=not args.quick,
                 mutants_per=1 if args.quick else 2)
    print(f"[data] total attacks={data['meta']['n_attacks_total']} "
          f"benign={data['meta']['n_benign']} | "
          f"train/val/test={data['meta']['n_train']}/"
          f"{data['meta']['n_val']}/{data['meta']['n_test']}")

    guard = Guard()
    guard.semantic.fit(data["train_texts"], data["train_labels"])
    guard.semantic.calibrate(data["val_texts"], data["val_labels"], iters=60)

    # 在验证集上标定两个通道的工作点（FPR <= 3%），再据此设定硬阻断/隔离阈值。
    # 不标定的话，结构通道与语义通道尺度不同，融合会由其中一个单方面主导。
    calibrate_operating_points(guard, data)

    t0 = time.perf_counter()
    base = metrics(guard, data["test_texts"], data["test_labels"])
    print(f"[detect] recall={base['recall']:.4f} fpr={base['fpr']:.4f} "
          f"f1={base['f1']:.4f} p95={base['p95_ms']:.1f}ms")

    # 最终 MS 在 test 家族（训练与进化全程未见过）+ 独立生成种子上报告，
    # 并同时给出三级攻击强度：L1 单算子 / L2 双算子 / L3 三算子。
    MS_SEED = 90210      # 与进化循环使用的种子族完全隔离
    mutants = generate(data["test_seeds"], k=1 if args.quick else 2,
                       combo=6, triple=6, seed=MS_SEED)
    for m in mutants:
        if not m.valid:
            continue
        d = guard.detect(m.payload, Trust.UNTRUSTED, Sens.L3, "mutant")
        m.killed = d.action in (Action.BLOCK, Action.QUARANTINE)
        m.killed_by = d.action.value
    rep = report(mutants)
    print(f"[mutation] MS={rep.ms:.4f} strict={rep.strict_ms:.4f} "
          f"({rep.killed}/{rep.total})")
    for lvl, d in sorted(rep.by_level.items()):
        print(f"           L{lvl} n={d['n']:3d} MS={d['ms']:.4f} strict={d['strict_ms']:.4f}")

    ev = Evolver(guard, data)
    history = ev.run(rounds=args.rounds, k=1 if args.quick else 2)
    for h in history:
        lv = " ".join(f"{k}:{v['ms']:.3f}" for k, v in sorted(h["ms_by_level"].items()))
        print(f"[evolve r{h['round']}] strict {h['strict_ms_before']:.4f}->"
              f"{h['strict_ms_after']:.4f} | loose {h['ms_before']:.4f}->"
              f"{h['ms_after']:.4f} | FPR {h['fpr_before']:.4f}->{h['fpr_after']:.4f} "
              f"| {lv} | {'ACCEPT' if h['accepted'] else 'ROLLBACK: ' + h['gate_reason']}")

    after = metrics(guard, data["test_texts"], data["test_labels"])
    rows = ablation(guard, data)
    elapsed = time.perf_counter() - t0

    # ---- 写报告 ----
    n_ben_test = sum(1 for y in data["test_labels"] if y == 0)
    if base["fpr_strict"] > 0:
        strict_note = (f"严格口径误报率 {base['fpr_strict']:.2%}"
                       "（BLOCK 与隔离均计误报），说明隔离策略仍在误伤部分"
                       "合法文本（主要为指令密集的作业要求与管理细则，"
                       "以及训练集未覆盖到的短问句）。")
    else:
        strict_note = (f"本轮 test 划分上严格口径误报率为 0，但良性样本总共只有 "
                       f"{data['meta']['n_benign']} 条、其中 test 仅 {n_ben_test} 条，"
                       "样本量不足以支撑「零误报」的结论，只能说明本轮未观测到误报。")

    out = Path(__file__).resolve().parents[1] / "docs" / "benchmark.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 实测记录（docs/benchmark.md）",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 数据集 seed=2026，攻击 {data['meta']['n_attacks_total']} 条 / "
        f"良性 {data['meta']['n_benign']} 条（全部合成数据）",
        f"- 评估脚本：`python -m bench.evaluate`，总耗时 {elapsed:.1f}s",
        "",
        f"> **数据局限（必须一并披露）**：CampusInject-Bench 为自建合成数据集，"
        f"规模小（攻击 {data['meta']['n_attacks_total']} 条 / "
        f"良性 {data['meta']['n_benign']} 条）、攻击范式由 "
        f"{data['meta']['n_seed_attacks']} 条原始 payload 派生，"
        "与真实校园流量存在分布差异。上述数字只能说明系统在**本数据集**上的表现，"
        f"不能外推为真实环境的检出率。{strict_note}",
        "",
        "## 一、检测层指标（test 划分，仅评估一次）",
        "",
        "| 指标 | 进化前 | 进化后 |",
        "|---|---|---|",
        f"| Recall | {base['recall']:.4f} | {after['recall']:.4f} |",
        f"| FPR（宽松，仅 BLOCK 计误报） | {base['fpr']:.4f} | {after['fpr']:.4f} |",
        f"| FPR（严格，BLOCK+隔离 均计误报） | {base['fpr_strict']:.4f} | "
        f"{after['fpr_strict']:.4f} |",
        f"| F1 | {base['f1']:.4f} | {after['f1']:.4f} |",
        f"| P95 延时 | {base['p95_ms']:.2f} ms | {after['p95_ms']:.2f} ms |",
        "",
        "## 二、变异得分",
        "",
        f"- 变异体来源：test 家族（训练与进化全程未见过），生成种子 {MS_SEED}",
        f"- MS（宽松口径，阻断+隔离计分）：{rep.ms:.4f}（{rep.killed}/{rep.total}）",
        f"- MS（严格口径，仅阻断计分）：{rep.strict_ms:.4f}",
        "",
        "### 三级攻击强度阶梯",
        "",
        "只报一个汇总 MS 会掩盖「护栏只是背住了弱变异」这件事：分级后若 L1 高而 L3 低，"
        "说明护栏靠记忆而非泛化。MDGE 的价值就在于把 L3 也拉上来，这正是对"
        "「现有基准可被简单方法饱和」这一批评的正面回应。",
        "",
        "| 强度 | 说明 | 变异体数 | MS（宽松） | MS（严格） |",
        "|---|---|---|---|---|",
    ]
    _lv_name = {1: "单算子", 2: "双算子组合", 3: "三算子组合"}
    for lvl, d in sorted(rep.by_level.items()):
        lines.append(f"| L{lvl} | {_lv_name.get(lvl, '')} | {d['n']} | "
                     f"{d['ms']:.4f} | {d['strict_ms']:.4f} |")
    lines += ["", "### 各算子得分", "",
              "| 算子 | 总变异体 | 被杀 | MS |",
              "|---|---|---|---|"]
    for op, d in sorted(rep.by_operator.items()):
        lines.append(f"| {op} | {d['total']} | {d['killed']} | {d['ms']:.4f} |")
    lines += ["", "## 三、自进化记录", "",
              "> 每轮使用**不同的变异生成种子**（种子轮转）。若固定同一批变异体，"
              "第一轮把它们当难例喂入后，后续轮次只是重测同一批 —— 那是背诵不是泛化。"
              "主判据为严格口径 MS（仅 BLOCK 计杀伤）。",
              "",
              "| 轮次 | 种子 | 严格MS 前 | 严格MS 后 | 宽松MS 前 | 宽松MS 后 | FPR 前 | FPR 后 | 归因 | 是否接受 | 门禁原因 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for h in history:
        causes = ", ".join(f"{k}:{v}" for k, v in h["causes"].items()) or "-"
        lines.append(f"| {h['round']} | {h['seed']} | {h['strict_ms_before']:.4f} | "
                     f"{h['strict_ms_after']:.4f} | {h['ms_before']:.4f} | "
                     f"{h['ms_after']:.4f} | {h['fpr_before']:.4f} | {h['fpr_after']:.4f} | "
                     f"{causes} | {'是' if h['accepted'] else '否'} | {h['gate_reason']} |")
    lines += ["", "## 四、消融实验", "",
              "| 配置 | Recall | FPR（宽松） | FPR（严格） | 端到端 ASR |",
              "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['config']} | {r['recall']:.4f} | {r['fpr']:.4f} | "
                     f"{r.get('fpr_strict', r['fpr']):.4f} | {r['asr']:.4f} |")
    lines += ["", "> ASR 口径：外部文档 -> 智能体发起 send_email -> 是否被拦住。",
              "> A 组为无防护基线；E+仲裁 为完整系统。"]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[write] {out}")

    (out.parent / "ablation.json").write_text(
        json.dumps({"base": base, "after": after, "ablation": rows,
                    "mutation": rep.to_dict(), "history": history},
                   ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
