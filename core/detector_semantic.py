"""M3 语义通道（M 通道）。

默认实现为**纯标准库**的字符 n-gram 对数几率朴素贝叶斯：
    logit = log P(注入|文本) - log P(正常|文本)
    score = sigmoid(logit)
可训练、可校准、可在无 GPU / 无 sklearn 的机器上跑通。

若环境中装有 scikit-learn，可用 SklearnSemantic 替换（TF-IDF + 逻辑回归），
接口完全一致，见本文件底部。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field


# 分布外判定的可信度半饱和常数：绝对 OOV 的 gram 数达到该值时，
# "这确实是分布外"这一断言的可信度约为 0.5。
#
# 标定规则（在 val 划分上）：取召回率不低于 1.0 的前提下，使严格口径误报率
# 最小的 K。实测扫描结果：
#     K=40   recall 1.0000  严格 FPR 0.4444
#     K=55   recall 1.0000  严格 FPR 0.2778
#     K=60   recall 1.0000  严格 FPR 0.1667
#     K=65   recall 1.0000  严格 FPR 0.1111   <- 平台起点
#     K=70   recall 1.0000  严格 FPR 0.1111   <- 取平台中点
#     K=75   recall 1.0000  严格 FPR 0.1111   <- 平台终点
#     K=80   recall 0.9740  严格 FPR 0.1111   <- 开始漏检
# 取 70 而不是 65，是为了落在稳定区间中部而不是贴在边界上。
OOD_CRED_K = 70


@dataclass
class SemScore:
    p: float            # 注入概率
    u_ood: float        # 分布外不确定度
    nearest_sim: float  # 与训练分布的最近相似度
    seg_probs: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"p": round(self.p, 4), "u_ood": round(self.u_ood, 4),
                "nearest_sim": round(self.nearest_sim, 4)}


def _norm_n(seq) -> tuple[int, ...]:
    """把 n 的写法统一成元组：3 -> (3,)，[2,3] -> (2,3)。"""
    if isinstance(seq, int):
        return (seq,)
    return tuple(seq)


def _ngrams(text: str, n=3) -> list[str]:
    """字符 n-gram，忽略空白；对编码混淆与同形字有一定鲁棒性。

    默认同时取 2-gram 与 3-gram。只用 3-gram 时，短文本（如"心理咨询怎么预约？"）
    的 3-gram 在训练集中几乎必然是未登录词，模型对它"没有任何信息"，
    log-odds 直接退化为先验，判不出良性。加入 2-gram 后短文本也能共享
    "怎么""多少""哪里""预约"这类通用片段，泛化到未见问句的能力明显改善。
    """
    t = re.sub(r"\s+", "", text.lower())
    if not t:
        return []
    out: list[str] = []
    for k in _norm_n(n):
        if len(t) < k:
            out.append(t)
            continue
        out.extend(t[i:i + k] for i in range(len(t) - k + 1))
    return out


class SemanticModel:
    """纯 Python 可训练语义通道。"""

    def __init__(self, n=(2, 3), alpha: float = 0.3,
                 balanced_prior: bool = True) -> None:
        self.n = _norm_n(n)
        self.alpha = alpha
        # 先验是否强制均衡。默认 True，理由见 fit() 中的说明。
        self.balanced_prior = balanced_prior
        self.pos: dict[str, float] = {}     # log P(token | 注入)
        self.neg: dict[str, float] = {}     # log P(token | 正常)
        self.prior = 0.0                    # log odds 先验
        self.vocab: set[str] = set()
        self.fitted = False
        self._pos_feat: list[set[str]] = []
        self._neg_feat: list[set[str]] = []
        self.ref_vocab: set[str] | None = None   # OOD 参考词表（首次训练时冻结）

    # ---- 训练 ----

    def fit(self, texts: list[str], labels: list[int],
            freeze_ref: bool = True) -> "SemanticModel":
        pos_counts: dict[str, int] = {}
        neg_counts: dict[str, int] = {}
        n_pos = n_neg = 0
        self._pos_feat, self._neg_feat = [], []

        for text, y in zip(texts, labels):
            grams = set(_ngrams(text, self.n))
            if y == 1:
                n_pos += 1
                self._pos_feat.append(grams)
                for g in grams:
                    pos_counts[g] = pos_counts.get(g, 0) + 1
            else:
                n_neg += 1
                self._neg_feat.append(grams)
                for g in grams:
                    neg_counts[g] = neg_counts.get(g, 0) + 1

        self.vocab = set(pos_counts) | set(neg_counts)
        v = max(1, len(self.vocab))
        tp = sum(pos_counts.values())
        tn = sum(neg_counts.values())
        self._pos_feat = self._pos_feat[:80]     # 仅保留少量样本供诊断，避免 O(n^2)
        self._neg_feat = self._neg_feat[:80]

        for g in self.vocab:
            self.pos[g] = math.log((pos_counts.get(g, 0) + self.alpha) / (tp + self.alpha * v))
            self.neg[g] = math.log((neg_counts.get(g, 0) + self.alpha) / (tn + self.alpha * v))

        # 先验：默认不使用经验类别比，改用均衡先验（log odds = 0）。
        #
        # 为什么：训练集类别比是**数据构造的产物**，不是真实世界的基础发生率。
        # 本项目训练集攻击样本远多于良性样本（实测 266 : 53），经验先验
        # log(267/54)=1.60，等于先验地认为任意文本有 83% 概率是注入。后果是：
        # 短良性问句（如"心理咨询服务怎么预约？"）的字符 3-gram 几乎全是未登录词，
        # _logodds 直接退化为先验值 1.60 -> p=0.72 -> 被判 BLOCK。
        # 工作点（把概率映射到判定）本来就该由 Platt 校准在验证集上学习，
        # 先验再叠一层经验偏置属于重复偏置，且方向不可控。
        if self.balanced_prior:
            self.prior = 0.0
        else:
            self.prior = math.log((n_pos + 1) / (n_neg + 1))
        # OOD 参考词表只在首次训练时确定：后续加入难例不应"抹平"分布外距离，
        # 否则本该被隔离的新型变体会因为词表变大而被判为正常。
        if not (freeze_ref and self.ref_vocab):
            self.ref_vocab = set(self.vocab)
        self.fitted = True
        return self

    # ---- 推理 ----

    def _logodds(self, text: str) -> float:
        grams = set(_ngrams(text, self.n))
        seen = grams & self.vocab
        # 未登录词按均匀分布处理，避免长度越长分越高
        s = self.prior
        for g in seen:
            s += self.pos[g] - self.neg[g]
        return s

    def predict_proba(self, text: str) -> float:
        if not self.fitted:
            return 0.0
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, self._logodds(text)))))

    def _jaccard_sim(self, grams: set[str], pool: list[set[str]]) -> float:
        best = 0.0
        for cand in pool[:400]:          # 采样，避免 O(n^2) 拖慢
            inter = len(grams & cand)
            union = len(grams | cand) or 1
            best = max(best, inter / union)
        return best

    def extend_ref(self, texts, margin: float = 0.0) -> int:
        """把难例中**模型确实学成攻击指示性**的 n-gram 并入分布外参考词表。

        返回并入的 n-gram 数。

        ref_vocab 之所以冻结，是为了避免"加入难例后词表变大，把本该隔离的新型
        变体判成分布内"。但对**已经明确学过**的攻击模式，继续按分布外处理是
        另一方向的错误：模型明明认得出，却因为不确定度被压成 QUARANTINE，
        永远拿不到 BLOCK（实测严格口径 MS 因此卡在 0.81 无法提升）。

        关键约束：只能并入满足 pos[g] - neg[g] > margin 的 n-gram，即模型
        判定为"更像攻击"的那些。**不能**把难例文本的 n-gram 整体并入 ——
        分散稀释类变体里塞满了正常填充文本，那些 n-gram 在模型中仍是良性侧
        证据（pos < neg）。把它们也算作"已学分布"，会让语义分低但不确定度
        也低的变体直接判 PASS，即从"隔离"退化为"放行"（实测每轮新增 4~5 个
        放行退化，比不进化更危险）。
        """
        if self.ref_vocab is None:
            self.ref_vocab = set(self.vocab)
        added = 0
        for t in texts:
            for g in set(_ngrams(t, self.n)):
                if g in self.ref_vocab:
                    continue
                if g in self.pos and g in self.neg:
                    if self.pos[g] - self.neg[g] > margin:
                        self.ref_vocab.add(g)
                        added += 1
        return added

    def ood(self, text: str) -> float:
        """分布外不确定度。

        用"与训练 n-gram 词表的重合度"衡量，复杂度 O(len(grams))，
        相比逐样本 Jaccard（O(n^2)）可在大规模变异体上实时计算。
        重合度越低，说明离训练分布越远，不确定度越高。

        样本量收缩：短文本的 gram 集合很小，重合率的分母也小，估计方差极大。
        直接用它当分布外信号会产生系统性误判 —— 实测所有 10 字左右的良性问句
        （"心理咨询服务怎么预约？"）u_ood 一律顶到上限 0.60，被全部隔离，
        严格口径误报率飙到 44.44%。十几个 gram 不足以断言"分布外"。

        收缩必须同时看**比例**和**绝对数量**：
            u = 1.5 · (n_oov / n) · (n_oov / (n_oov + K))
        只乘比例不行（短文本比例天然接近 1）；只按总数收缩也不行（长文本里
        少量 OOV 会被放大）。两项相乘后：短文本的绝对 OOV 数少 -> 收缩强；
        长文本即使 OOV 比例低，只要绝对数大 -> 仍保留较高不确定度。
        """
        if not self.fitted:
            return 0.3
        grams = set(_ngrams(text, self.n))
        if not grams:
            return 0.5
        ref = self.ref_vocab or self.vocab
        n = len(grams)
        n_oov = len(grams - ref)
        if n_oov == 0:
            return 0.0
        cred = n_oov / (n_oov + OOD_CRED_K)
        return max(0.0, min(0.6, 1.5 * (n_oov / n) * cred))


# --------------------------------------------------------------------------
# 片段切分与聚合
# --------------------------------------------------------------------------

ALPHA = 0.6   # max 与 mean 的加权系数（需标定）


def split_segments(text: str, max_len: int = 128, overlap: int = 16) -> list[str]:
    if len(text) <= max_len:
        return [text] if text.strip() else []
    step = max(1, max_len - overlap)
    return [text[i:i + max_len] for i in range(0, len(text), step)]


class SemanticChannel:
    """语义通道封装：切分 -> 逐段打分 -> 聚合 -> 校准 -> OOD。"""

    def __init__(self, model: SemanticModel | None = None, alpha: float = ALPHA) -> None:
        self.model = model or SemanticModel()
        self.alpha = alpha
        self.calib_a, self.calib_b = 1.0, 0.0   # Platt scaling: z = a*logit + b

    def fit(self, texts: list[str], labels: list[int]) -> None:
        self.model.fit(texts, labels)

    def calibrate(self, texts: list[str], labels: list[int], iters: int = 80, lr: float = 0.05):
        """在验证集上做 Platt scaling（梯度下降拟合 a, b）。"""
        if not self.model.fitted:
            return
        a, b = self.calib_a, self.calib_b
        for _ in range(iters):
            ga = gb = 0.0
            for t, y in zip(texts, labels):
                z = a * self.model._logodds(t) + b
                p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
                err = p - y
                ga += err * self.model._logodds(t)
                gb += err
            n = max(1, len(texts))
            a -= lr * ga / n
            b -= lr * gb / n
        self.calib_a, self.calib_b = a, b

    def score(self, text: str) -> SemScore:
        if not self.model.fitted:
            return SemScore(0.0, 0.3, 0.0)
        segs = split_segments(text)
        probs = []
        for s in segs:
            z = self.calib_a * self.model._logodds(s) + self.calib_b
            probs.append(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))))
        if not probs:
            return SemScore(0.0, 0.3, 0.0)
        p = self.alpha * max(probs) + (1 - self.alpha) * (sum(probs) / len(probs))
        return SemScore(p, self.model.ood(text), 1.0, probs)


# --------------------------------------------------------------------------
# 可选的 sklearn 实现（若环境已安装，可在配置中切换）
# --------------------------------------------------------------------------

class SklearnSemantic:
    """接口兼容的 sklearn 实现：TF-IDF(char 1-2gram) + LogisticRegression。

    需要 scikit-learn；未安装时导入本类会抛错，调用方应捕获并回退到
    SemanticChannel。
    """

    def __init__(self, alpha: float = ALPHA) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 2),
                                   max_features=50000, min_df=1)
        self.clf = LogisticRegression(max_iter=1000)
        self.alpha = alpha
        self.fitted = False
        self._X = None

    def fit(self, texts, labels):
        from sklearn.metrics.pairwise import cosine_similarity
        self._X = self.vec.fit_transform(texts)
        self.clf.fit(self._X, labels)
        self.fitted = True
        self._cos = cosine_similarity
        return self

    def score(self, text: str) -> SemScore:
        if not self.fitted:
            return SemScore(0.0, 0.3, 0.0)
        segs = split_segments(text)
        X = self.vec.transform(segs)
        probs = [float(p[1]) for p in self.clf.predict_proba(X)]
        p = self.alpha * max(probs) + (1 - self.alpha) * (sum(probs) / len(probs))
        sim = float(self._cos(X, self._X).max())
        return SemScore(p, max(0.0, min(0.6, 1.0 - sim)), sim, probs)
