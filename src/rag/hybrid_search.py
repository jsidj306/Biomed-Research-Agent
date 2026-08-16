"""
混合检索 — 语义检索 + BM25 + RRF 融合 + BGE-Reranker 重排序。

纯语义检索（向量）擅长「意思相近但用词不同」，BM25 擅长「精确词匹配」
（基因名 / 药物名 / 缩写）。生物医学术语高度专业，纯语义检索会漏掉必须
精确匹配的术语，所以要用 RRF（Reciprocal Rank Fusion）把两路召回融合，
再用 BGE-Reranker 精排。

两段式检索:
    第一段（粗排）: 向量 + BM25 各召回 N 篇 → RRF 融合出候选集
    第二段（精排）: BGE-Reranker 对候选集打分重排 Top-K

评测指标（第3-4周硬性验收线）:
    - MRR (Mean Reciprocal Rank)      > 0.70
    - Recall@10                        > 0.85

使用示例:
    from src.rag.hybrid_search import HybridRetriever, BM25Index

    corpus = ["CAR-T cells target CD19 antigen", "mRNA vaccine for cancer", ...]
    retriever = HybridRetriever(corpus)

    def semantic_fn(query):
        # 生产环境：调用 ChromaDB 语义检索，返回文档下标排序
        return [0, 2, 1]

    results = retriever.search("CAR-T CD19", semantic_fn, k=3)
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


# ── 分词 ──────────────────────────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    """小写 + 按非字母数字分词（适合英文生物医学文本）。"""
    return re.findall(r"[a-z0-9]+", text.lower())


# ── BM25 稀疏检索 ─────────────────────────────────────────────────────────


class BM25Index:
    """纯 Python BM25 稀疏检索索引。

    对语料建立倒排统计（文档频率 / 平均文档长度），支持关键词精确匹配。

    Attributes:
        documents: 语料文本列表
        k1:        BM25 词频饱和参数（默认 1.5）
        b:         长度归一化参数（默认 0.75）
    """

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        """构建 BM25 索引。

        Args:
            documents: 语料文本列表
            k1:        BM25 k1 参数
            b:         BM25 b 参数
        """
        self.documents = list(documents)
        self.k1 = k1
        self.b = b

        self._corpus = [_tokenize(d) for d in self.documents]
        self._doc_len = [len(tokens) for tokens in self._corpus]
        self._n = len(self._corpus)
        self._avgdl = (sum(self._doc_len) / self._n) if self._n else 0.0

        # 文档频率 df[term] = 包含该词的文档数
        self._df: Counter = Counter()
        for tokens in self._corpus:
            for term in set(tokens):
                self._df[term] += 1

        logger.info("BM25 index built: %d docs, avgdl=%.1f", self._n, self._avgdl)

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """BM25 检索，返回 (文档下标, 分数) 按分数降序。

        Args:
            query: 查询文本
            k:     返回结果数

        Returns:
            List[(int, float)] — (文档下标, BM25 分数)
        """
        q_tokens = _tokenize(query)
        if not q_tokens or self._n == 0:
            return []

        scored: list[tuple[int, float]] = []
        for idx, tokens in enumerate(self._corpus):
            tf = Counter(tokens)
            dl = self._doc_len[idx]
            score = 0.0
            for term in q_tokens:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                idf = self._idf(term)
                if self._avgdl > 0:
                    denom = f + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                else:
                    denom = f + self.k1
                score += idf * f * (self.k1 + 1) / denom
            if score > 0:
                scored.append((idx, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


# ── RRF 融合 ──────────────────────────────────────────────────────────────


def reciprocal_rank_fusion(rankings: list[list], k: int = 60) -> list:
    """Reciprocal Rank Fusion — 倒数排名融合多路检索结果。

    两路检索各给一个排序，融合方式是把「名次倒数相加」:
        RRF_score(doc) = Σ 1/(k + rank_i(doc))
    排第 1 的得 1/61，排第 10 的得 1/70，融合分越高越靠前。
    它不依赖具体分数、实现极简，是多路召回融合的标准做法。

    Args:
        rankings: 多路排序结果，每个元素是按相关度降序的文档标识列表
        k:        RRF 平滑参数（默认 60）

    Returns:
        list — 融合后的文档标识列表（按融合分降序）
    """
    scores: dict[Any, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores, key=scores.get, reverse=True)


# ── BGE-Reranker 重排序 ───────────────────────────────────────────────────


class HybridReranker:
    """BGE-Reranker 交叉编码器重排序封装。

    粗召回后精排：用 query + 每篇候选的「成对」打分，比纯向量检索准很多。
    模型懒加载（首次 rerank 才下载），便于测试时注入 mock。

    Attributes:
        model_name: HuggingFace 模型名（默认 BAAI/bge-reranker-base）
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        """懒加载 CrossEncoder 模型。"""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker: %s", self.model_name)
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(self, query: str, documents: list[str], top_k: int = 10) -> list[dict]:
        """对候选文档集精排。

        Args:
            query:     查询文本
            documents: 候选文档文本列表
            top_k:     返回结果数

        Returns:
            List[dict] — [{"content", "score"}, ...] 按 score 降序
        """
        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [
            {"content": doc, "score": float(score)}
            for doc, score in ranked[:top_k]
        ]


# ── 混合检索编排 ─────────────────────────────────────────────────────────


class HybridRetriever:
    """混合检索编排：BM25 + 语义 → RRF 融合 → 可选 Reranker 精排。

    Attributes:
        corpus:   语料文本列表
        bm25:     BM25 稀疏检索索引
        reranker: 可选的重排序器
    """

    def __init__(
        self,
        corpus: list[str],
        reranker: HybridReranker | None = None,
    ):
        """初始化混合检索器。

        Args:
            corpus:   语料文本列表
            reranker: 可选的重排序器（None 则跳过精排）
        """
        self.corpus = list(corpus)
        self.bm25 = BM25Index(self.corpus)
        self.reranker = reranker
        logger.info("HybridRetriever ready: %d docs, reranker=%s",
                    len(self.corpus), bool(reranker))

    def search(
        self,
        query: str,
        semantic_fn: Callable[[str], list[int]],
        k: int = 10,
    ) -> list[dict]:
        """混合检索：语义 + BM25 → RRF → (Reranker)。

        Args:
            query:       查询文本
            semantic_fn: 语义检索函数，输入 query 返回按相关度降序的文档下标列表
            k:           最终返回结果数

        Returns:
            List[dict] — [{"index", "content", ("score")}, ...]
        """
        sem_ranking = list(semantic_fn(query))
        bm25_ranking = [idx for idx, _ in self.bm25.search(query, k=len(self.corpus))]

        fused = reciprocal_rank_fusion([sem_ranking, bm25_ranking])

        # 候选池取融合后的前 2k，交给 reranker 精排
        candidates = fused[: k * 2]

        if self.reranker is not None:
            docs = [self.corpus[i] for i in candidates]
            reranked = self.reranker.rerank(query, docs, top_k=k)
            # 用 content 反查下标，保持可追踪
            content_to_idx = {self.corpus[i]: i for i in candidates}
            return [
                {"index": content_to_idx[r["content"]], "content": r["content"], "score": r["score"]}
                for r in reranked
                if r["content"] in content_to_idx
            ]

        return [
            {"index": i, "content": self.corpus[i]}
            for i in fused[:k]
        ]


# ── 检索评测 ──────────────────────────────────────────────────────────────


def mean_reciprocal_rank(
    eval_set: list[tuple[str, set]],
    search_fn: Callable[[str], list],
    k: int = 10,
) -> float:
    """计算 MRR（Mean Reciprocal Rank）。

    衡量「第一条相关结果排在第几位」，越靠前越好。

    Args:
        eval_set: [(query, 相关文档标识集合), ...]
        search_fn: 检索函数，query → 按相关度降序的文档标识列表
        k:        截断位置

    Returns:
        float — 平均倒数排名 (0-1)
    """
    if not eval_set:
        return 0.0

    rr_sum = 0.0
    for query, relevant in eval_set:
        ranked = list(search_fn(query))[:k]
        for rank, doc_id in enumerate(ranked):
            if doc_id in relevant:
                rr_sum += 1.0 / (rank + 1)
                break
    return rr_sum / len(eval_set)


def recall_at_k(
    eval_set: list[tuple[str, set]],
    search_fn: Callable[[str], list],
    k: int = 10,
) -> float:
    """计算 Recall@k。

    衡量「前 k 条里捞回了多少相关文档」。

    Args:
        eval_set: [(query, 相关文档标识集合), ...]
        search_fn: 检索函数
        k:        截断位置

    Returns:
        float — 平均召回率 (0-1)
    """
    if not eval_set:
        return 0.0

    recall_sum = 0.0
    for query, relevant in eval_set:
        if not relevant:
            continue
        ranked = set(list(search_fn(query))[:k])
        recall_sum += len(ranked & relevant) / len(relevant)
    return recall_sum / len(eval_set)
