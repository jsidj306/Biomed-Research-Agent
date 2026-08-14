"""
混合检索（BM25 + RRF + Reranker）— 单元测试。

运行: uv run pytest tests/test_hybrid_search.py -v

不下载真实重排模型，BGE-Reranker 用 mock 验证封装逻辑。
"""

from unittest.mock import MagicMock

import pytest

from src.rag.hybrid_search import (
    BM25Index,
    HybridRetriever,
    HybridReranker,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank_fusion,
)


# ── BM25 测试 ─────────────────────────────────────────────────────────────


class TestBM25:
    """测试 BM25 稀疏检索。"""

    def test_build_and_search_keyword_match(self):
        """BM25 应命中含精确关键词的文档。"""
        corpus = [
            "CAR-T cells target the CD19 antigen on B cells",
            "mRNA vaccines encode spike protein",
            "CRISPR-Cas9 edits DNA sequences",
        ]
        index = BM25Index(corpus)

        results = index.search("CAR-T CD19", k=2)

        assert len(results) >= 1
        assert results[0][0] == 0  # 第一篇含 CAR-T 和 CD19

    def test_search_returns_empty_for_no_match(self):
        """无关键词命中时返回空列表。"""
        index = BM25Index(["alpha beta gamma"])
        assert index.search("zzzz") == []

    def test_search_ranked_by_score(self):
        """分数应按降序排列。"""
        corpus = [
            "COVID-19 vaccine mRNA",
            "COVID-19 vaccine mRNA efficacy trial",
            "unrelated text about cats",
        ]
        index = BM25Index(corpus)
        results = index.search("COVID-19 vaccine mRNA", k=3)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


# ── RRF 融合测试 ──────────────────────────────────────────────────────────


class TestRRF:
    """测试 Reciprocal Rank Fusion。"""

    def test_fusion_ranks_common_doc_first(self):
        """两路都排第一的文档应融合后居首。"""
        fused = reciprocal_rank_fusion([["a", "b"], ["a", "c"]])
        assert fused[0] == "a"

    def test_fusion_includes_docs_from_both_lists(self):
        """融合结果应包含两路的文档。"""
        fused = reciprocal_rank_fusion([["a"], ["b"]])
        assert set(fused) == {"a", "b"}

    def test_fusion_higher_rank_wins(self):
        """排名靠前的文档融合分更高。"""
        fused = reciprocal_rank_fusion([["x", "y"]])
        assert fused[0] == "x"


# ── Reranker 测试 ─────────────────────────────────────────────────────────


class TestReranker:
    """测试 BGE-Reranker 封装（mock 模型）。"""

    def test_rerank_sorts_by_score(self):
        """rerank 应返回按分数降序的结果。"""
        reranker = HybridReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.9, 0.1]
        reranker._model = mock_model

        docs = ["doc A", "doc B", "doc C"]
        results = reranker.rerank("query", docs, top_k=2)

        assert len(results) == 2
        assert results[0]["content"] == "doc B"
        assert results[0]["score"] == 0.9

    def test_rerank_empty_docs(self):
        reranker = HybridReranker()
        assert reranker.rerank("query", [], top_k=10) == []


# ── HybridRetriever 端到端测试 ────────────────────────────────────────────


class TestHybridRetriever:
    """测试混合检索编排（mock 语义检索）。"""

    def test_hybrid_search_without_reranker(self):
        """无 reranker 时应返回 RRF 融合后的结果。"""
        corpus = ["CAR-T CD19 therapy", "mRNA vaccine", "CRISPR editing"]
        retriever = HybridRetriever(corpus)

        def semantic_fn(query):
            return [0, 2, 1]

        results = retriever.search("CAR-T", semantic_fn, k=3)

        assert len(results) >= 1
        assert "index" in results[0]
        assert "content" in results[0]

    def test_hybrid_search_with_mock_reranker(self):
        """带 reranker 时应返回精排结果（含 score）。"""
        corpus = ["CAR-T CD19 therapy", "mRNA vaccine", "CRISPR editing"]
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {"content": "CAR-T CD19 therapy", "score": 0.95},
            {"content": "mRNA vaccine", "score": 0.4},
        ]
        retriever = HybridRetriever(corpus, reranker=mock_reranker)

        def semantic_fn(query):
            return [0, 1, 2]

        results = retriever.search("CAR-T", semantic_fn, k=2)

        assert results[0]["content"] == "CAR-T CD19 therapy"
        assert results[0]["index"] == 0
        assert "score" in results[0]


# ── 评测指标测试 ──────────────────────────────────────────────────────────


class TestEvaluation:
    """测试 MRR / Recall@10 计算。"""

    def test_mrr_perfect_ranking(self):
        """相关文档都排第一 → MRR = 1.0。"""
        eval_set = [("q1", {0}), ("q2", {1})]
        search_fn = {"q1": lambda q: [0, 1], "q2": lambda q: [1, 0]}
        mrr = mean_reciprocal_rank(eval_set, lambda q: search_fn[q](q))
        assert mrr == pytest.approx(1.0)

    def test_mrr_second_place(self):
        """相关文档排第二 → 单条 RR = 0.5。"""
        eval_set = [("q1", {1})]
        mrr = mean_reciprocal_rank(eval_set, lambda q: [0, 1, 2])
        assert mrr == pytest.approx(0.5)

    def test_recall_at_k(self):
        """前 k 条捞回的相关文档比例。"""
        eval_set = [("q1", {0, 1, 2, 3})]
        # 前 2 条命中 2/4
        recall = recall_at_k(eval_set, lambda q: [0, 1, 9, 9], k=2)
        assert recall == pytest.approx(0.5)

    def test_empty_eval_set(self):
        """空评测集返回 0。"""
        assert mean_reciprocal_rank([], lambda q: []) == 0.0
        assert recall_at_k([], lambda q: []) == 0.0
