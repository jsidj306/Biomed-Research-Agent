"""
检索评测集 — 单元测试。

运行: uv run pytest tests/test_eval_set.py -v

校验评测集的规模、标注完整性与可测量性（用 TF-IDF 语义路 + BM25 混合，
纯 CPU、确定性，不下载模型）。
"""

from src.rag.eval_set import CORPUS, QUERIES, TfidfSemanticRanker, build_eval_set
from src.rag.hybrid_search import HybridRetriever, mean_reciprocal_rank, recall_at_k


# ── 评测集结构测试 ─────────────────────────────────────────────────────────


class TestEvalSetStructure:
    """测试评测集的规模与标注完整性。"""

    def test_corpus_size(self):
        """语料应至少 50 篇，覆盖足够多主题。"""
        assert len(CORPUS) >= 50

    def test_query_count(self):
        """查询应至少 30 条，满足第3-4周验收要求。"""
        assert len(QUERIES) >= 30

    def test_every_query_has_labels(self):
        """每条查询都应有非空的相关文档标注。"""
        for query, relevant in build_eval_set():
            assert relevant, f"查询无标注: {query}"

    def test_label_ids_in_range(self):
        """相关文档 ID 必须落在语料下标范围内。"""
        for _, relevant in build_eval_set():
            assert all(0 <= i < len(CORPUS) for i in relevant)

    def test_corpus_fields_complete(self):
        """每篇语料应有 id / title / abstract。"""
        for doc in CORPUS:
            assert doc["title"].strip()
            assert doc["abstract"].strip()


# ── 可测量性测试 ───────────────────────────────────────────────────────────


class TestMeasurability:
    """测试评测集能被混合检索器真实测量（确定性、无模型下载）。"""

    def test_mrr_recall_computable(self):
        """混合检索应能算出 [0,1] 内的 MRR / Recall@10，且非全零。"""
        corpus = [f"{d['title']}. {d['abstract']}" for d in CORPUS]
        ranker = TfidfSemanticRanker(corpus)
        retriever = HybridRetriever(corpus)
        eval_set = build_eval_set()

        search_fn = lambda q: [r["index"] for r in retriever.search(q, ranker.rank, k=10)]  # noqa: E731

        mrr = mean_reciprocal_rank(eval_set, search_fn)
        recall = recall_at_k(eval_set, search_fn)

        assert 0.0 <= mrr <= 1.0
        assert 0.0 <= recall <= 1.0
        assert mrr > 0.0
        assert recall > 0.0
