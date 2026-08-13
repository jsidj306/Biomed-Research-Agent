"""Tests for RAGPipeline."""

import pytest

from src.rag.rag_pipeline import RAGPipeline


@pytest.fixture
def pipeline(monkeypatch, tmp_path, fake_emb):
    """用 FakeEmbeddings 替换真实嵌入模型，避免下载 PubMedBERT。"""
    import src.rag.rag_pipeline as rp

    class _FakeEmbeddingsWrapper:
        def __init__(self, model_name="pubmedbert"):
            self.model_name = model_name
            self.embedding_model = fake_emb

    monkeypatch.setattr(rp, "BiomedicalEmbeddings", _FakeEmbeddingsWrapper)
    return RAGPipeline(
        persist_directory=str(tmp_path),
        collection_name="test_rag",
    )


class TestRAGPipeline:
    """RAG Pipeline 端到端测试（FakeEmbeddings + 临时 ChromaDB）。"""

    def test_ingest_texts_and_search(self, pipeline):
        """摄入文本后语义搜索能返回相关文档。"""
        count = pipeline.ingest_texts(
            [
                "CRISPR-Cas9 is a revolutionary gene editing technology.",
                "CAR-T cell therapy uses engineered T cells to fight cancer.",
            ],
            [{"pmid": "1"}, {"pmid": "2"}],
        )
        assert count > 0

        results = pipeline.search("gene editing", k=2)
        assert len(results) == 2
        assert any("CRISPR" in r["content"] for r in results)

    def test_ingest_texts_without_metadata(self, pipeline):
        """无元数据时摄入仍应成功。"""
        count = pipeline.ingest_texts(["A single test document about genomics."])
        assert count > 0

    def test_get_stats_empty(self, pipeline):
        """索引前 stats 应反映空集合与配置。"""
        stats = pipeline.get_stats()
        assert stats["count"] == 0
        assert stats["model"] == "pubmedbert"
        assert stats["chunk_size"] == 1000
        assert stats["chunk_overlap"] == 200

    def test_get_stats_after_ingest(self, pipeline):
        """索引后 stats.count 应大于 0。"""
        pipeline.ingest_texts(["CRISPR gene editing test document."])
        stats = pipeline.get_stats()
        assert stats["count"] > 0
