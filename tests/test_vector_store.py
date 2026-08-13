"""Tests for BiomedicalVectorStore."""

import shutil
import tempfile

import pytest
from langchain_core.documents import Document

from src.rag.vector_store import BiomedicalVectorStore


class FakeEmbeddings:
    """Fake embedding function for testing — returns deterministic vectors."""

    def embed_documents(self, texts):
        return [[float(hash(t) % 100) / 100.0] * 384 for t in texts]

    def embed_query(self, text):
        return [float(hash(text) % 100) / 100.0] * 384


class TestBiomedicalVectorStore:
    """向量存储测试（使用 FakeEmbeddings 避免下载模型）。"""

    @pytest.fixture
    def store(self):
        """创建临时目录的向量存储。"""
        tmpdir = tempfile.mkdtemp()
        store = BiomedicalVectorStore(
            persist_directory=tmpdir,
            collection_name="test_collection",
        )
        yield store
        # Cleanup
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def fake_emb(self):
        return FakeEmbeddings()

    @pytest.fixture
    def sample_docs(self):
        return [
            Document(
                page_content="CRISPR-Cas9 is a revolutionary gene editing technology.",
                metadata={"pmid": "1", "title": "CRISPR Review"},
            ),
            Document(
                page_content="CAR-T cell therapy uses engineered T cells to fight cancer.",
                metadata={"pmid": "2", "title": "CAR-T Study"},
            ),
            Document(
                page_content="Single-cell RNA sequencing reveals cellular heterogeneity.",
                metadata={"pmid": "3", "title": "scRNA-seq Paper"},
            ),
        ]

    def test_index_and_search(self, store, fake_emb, sample_docs):
        """索引后搜索应返回正确文档。"""
        count = store.index_documents(sample_docs, fake_emb)
        assert count == 3

        results = store.search("gene editing", fake_emb, k=2)
        assert len(results) == 2
        assert "CRISPR" in results[0]["content"]
        assert "metadata" in results[0]
        assert "score" in results[0]

    def test_empty_search(self, store, fake_emb, sample_docs):
        """索引后查询无关内容 — 仍应返回结果（向量搜索总会返回最近邻）。"""
        store.index_documents(sample_docs, fake_emb)
        results = store.search("zzz_nonexistent_topic_zzz", fake_emb, k=1)
        assert len(results) == 1  # 即使不相关也会返回

    def test_empty_index(self, store, fake_emb):
        """空索引不应崩溃。"""
        count = store.index_documents([], fake_emb)
        assert count == 0

    def test_get_stats_before_index(self, store):
        """索引前 stats.count 应为 0。"""
        stats = store.get_stats()
        assert stats["count"] == 0

    def test_get_stats_after_index(self, store, fake_emb, sample_docs):
        """索引后 stats.count 应匹配。"""
        store.index_documents(sample_docs, fake_emb)
        stats = store.get_stats()
        assert stats["count"] == 3
        assert stats["name"] == "test_collection"
