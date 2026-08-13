"""Tests for BiomedicalEmbeddings.

NOTE: These tests load the actual PubMedBERT model (large download on first run).
Mark with pytest -m "slow" to optionally skip in CI fast path.
"""

import numpy as np
import pytest

from src.rag.embeddings import BiomedicalEmbeddings, _MODEL_REGISTRY


class TestModelRegistry:
    """模型注册表测试。"""

    def test_registry_has_required_models(self):
        assert "pubmedbert" in _MODEL_REGISTRY
        assert "biolinkbert" in _MODEL_REGISTRY
        assert "biobert" in _MODEL_REGISTRY

    def test_pubmedbert_model_name(self):
        emb = BiomedicalEmbeddings("pubmedbert")
        assert "PubMedBert" in emb.model_name


@pytest.mark.slow
class TestBiomedicalEmbeddings:
    """嵌入模型测试（需要下载模型）。"""

    @pytest.fixture(scope="class")
    def embeddings(self):
        """共享嵌入模型实例，避免重复下载。"""
        return BiomedicalEmbeddings("pubmedbert")

    def test_embed_query_dimensions(self, embeddings):
        """查询嵌入应为 768 维。"""
        vec = embeddings.embed_query("CRISPR-Cas9 gene editing")
        assert len(vec) == 768
        assert all(isinstance(v, float) for v in vec)

    def test_embed_documents_dimensions(self, embeddings):
        """批量文档嵌入应返回正确数量的 768 维向量。"""
        texts = [
            "CRISPR is a gene editing tool.",
            "Cancer immunotherapy uses T cells.",
            "Single-cell sequencing reveals heterogeneity.",
        ]
        vectors = embeddings.embed_documents(texts)
        assert len(vectors) == 3
        for vec in vectors:
            assert len(vec) == 768

    def test_embed_query_normalized(self, embeddings):
        """归一化后 L2 范数应接近 1.0。"""
        vec = embeddings.embed_query("CRISPR cancer therapy")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 0.01, f"L2 norm={norm}, expected ~1.0"

    def test_semantic_similarity(self, embeddings):
        """语义相似的医学术语应有更高的余弦相似度。"""
        vec_cancer = embeddings.embed_query("cancer immunotherapy")
        vec_tumor = embeddings.embed_query("tumor treatment with T cells")
        vec_unrelated = embeddings.embed_query("computer science algorithm sorting")

        # 余弦相似度 = 点积（归一化后等价）
        sim_medical = float(np.dot(vec_cancer, vec_tumor))
        sim_unrelated = float(np.dot(vec_cancer, vec_unrelated))

        assert sim_medical > sim_unrelated, (
            f"Medical similarity ({sim_medical:.3f}) should > "
            f"unrelated ({sim_unrelated:.3f})"
        )
