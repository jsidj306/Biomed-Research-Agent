"""
Biomedical Embeddings — 生物医学领域嵌入模型封装。

使用 PubMedBERT (pritamdeka/S-PubMedBert-MS-MARCO) 作为默认嵌入模型:
- 在 1400 万篇 PubMed 论文上预训练
- 对基因/蛋白/药物/疾病术语理解远超通用模型（如 text-embedding-3）
- 768 维向量，归一化输出
- HuggingFace 本地运行，零 API 费用，数据不出本地

首次加载自动从 HuggingFace 下载模型（~440MB），约需 1-3 分钟。

备选模型:
- BioLinkBERT: michiyasunaga/BioLinkBERT-base（知识图谱增强）
- BioBERT: dmis-lab/biobert-v1.1
"""

import logging
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# 推荐的生物医学嵌入模型（按推荐度排序）
_MODEL_REGISTRY = {
    "pubmedbert": "pritamdeka/S-PubMedBert-MS-MARCO",
    "biolinkbert": "michiyasunaga/BioLinkBERT-base",
    "biobert": "dmis-lab/biobert-v1.1",
    "pubmedbert_scifact": "pritamdeka/S-PubMedBert-MS-MARCO-SCIFACT",
}


class BiomedicalEmbeddings:
    """生物医学领域嵌入模型。

    封装 HuggingFaceEmbeddings，提供统一的 embed_documents / embed_query 接口。

    Attributes:
        model_name:  使用的 HuggingFace 模型名
        embedding_model: LangChain HuggingFaceEmbeddings 实例
    """

    def __init__(
        self,
        model_name: str = "pubmedbert",
        device: str = "cpu",
    ):
        """初始化嵌入模型。

        Args:
            model_name: 模型简称 ("pubmedbert" / "biolinkbert" / "biobert")
                        或完整的 HuggingFace 模型 ID
            device:     运行设备 ("cpu" / "cuda")
        """
        # 解析简称 → 完整模型 ID
        resolved = _MODEL_REGISTRY.get(model_name, model_name)
        self.model_name = resolved

        logger.info("Loading embedding model: %s (device=%s)", resolved, device)

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=resolved,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        # 标记实际加载的模型（防止 resolved 和 model_name 不一致）
        self.model_name = resolved
        logger.info("Embedding model ready: %s", resolved)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档块。

        Args:
            texts: 文本列表

        Returns:
            List[List[float]] — 每个文本对应一个 768 维向量
        """
        return self.embedding_model.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        """嵌入搜索查询。

        Args:
            query: 用户查询文本

        Returns:
            List[float] — 768 维查询向量
        """
        return self.embedding_model.embed_query(query)
