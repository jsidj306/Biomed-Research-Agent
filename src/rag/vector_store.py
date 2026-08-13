"""
Biomedical Vector Store — ChromaDB 向量存储封装。

特性:
- 持久化存储到本地磁盘 (data/chroma_db/)
- 批量索引（batch_size=100，显示进度）
- 语义搜索 + 相关性分数
- 开发友好：同名 collection 自动重建
- 与 LangChain Chroma wrapper 深度集成
"""

import logging
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)


class BiomedicalVectorStore:
    """生物医学向量存储。

    底层 ChromaDB PersistentClient，持久化到本地磁盘。

    Attributes:
        persist_directory: ChromaDB 数据文件目录
        collection_name:   向量集合名称
        client:            ChromaDB 客户端实例
    """

    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "biomedical_papers",
    ):
        """初始化向量存储。

        Args:
            persist_directory: 持久化目录路径
            collection_name:   集合名称
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        logger.info(
            "ChromaDB ready: %s (collection='%s')",
            persist_directory, collection_name,
        )

    # ── 索引 ──────────────────────────────────────────────────────────

    def index_documents(
        self,
        documents: List[Document],
        embedding_function,
        batch_size: int = 100,
    ) -> int:
        """将文档块嵌入并存入 ChromaDB。

        自动重建同名 collection（开发阶段方便迭代）。

        Args:
            documents:         文档块列表（来自 TextSplitter）
            embedding_function: 嵌入模型实例（HuggingFaceEmbeddings 兼容）
            batch_size:        批量添加的大小

        Returns:
            int — 已索引的文档块数量
        """
        total = len(documents)
        if total == 0:
            logger.warning("No documents to index")
            return 0

        logger.info("Indexing %d chunks → ChromaDB...", total)

        # 开发阶段：重建 collection 避免旧数据干扰
        self._recreate_collection(embedding_function)

        vectordb = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=embedding_function,
        )

        # 批量添加
        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            vectordb.add_documents(batch)
            logger.info("  Indexed %d/%d chunks", min(i + batch_size, total), total)

        logger.info("Indexing complete: %d chunks in ChromaDB", total)
        return total

    def _recreate_collection(self, embedding_function):
        """删除旧 collection 并重建。"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info("Deleted existing collection: %s", self.collection_name)
        except Exception:
            pass  # Collection 不存在 = 无需删除

        # 通过创建 Chroma wrapper 触发 collection 创建
        Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=embedding_function,
        )

    # ── 搜索 ──────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        embedding_function,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """语义搜索 — 返回最相关的 k 个文档块。

        Args:
            query:             搜索查询文本
            embedding_function: 嵌入模型实例
            k:                 返回结果数

        Returns:
            [
                {
                    "content": "...",
                    "metadata": {...},  # 原始元数据 + chunk_id + chunk_size
                    "score": 0.85,       # 相关性分数（距离，越小越相关）
                },
                ...
            ]
        """
        vectordb = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=embedding_function,
        )

        results = vectordb.similarity_search_with_score(query, k=k)

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
            }
            for doc, score in results
        ]

    # ── 统计 ──────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """获取集合统计信息。

        Returns:
            {"name": str, "count": int}
        """
        try:
            collection = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "count": collection.count(),
                "persist_directory": self.persist_directory,
            }
        except Exception:
            return {
                "name": self.collection_name,
                "count": 0,
                "persist_directory": self.persist_directory,
            }
