"""
RAG Pipeline — 一键从 PDF 到可检索知识库。

串联所有 RAG 模块:
  BiomedicalDocumentLoader → BiomedicalTextSplitter → BiomedicalEmbeddings → BiomedicalVectorStore

使用示例:
    pipeline = RAGPipeline(model_name="pubmedbert")
    n = pipeline.ingest_pdf("data/papers/test.pdf")      # PDF → ChromaDB
    results = pipeline.search("CRISPR delivery methods")  # 语义检索
    for r in results:
        print(r["content"][:100], r["score"])
"""

import logging
from typing import Any

from src.rag.document_loader import BiomedicalDocumentLoader
from src.rag.embeddings import BiomedicalEmbeddings
from src.rag.text_splitter import BiomedicalTextSplitter
from src.rag.vector_store import BiomedicalVectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """生物医学 RAG 流水线。

    封装完整的文档摄入 + 语义搜索流程。

    Attributes:
        loader:     文档加载器
        splitter:   文本分块器
        embeddings: 嵌入模型
        store:      向量存储
    """

    def __init__(
        self,
        model_name: str = "pubmedbert",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "biomedical_papers",
    ):
        """初始化 RAG Pipeline。

        Args:
            model_name:       嵌入模型简称
            chunk_size:       分块大小（字符数）
            chunk_overlap:    分块重叠（字符数）
            persist_directory: ChromaDB 数据目录
            collection_name:  向量集合名称
        """
        self.loader = BiomedicalDocumentLoader()
        self.splitter = BiomedicalTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embeddings = BiomedicalEmbeddings(model_name=model_name)
        self.store = BiomedicalVectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        logger.info(
            "RAG Pipeline ready: model=%s, chunk=%d/%d",
            model_name, chunk_size, chunk_overlap,
        )

    # ── 摄入 ──────────────────────────────────────────────────────────

    def ingest_pdf(self, pdf_path: str) -> int:
        """摄入一篇 PDF：加载 → 分块 → 嵌入 → 存入 ChromaDB。

        Args:
            pdf_path: PDF 文件路径

        Returns:
            int — 存入 ChromaDB 的文档块数量
        """
        logger.info("=== Ingesting: %s ===", pdf_path)

        # Step 1: Load
        docs = self.loader.load(pdf_path)
        logger.info("Step 1/3: Loaded %d pages/documents", len(docs))

        # Step 2: Split
        chunks = self.splitter.split(docs)
        logger.info("Step 2/3: Split into %d chunks", len(chunks))

        # Step 3: Embed + Store
        count = self.store.index_documents(
            chunks,
            self.embeddings.embedding_model,
        )
        logger.info("Step 3/3: Indexed %d chunks → ChromaDB", count)

        return count

    def ingest_texts(self, texts: list[str], metadata_list: list[dict] | None = None) -> int:
        """摄入纯文本列表（如 PubMed 摘要）。

        Args:
            texts:         文本列表
            metadata_list: 每条文本的元数据（可选）

        Returns:
            int — 存入的文档块数量
        """
        from langchain_core.documents import Document

        if metadata_list is None:
            metadata_list = [{}] * len(texts)

        docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadata_list)
        ]

        chunks = self.splitter.split(docs)
        count = self.store.index_documents(chunks, self.embeddings.embedding_model)
        logger.info("Ingested %d texts → %d chunks", len(texts), count)
        return count

    # ── 搜索 ──────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """语义搜索知识库。

        Args:
            query: 搜索查询
            k:     返回结果数

        Returns:
            [
                {
                    "content": "...",
                    "metadata": {...},
                    "score": 0.85,    # L2距离，越小越相关
                },
                ...
            ]
        """
        return self.store.search(
            query,
            self.embeddings.embedding_model,
            k=k,
        )

    # ── 统计 ──────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """获取 Pipeline 整体统计。"""
        return {
            **self.store.get_stats(),
            "model": self.embeddings.model_name,
            "chunk_size": self.splitter.chunk_size,
            "chunk_overlap": self.splitter.chunk_overlap,
        }
