"""Biomedical Research Agent — RAG Pipeline.

PDF加载 → 文本分块 → 向量嵌入 → ChromaDB存储 → 语义检索
"""

from src.rag.rag_pipeline import RAGPipeline

__all__ = ["RAGPipeline"]
