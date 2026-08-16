"""Biomedical Research Agent — Knowledge Graph package.

知识图谱模块：把论文里的实体关系（Drug / Target / Disease / Gene）抽成三元组，
支持 LLM 主抽取 + 词典兜底，写入 Neo4j 并提供 Cypher / Mermaid 导出。

组件:
- KnowledgeGraphExtractor: 实体关系抽取（LLM + 词典兜底）
- Neo4jWriter:           Neo4j 写入（MERGE 去重）+ Cypher 生成
- to_mermaid:            三元组 → Mermaid 图（无依赖，用于文档/Streamlit 展示）
"""

from src.knowledge_graph.extractor import (
    KnowledgeGraphExtractor,
    to_mermaid,
)
from src.knowledge_graph.neo4j_writer import Neo4jWriter

__all__ = ["KnowledgeGraphExtractor", "Neo4jWriter", "to_mermaid"]
