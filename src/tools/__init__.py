"""Biomedical Research Agent - Tools package.

LangChain BaseTool wrappers for biomedical literature search APIs.
统一返回 {papers, total_count, query, source} 格式，便于多源合并。
"""

from src.tools.arxiv_tool import ArxivSearchTool
from src.tools.biorxiv_tool import BioRxivSearchTool, MedRxivSearchTool
from src.tools.cnki_tool import CNKISearchTool
from src.tools.pubmed_tool import PubMedSearchTool
from src.tools.semantic_scholar_tool import SemanticScholarSearchTool

__all__ = [
    "ArxivSearchTool",
    "BioRxivSearchTool",
    "CNKISearchTool",
    "MedRxivSearchTool",
    "PubMedSearchTool",
    "SemanticScholarSearchTool",
]
