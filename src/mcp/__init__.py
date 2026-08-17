"""
MCP（Model Context Protocol）服务封装。

把 7-Agent 研究助手的核心能力暴露为 MCP 工具，供 Claude Desktop / Cursor 等
MCP 客户端直接调用，无需打开 Streamlit UI。

四个工具:
  - search_papers    多源并行文献检索
  - summarize_paper  单篇论文结构化摘要
  - generate_report  完整 7-Agent 流水线（六段式报告 + 研究方向）
  - query_kg         从文本抽取知识图谱三元组
"""

from src.mcp.server import (
    generate_report,
    mcp,
    query_kg,
    search_papers,
    summarize_paper,
)

__all__ = [
    "generate_report",
    "mcp",
    "query_kg",
    "search_papers",
    "summarize_paper",
]
