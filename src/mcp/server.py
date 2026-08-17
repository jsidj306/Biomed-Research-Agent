"""
MCP Server — 把研究助手的能力封装为 MCP 工具。

用 FastMCP 把四个核心能力暴露给 MCP 客户端（Claude Desktop / Cursor 等）：
  - search_papers    多源并行文献检索（PubMed / arXiv / Semantic Scholar / bioRxiv / medRxiv）
  - summarize_paper  单篇论文结构化摘要
  - generate_report  完整 7-Agent 流水线，生成六段式报告 + 研究方向
  - query_kg         从文本抽取知识图谱三元组（药物-靶点-疾病），附 Cypher + Mermaid

工具内部采用惰性导入 + 依赖注入友好设计：重活（LLM / 网络 / Neo4j）都在调用时
才导入，测试可直接 patch 底层类而不触发真实调用。

运行（stdio 传输，供 Claude Desktop / Cursor 配置）::

    uv run python src/mcp/server.py
"""

from __future__ import annotations

import logging
from typing import Any

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

# 全局 MCP 服务实例（工具注册入口）
mcp = FastMCP("biomed-research-agent")


@mcp.tool()
def search_papers(query: str, max_results: int = 10) -> dict[str, Any]:
    """多源并行检索生物医学文献（PubMed / arXiv / Semantic Scholar / bioRxiv / medRxiv）。

    Args:
        query:       检索查询词，如 "CAR-T therapy for solid tumors"
        max_results: 每个数据源返回结果上限（默认 10）

    Returns:
        {"query", "count", "papers"} — 归一化去重后的论文列表
    """
    from src.agents.search_agent import SearchAgent

    papers = SearchAgent().search_all(query, max_results=max_results)
    return {"query": query, "count": len(papers), "papers": papers}


@mcp.tool()
def summarize_paper(
    title: str,
    abstract: str,
    language: str = "english",
) -> dict[str, Any]:
    """对单篇论文生成结构化深度摘要。

    Args:
        title:    论文标题
        abstract: 论文摘要文本
        language: 输出语言（english / chinese）

    Returns:
        结构化摘要字典（background / methods / results / conclusions 等字段）
    """
    from src.agents.summarizer import PaperSummarizer

    return PaperSummarizer().summarize(
        title=title,
        abstract=abstract,
        summary_type="detailed",
        language=language,
    )


@mcp.tool()
def generate_report(query: str) -> dict[str, Any]:
    """运行完整 7-Agent 流水线，生成带引用溯源的结构化文献综述报告。

    Args:
        query: 研究问题，如 "What is the latest evidence for CAR-T in solid tumors?"

    Returns:
        {"report", "directions", "critic_score", "critic_issues", "papers_count"}
    """
    from src.agents.graph import build_graph

    app = build_graph()
    result = app.invoke({"query": query})
    return {
        "report": result.get("report", ""),
        "directions": result.get("directions", []),
        "critic_score": result.get("critic_score"),
        "critic_issues": result.get("critic_issues", []),
        "papers_count": len(result.get("papers", [])),
    }


@mcp.tool()
def query_kg(text: str) -> dict[str, Any]:
    """从生物医学文本抽取知识图谱三元组（药物-靶点-疾病），附 Cypher 与 Mermaid。

    Args:
        text: 生物医学文本（论文摘要或正文），如 "CAR-T therapy treats B-cell leukemia"

    Returns:
        {"triples", "cypher", "mermaid"} — 三元组列表 + 可执行 Cypher + Mermaid 图
    """
    from src.knowledge_graph.extractor import KnowledgeGraphExtractor, to_mermaid
    from src.knowledge_graph.neo4j_writer import Neo4jWriter

    triples = KnowledgeGraphExtractor().extract_triples(text)
    return {
        "triples": triples,
        "cypher": Neo4jWriter.build_cypher(triples),
        "mermaid": to_mermaid(triples),
    }


if __name__ == "__main__":  # pragma: no cover
    mcp.run(transport="stdio")
