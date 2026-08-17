"""
MCP Server — 单元测试。

运行: uv run pytest tests/test_mcp_server.py -v

验证 4 个 MCP 工具（search_papers / summarize_paper / generate_report / query_kg）
的注册与委托逻辑。patch 掉重依赖（SearchAgent / PaperSummarizer / build_graph /
KnowledgeGraphExtractor），不触发真实 LLM / 网络 / 数据库调用。
"""

import asyncio
from unittest.mock import MagicMock, patch

from src.mcp import server
from src.mcp.server import mcp

# ── 工具注册 ───────────────────────────────────────────────────────────────


def test_mcp_server_registers_four_tools():
    """4 个工具均已注册到 MCP 服务实例。"""
    names = {tool.name for tool in asyncio.run(mcp.list_tools())}
    assert {"search_papers", "summarize_paper", "generate_report", "query_kg"} <= names


def test_mcp_server_name():
    """MCP 服务名称正确。"""
    assert mcp.name == "biomed-research-agent"


# ── search_papers ──────────────────────────────────────────────────────────


def test_search_papers_delegates_to_search_agent():
    """search_papers 调用 SearchAgent.search_all 并返回 {query, count, papers}。"""
    fake_paper = {
        "title": "CAR-T for solid tumors",
        "abstract": "A phase II trial...",
        "source": "pubmed",
    }
    fake_agent = MagicMock()
    fake_agent.return_value.search_all.return_value = [fake_paper]

    with patch("src.agents.search_agent.SearchAgent", fake_agent):
        result = server.search_papers("CAR-T", max_results=5)

    fake_agent.return_value.search_all.assert_called_once_with(
        "CAR-T", max_results=5
    )
    assert result["query"] == "CAR-T"
    assert result["count"] == 1
    assert result["papers"] == [fake_paper]


# ── summarize_paper ────────────────────────────────────────────────────────


def test_summarize_paper_delegates_to_summarizer():
    """summarize_paper 用 detailed 模式委托 PaperSummarizer.summarize。"""
    fake_summary = {"paper_title": "X", "detailed": {"methods": "..."}}
    fake_cls = MagicMock()
    fake_cls.return_value.summarize.return_value = fake_summary

    with patch("src.agents.summarizer.PaperSummarizer", fake_cls):
        result = server.summarize_paper("Title", "Abstract text...", "chinese")

    fake_cls.return_value.summarize.assert_called_once_with(
        title="Title", abstract="Abstract text...", summary_type="detailed", language="chinese"
    )
    assert result == fake_summary


# ── generate_report ────────────────────────────────────────────────────────


def test_generate_report_runs_pipeline():
    """generate_report 调用 build_graph().invoke() 并抽取报告字段。"""
    fake_app = MagicMock()
    fake_app.invoke.return_value = {
        "report": "# Report\n...",
        "directions": [{"hypothesis": "h1"}],
        "critic_score": 7.5,
        "critic_issues": [],
        "papers": [{}, {}, {}],
    }
    fake_build = MagicMock(return_value=fake_app)

    with patch("src.agents.graph.build_graph", fake_build):
        result = server.generate_report("CAR-T solid tumors?")

    fake_build.assert_called_once()
    fake_app.invoke.assert_called_once_with({"query": "CAR-T solid tumors?"})
    assert result["report"].startswith("# Report")
    assert result["critic_score"] == 7.5
    assert result["papers_count"] == 3
    assert result["directions"] == [{"hypothesis": "h1"}]


# ── query_kg ───────────────────────────────────────────────────────────────


def test_query_kg_extracts_triples_and_cypher():
    """query_kg 抽取三元组并生成对应 Cypher 与 Mermaid。"""
    fixed_triples = [
        {
            "head_type": "drug",
            "head_name": "CAR-T",
            "relation": "TREATS",
            "tail_type": "disease",
            "tail_name": "leukemia",
        }
    ]
    fake_cls = MagicMock()
    fake_cls.return_value.extract_triples.return_value = fixed_triples

    with patch("src.knowledge_graph.extractor.KnowledgeGraphExtractor", fake_cls):
        result = server.query_kg("CAR-T therapy treats leukemia")

    assert result["triples"] == fixed_triples
    assert len(result["cypher"]) == 1
    assert "TREATS" in result["cypher"][0]
    assert "CAR-T" in result["cypher"][0]
    assert "TREATS" in result["mermaid"]
    assert "CAR-T" in result["mermaid"]
