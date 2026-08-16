"""
Writer Agent — 单元测试。

运行: uv run pytest tests/test_writer_agent.py -v

验证六段式报告结构、引用协议、BibTeX 导出和空输入降级，
通过 mock LLM 返回固定文本，不触发真实 API 调用。
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.agents.writer_agent import CITATION_FMT, REPORT_SECTIONS, WriterAgent

# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def papers():
    return [
        {
            "title": "CAR-T therapy for solid tumors",
            "authors": ["Jane Smith", "John Doe", "Alice Wu", "Bob Lee"],
            "journal": "Nature Medicine",
            "year": "2024",
            "pmid": "12345678",
            "doi": "10.1000/xyz",
        },
        {
            "title": "mRNA vaccine in pancreatic cancer",
            "authors": ["Wei Chen"],
            "journal": "Cancer Cell",
            "year": "2023",
            "pmid": "87654321",
        },
    ]


def _make_writer(mock_llm):
    with patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "https://test.api.com/v1"},
    ), patch("src.agents.writer_agent.create_chat_model", return_value=mock_llm):
        writer = WriterAgent()
    writer.llm = mock_llm
    return writer


def _mock_response(content):
    resp = MagicMock()
    resp.content = content
    return resp


# ── References 生成测试（代码拼接）────────────────────────────────────────


class TestBuildReferences:
    """测试 References 段落的代码模板拼接。"""

    def test_references_numbered_and_formatted(self, papers):
        refs = WriterAgent.build_references(papers)
        assert "[1]" in refs
        assert "[2]" in refs
        assert "Nature Medicine" in refs
        assert "PMID:12345678" in refs

    def test_references_author_et_al(self, papers):
        """超过 3 位作者应截断 + et al.。"""
        refs = WriterAgent.build_references(papers)
        assert "et al." in refs

    def test_empty_papers(self):
        assert WriterAgent.build_references([]) == ""


# ── BibTeX 导出测试 ──────────────────────────────────────────────────────


class TestBuildBibtex:
    """测试 BibTeX 条目导出。"""

    def test_bibtex_structure(self, papers):
        bibtex = WriterAgent.build_bibtex(papers)
        assert bibtex.count("@article{") == 2
        assert "title = {CAR-T therapy for solid tumors}" in bibtex
        assert "pmid = {12345678}" in bibtex

    def test_bibtex_key_first_author_surname_year(self, papers):
        bibtex = WriterAgent.build_bibtex(papers)
        assert "@article{smith2024," in bibtex  # 第一作者 Smith + 2024

    def test_empty_papers(self):
        assert WriterAgent.build_bibtex([]) == ""


# ── 引用提取测试 ─────────────────────────────────────────────────────────


class TestExtractCitations:
    """测试正文引用编号提取。"""

    def test_extract_sorted_unique(self):
        citations = WriterAgent.extract_citations("结论 [3] 与 [1] 一致，另见 [3]")
        assert citations == [1, 3]


# ── 报告生成测试 ─────────────────────────────────────────────────────────


class TestWriteReport:
    """测试六段式报告生成。"""

    def test_six_sections_present(self, papers):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_response("这是某章节的正文内容 [1]。")
        writer = _make_writer(mock_llm)

        report = writer.write_report({"papers": papers, "summaries": []})

        for section in REPORT_SECTIONS:
            assert f"## {section}" in report
        # References 由代码拼接，应含真实 PMID
        assert "PMID:12345678" in report

    def test_reference_count_matches_papers(self, papers):
        """References 编号应与 papers 数量一致。"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_response("正文 [1]。")
        writer = _make_writer(mock_llm)

        report = writer.write_report({"papers": papers, "summaries": []})

        assert "[1]" in report
        assert "[2]" in report
        assert "[3]" not in report  # 只有 2 篇

    def test_empty_papers_placeholder(self):
        mock_llm = MagicMock()
        writer = _make_writer(mock_llm)

        report = writer.write_report({"papers": [], "summaries": []})

        assert "未检索到论文" in report


# ── 常量测试 ─────────────────────────────────────────────────────────────


class TestConstants:
    def test_report_sections_order(self):
        assert REPORT_SECTIONS[-1] == "References"
        assert REPORT_SECTIONS[0] == "Abstract"

    def test_citation_fmt_placeholders(self):
        assert "{i}" in CITATION_FMT
        assert "{pmid}" in CITATION_FMT
