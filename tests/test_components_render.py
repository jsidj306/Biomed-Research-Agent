"""Tests for UI components — mock Streamlit rendering.

覆盖 src/ui/components.py 的渲染函数（不依赖真实 Streamlit 运行时）。
"""

from unittest.mock import MagicMock, patch

import pytest

from src.ui.components import (
    _linkify_citations,
    build_pmid_link,
    render_brief_summary,
    render_detailed_summary,
    render_key_findings,
    render_paper_card,
    render_paper_list,
    render_report,
    render_sidebar,
    render_summary_tabs,
)


@pytest.fixture
def mock_st():
    """Mock streamlit，返回支持 with 语句的对象。"""
    with patch("src.ui.components.st") as mock:
        mock.columns.return_value = [MagicMock() for _ in range(4)]
        mock.tabs.return_value = [MagicMock() for _ in range(3)]
        yield mock


@pytest.fixture
def paper():
    return {
        "pmid": "38200123",
        "title": "mRNA vaccine in pancreatic cancer",
        "authors": ["Rojas LA", "Sethna Z", "Soares KC", "Smith J"],
        "year": "2025",
        "journal": "Nature Medicine",
        "abstract": "A phase I/II trial evaluating mRNA vaccine...",
        "source": "pubmed",
        "url": "https://pubmed.ncbi.nlm.nih.gov/38200123/",
        "doi": "10.1038/s41591-025-00001",
    }


# ── 摘要渲染 ──────────────────────────────────────────────────────────────


class TestRenderSummaries:
    def test_render_brief_summary(self, mock_st):
        render_brief_summary({"one_liner": "A finding.", "takeaway": "Implication."})
        assert mock_st.markdown.call_count >= 2

    def test_render_brief_summary_empty(self, mock_st):
        render_brief_summary(None)
        assert not mock_st.markdown.called

    def test_render_detailed_summary(self, mock_st):
        render_detailed_summary({
            "background": "b", "methods": "m",
            "results": "r", "conclusions": "c",
        })
        assert mock_st.markdown.call_count >= 4

    def test_render_key_findings(self, mock_st):
        render_key_findings({"findings": ["f1", "f2"], "significance": "s"})
        assert mock_st.markdown.called

    def test_render_summary_tabs_all(self, mock_st):
        result = {
            "brief": {"one_liner": "a", "takeaway": "b"},
            "detailed": {"background": "b"},
            "key_findings": {"findings": ["f"]},
            "summary_type": "all",
        }
        render_summary_tabs(result)
        assert mock_st.tabs.called

    def test_render_summary_tabs_no_summary(self, mock_st):
        result = {
            "brief": None, "detailed": None,
            "key_findings": None, "summary_type": "all",
        }
        render_summary_tabs(result)
        assert mock_st.info.called


# ── 论文卡片渲染 ──────────────────────────────────────────────────────────


class TestRenderPaperCard:
    def test_render_paper_card(self, mock_st, paper):
        render_paper_card(paper, index=1)
        assert mock_st.markdown.called
        assert mock_st.container.called
        assert mock_st.columns.called

    def test_render_paper_list_empty(self, mock_st):
        render_paper_list([])
        assert mock_st.info.called

    def test_render_paper_list(self, mock_st, paper):
        render_paper_list([paper, paper], max_display=1)
        assert mock_st.markdown.called


# ── 引用溯源 ──────────────────────────────────────────────────────────────


class TestCitationTraceability:
    """测试 PMID 链接与 [n] 引用溯源。"""

    def test_build_pmid_link(self):
        link = build_pmid_link("38200123")
        assert "pubmed.ncbi.nlm.nih.gov/38200123" in link

    def test_build_pmid_link_empty(self):
        assert build_pmid_link("") == ""

    def test_linkify_replaces_valid_citation(self):
        report = "结论 [1] 表明……"
        papers = [{"pmid": "38200123"}]
        rendered = _linkify_citations(report, papers)
        assert "[1](https://pubmed.ncbi.nlm.nih.gov/38200123/)" in rendered

    def test_linkify_keeps_fabricated_citation(self):
        """超出 papers 范围的引用应保留原样（杜绝编造引用）。"""
        report = "结论 [5] 表明……"
        papers = [{"pmid": "38200123"}]
        rendered = _linkify_citations(report, papers)
        assert "[5]" in rendered
        assert "pubmed" not in rendered

    def test_linkify_no_papers_unchanged(self):
        report = "结论 [1] 表明……"
        assert _linkify_citations(report, []) == report

    def test_render_report(self, mock_st):
        render_report("## Abstract\n\n内容 [1]。", [{"pmid": "38200123"}])
        assert mock_st.markdown.called

    def test_render_report_empty(self, mock_st):
        render_report("")
        assert mock_st.info.called


# ── 侧边栏渲染 ────────────────────────────────────────────────────────────


class TestRenderSidebar:
    def test_render_sidebar(self, mock_st):
        mock_st.multiselect.return_value = ["PubMed", "Arxiv"]
        mock_st.slider.return_value = 5
        mock_st.selectbox.side_effect = ["all", "english", "glm-5.2"]

        config = render_sidebar()

        assert config["search_source"] == ["PubMed", "Arxiv"]
        assert config["max_results"] == 5
        assert config["summary_type"] == "all"
        assert config["language"] == "english"
        assert config["model"] == "glm-5.2"
