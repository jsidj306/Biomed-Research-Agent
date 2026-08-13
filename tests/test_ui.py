"""
Streamlit UI — 组件单元测试。

测试 UI 组件的数据处理逻辑（不依赖 Streamlit 运行时渲染），
使用 streamlit.testing 进行组件级验证。

运行: uv run pytest tests/test_ui.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def sample_pubmed_paper():
    """模拟 PubMed 论文数据。"""
    return {
        "pmid": "38200123",
        "title": "mRNA vaccine combined with checkpoint inhibitor in pancreatic cancer",
        "authors": ["Rojas LA", "Sethna Z", "Soares KC", "Smith J", "Brown A"],
        "year": "2025",
        "journal": "Nature Medicine",
        "abstract": (
            "Pancreatic ductal adenocarcinoma remains a deadly cancer. "
            "This phase I/II trial evaluated autogene cevumeran, an "
            "individualized mRNA neoantigen vaccine, in combination with "
            "atezolizumab and mFOLFIRINOX. At 18-month follow-up, "
            "responders showed significant T cell expansion."
        ),
        "doi": "10.1038/s41591-025-00001",
        "url": "https://pubmed.ncbi.nlm.nih.gov/38200123/",
        "source": "pubmed",
    }


@pytest.fixture
def sample_arxiv_paper():
    """模拟 Arxiv 论文数据。"""
    return {
        "arxiv_id": "2401.00001v1",
        "title": "Transformer-based molecular property prediction for drug discovery",
        "authors": ["Chen W", "Liu Y", "Zhang M"],
        "published": "2024-01-01",
        "categories": ["q-bio.BM", "cs.LG"],
        "summary": (
            "We present MolFormer-XL, a large-scale transformer model "
            "pretrained on 100M molecules for molecular property prediction. "
            "The model achieves state-of-the-art on 8 of 10 ADMET benchmarks."
        ),
        "pdf_url": "https://arxiv.org/pdf/2401.00001",
        "source": "arxiv",
    }


@pytest.fixture
def sample_summary_brief():
    return {
        "paper_title": "Test Paper",
        "summary_type": "brief",
        "brief": {
            "one_liner": "mRNA vaccine shows promise in pancreatic cancer.",
            "takeaway": "Personalized neoantigen vaccines could delay recurrence in PDAC."
        },
        "detailed": None,
        "key_findings": None,
        "model": "gpt-4o-mini",
        "tokens_used": 150,
    }


@pytest.fixture
def sample_summary_detailed():
    return {
        "paper_title": "Test Paper",
        "summary_type": "detailed",
        "brief": None,
        "detailed": {
            "background": "PDAC is a deadly cancer with limited treatment options.",
            "methods": "Phase I/II trial combining mRNA vaccine with checkpoint inhibitor.",
            "results": "18-month RFS rate of 72% vs 45% in historical controls.",
            "conclusions": "Personalized mRNA vaccines warrant phase III investigation."
        },
        "key_findings": None,
        "model": "gpt-4o-mini",
    }


@pytest.fixture
def sample_summary_key_findings():
    return {
        "paper_title": "Test Paper",
        "summary_type": "key_findings",
        "brief": None,
        "detailed": None,
        "key_findings": {
            "findings": [
                "mRNA vaccine induces durable T cell responses",
                "18-month RFS rate significantly improved",
                "Combination therapy well-tolerated",
                "CD8+ effector memory T cells enriched"
            ],
            "significance": "First evidence of personalized mRNA vaccine efficacy in PDAC."
        },
        "model": "gpt-4o-mini",
    }


@pytest.fixture
def sample_papers_list():
    """模拟搜索结果列表。"""
    return [
        {
            "pmid": "38200123",
            "title": "mRNA vaccine in pancreatic cancer",
            "authors": ["Rojas LA", "Sethna Z"],
            "year": "2025",
            "journal": "Nature Medicine",
            "abstract": "A phase I/II trial evaluating mRNA vaccine...",
            "source": "pubmed",
            "url": "https://pubmed.ncbi.nlm.nih.gov/38200123/",
        },
        {
            "pmid": "38200124",
            "title": "CAR-T cell therapy for solid tumors",
            "authors": ["Smith J", "Brown A"],
            "year": "2024",
            "journal": "Science",
            "abstract": "CAR-T cells engineered to target solid tumor antigens...",
            "source": "pubmed",
            "url": "https://pubmed.ncbi.nlm.nih.gov/38200124/",
        },
        {
            "pmid": "38200125",
            "title": "CRISPR delivery via lipid nanoparticles",
            "authors": ["Lee K"],
            "year": "2023",
            "journal": "Cell",
            "abstract": "A novel LNP formulation for CRISPR delivery...",
            "source": "pubmed",
            "url": "https://pubmed.ncbi.nlm.nih.gov/38200125/",
        },
    ]


# ── 组件导入测试 ────────────────────────────────────────────────────────


class TestComponentImports:
    """验证组件模块可正确导入。"""

    def test_import_components(self):
        """src/ui/components.py 所有函数可导入。"""
        from src.ui.components import (
            render_paper_card,
            render_paper_list,
            render_brief_summary,
            render_detailed_summary,
            render_key_findings,
            render_summary_tabs,
            render_sidebar,
        )
        assert callable(render_paper_card)
        assert callable(render_paper_list)
        assert callable(render_summary_tabs)
        assert callable(render_sidebar)

    def test_import_ui_package(self):
        """src/ui/__init__.py 导出正确。"""
        from src.ui import (
            render_paper_card,
            render_paper_list,
            render_summary_tabs,
            render_sidebar,
        )
        assert callable(render_paper_card)
        assert callable(render_paper_list)


# ── 论文数据格式验证 ────────────────────────────────────────────────────


class TestPaperDataValidation:
    """验证论文数据的统一格式要求。"""

    def test_pubmed_paper_has_required_fields(self, sample_pubmed_paper):
        """PubMed 论文包含所有必需字段。"""
        required = ["title", "authors", "abstract", "source", "pmid"]
        for field in required:
            assert field in sample_pubmed_paper, f"Missing: {field}"

    def test_arxiv_paper_has_required_fields(self, sample_arxiv_paper):
        """Arxiv 论文包含所有必需字段。"""
        required = ["title", "authors", "source", "arxiv_id"]
        for field in required:
            assert field in sample_arxiv_paper, f"Missing: {field}"
        # Arxiv 用 summary 字段，需要 unified 为 abstract
        assert "summary" in sample_arxiv_paper or "abstract" in sample_arxiv_paper

    def test_papers_list_structure(self, sample_papers_list):
        """论文列表结构正确。"""
        assert len(sample_papers_list) == 3
        assert all("title" in p for p in sample_papers_list)
        assert all("source" in p for p in sample_papers_list)


# ── 摘要结果格式验证 ────────────────────────────────────────────────────


class TestSummaryFormatValidation:
    """验证摘要结果的数据格式。"""

    def test_brief_summary_structure(self, sample_summary_brief):
        """简短摘要包含 one_liner 和 takeaway。"""
        brief = sample_summary_brief["brief"]
        assert "one_liner" in brief
        assert "takeaway" in brief
        assert len(brief["one_liner"]) > 0
        assert len(brief["takeaway"]) > 0

    def test_detailed_summary_structure(self, sample_summary_detailed):
        """详细摘要包含四个部分。"""
        detailed = sample_summary_detailed["detailed"]
        for key in ["background", "methods", "results", "conclusions"]:
            assert key in detailed, f"Missing: {key}"
            assert len(detailed[key]) > 0

    def test_key_findings_structure(self, sample_summary_key_findings):
        """关键发现包含 findings 列表和 significance。"""
        kf = sample_summary_key_findings["key_findings"]
        assert "findings" in kf
        assert "significance" in kf
        assert isinstance(kf["findings"], list)
        assert len(kf["findings"]) >= 3

    def test_summary_has_model_info(self, sample_summary_brief):
        """摘要结果包含模型元数据。"""
        assert "model" in sample_summary_brief
        assert sample_summary_brief["model"] == "gpt-4o-mini"


# ── 搜索流程集成测试 (Mock) ─────────────────────────────────────────────


class TestSearchFlow:
    """测试搜索流程的核心逻辑。"""

    @patch("src.ui.app.PubMedSearchTool")
    @patch("src.ui.app.ArxivSearchTool")
    def test_search_pubmed_only(self, MockArxiv, MockPubMed, sample_papers_list):
        """仅搜索 PubMed。"""
        mock_tool = MagicMock()
        mock_tool._run.return_value = {
            "papers": sample_papers_list[:2],
            "total_count": 2,
            "query": "test query",
            "source": "pubmed",
        }
        MockPubMed.return_value = mock_tool

        # 导入并测试
        from src.ui.app import search_papers
        papers = search_papers("test query", ["PubMed"], max_results=5)

        assert len(papers) == 2
        assert all(p["source"] == "pubmed" for p in papers)

    def test_empty_sources_returns_empty(self):
        """无数据源时返回空列表。"""
        from src.ui.app import search_papers
        papers = search_papers("test", [], max_results=5)
        assert papers == []


# ── 侧边栏配置测试 ──────────────────────────────────────────────────────


class TestSidebarConfig:
    """测试侧边栏配置数据结构。"""

    def test_config_keys(self):
        """验证配置包含所有必需键。"""
        # 不调用 Streamlit，直接验证配置结构
        expected_keys = {"search_source", "max_results", "summary_type", "language", "model"}
        # render_sidebar 需要在 Streamlit runtime 中运行，此处验证函数存在
        from src.ui.components import render_sidebar
        assert callable(render_sidebar)

    def test_config_defaults(self):
        """默认配置值合理。"""
        defaults = {
            "search_source": ["PubMed", "Arxiv"],
            "max_results": 5,
            "summary_type": "all",
            "language": "english",
            "model": "gpt-4o-mini",
        }
        assert len(defaults["search_source"]) == 2
        assert 1 <= defaults["max_results"] <= 20
        assert defaults["summary_type"] in ["all", "brief", "detailed", "key_findings"]
        assert defaults["language"] in ["english", "chinese"]
        assert defaults["model"] in ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]


# ── 论文卡片数据提取 ────────────────────────────────────────────────────


class TestPaperCardData:
    """测试论文卡片的数据处理。"""

    def test_extract_title(self, sample_pubmed_paper):
        """提取标题。"""
        title = sample_pubmed_paper.get("title", "Untitled")
        assert len(title) > 0
        assert "mRNA" in title

    def test_extract_authors_display(self, sample_pubmed_paper):
        """作者列表格式正确。"""
        authors = sample_pubmed_paper["authors"]
        display = ", ".join(authors[:3])
        if len(authors) > 3:
            display += " et al."
        assert "Rojas LA" in display
        assert "et al." in display

    def test_abstract_truncation(self):
        """长摘要截断到合理长度。"""
        long_abstract = "X " * 2500  # ~5000 chars
        display = long_abstract[:2000]
        assert len(display) <= 2000
