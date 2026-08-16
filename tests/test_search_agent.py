"""
Search Agent — 单元测试。

运行: uv run pytest tests/test_search_agent.py -v

通过 mock 5 个数据源 Tool，验证归一化、去重、并发调用和降级策略，
不触发真实 API 调用。
"""

import pytest

from src.agents.search_agent import (
    SearchAgent,
    deduplicate_papers,
    normalize_paper,
)

# ── Fixtures：mock 数据源 Tool ───────────────────────────────────────────


class _MockTool:
    """模拟一个数据源 Tool。"""

    def __init__(self, name, papers, source=None, exc=None):
        self.name = name
        self._papers = papers
        self._source = source or name
        self._exc = exc

    def _run(self, query, max_results=10):
        if self._exc is not None:
            raise self._exc
        return {
            "papers": self._papers,
            "total_count": len(self._papers),
            "query": query,
            "source": self._source,
        }


@pytest.fixture
def pubmed_paper():
    return {
        "pmid": "12345",
        "title": "CAR-T therapy for solid tumors",
        "authors": ["Smith J", "Lee K"],
        "year": "2024",
        "journal": "Nature Medicine",
        "abstract": "CAR-T cells show efficacy in solid tumors.",
        "doi": "10.1038/s41591-024-0001",
        "url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
    }


@pytest.fixture
def arxiv_paper():
    return {
        "arxiv_id": "2401.00001",
        "title": "Deep learning for drug discovery",
        "authors": ["Chen W", "Wang Y"],
        "published": "2024-01-15",
        "summary": "A transformer model for molecular property prediction.",
        "pdf_url": "https://arxiv.org/pdf/2401.00001",
    }


# ── 归一化测试 ───────────────────────────────────────────────────────────


class TestNormalizePaper:
    """测试不同数据源字段到统一 schema 的归一化。"""

    def test_normalize_pubmed(self, pubmed_paper):
        """PubMed 论文应保留全部核心字段。"""
        norm = normalize_paper(pubmed_paper, "pubmed")
        assert norm["title"] == "CAR-T therapy for solid tumors"
        assert norm["source"] == "pubmed"
        assert norm["doi"] == "10.1038/s41591-024-0001"
        assert norm["pmid"] == "12345"
        assert norm["authors"] == ["Smith J", "Lee K"]

    def test_normalize_arxiv_maps_fields(self, arxiv_paper):
        """arxiv 的 summary/pdf_url/published 应映射到 abstract/url/year。"""
        norm = normalize_paper(arxiv_paper, "arxiv")
        assert norm["abstract"] == "A transformer model for molecular property prediction."
        assert norm["url"] == "https://arxiv.org/pdf/2401.00001"
        assert norm["year"] == "2024"
        assert norm["journal"] == "arXiv preprint"

    def test_normalize_authors_string(self):
        """authors 为字符串时应拆分成列表。"""
        raw = {"title": "T", "authors": "Smith J, Lee K, Chen W"}
        norm = normalize_paper(raw, "pubmed")
        assert norm["authors"] == ["Smith J", "Lee K", "Chen W"]

    def test_normalize_missing_fields(self):
        """缺失字段应返回空值而非抛异常。"""
        norm = normalize_paper({}, "pubmed")
        assert norm["title"] == ""
        assert norm["authors"] == []
        assert norm["doi"] == ""


# ── 去重测试 ─────────────────────────────────────────────────────────────


class TestDeduplicate:
    """测试三级去重逻辑。"""

    def test_dedup_by_doi(self):
        """同 DOI 论文应被去重。"""
        papers = [
            {"title": "A", "doi": "10.1/x", "pmid": ""},
            {"title": "A (different title)", "doi": "10.1/X", "pmid": ""},
        ]
        result = deduplicate_papers(papers)
        assert len(result) == 1

    def test_dedup_by_pmid(self):
        """同 PMID 论文应被去重。"""
        papers = [
            {"title": "A", "doi": "", "pmid": "999"},
            {"title": "B", "doi": "", "pmid": "999"},
        ]
        assert len(deduplicate_papers(papers)) == 1

    def test_dedup_cross_source_by_title(self):
        """跨源重复（标题大小写/标点差异）应被软去重。"""
        papers = [
            {"title": "CAR-T therapy for Solid Tumors", "doi": "", "pmid": ""},
            {"title": "CAR-T therapy for solid tumors!", "doi": "", "pmid": ""},
        ]
        assert len(deduplicate_papers(papers)) == 1

    def test_no_false_dedup_empty_titles(self):
        """多篇无标题论文不应被误判为重复。"""
        papers = [
            {"title": "", "doi": "", "pmid": ""},
            {"title": "", "doi": "", "pmid": ""},
        ]
        assert len(deduplicate_papers(papers)) == 2


# ── 并发搜索 + 合并测试 ───────────────────────────────────────────────────


class TestSearchAll:
    """测试多源并发检索与结果合并。"""

    def test_search_all_merges_and_tags_sources(self):
        """search_all 应合并多源结果并为每篇标注 source。"""
        tool_a = _MockTool(
            "pubmed",
            [{"title": "Paper A", "abstract": "abs A", "doi": "10.1/a"}],
        )
        tool_b = _MockTool(
            "arxiv",
            [{"title": "Paper B", "summary": "abs B", "pdf_url": "http://x"}],
        )
        agent = SearchAgent(tools=[tool_a, tool_b])

        papers = agent.search_all("query")

        assert len(papers) == 2
        sources = {p["source"] for p in papers}
        assert sources == {"pubmed", "arxiv"}

    def test_search_all_dedups_across_sources(self):
        """两个源命中同一篇（同 DOI）时应去重。"""
        shared = {"title": "Same paper", "doi": "10.1/same", "abstract": "x"}
        tool_a = _MockTool("pubmed", [shared])
        tool_b = _MockTool("semantic_scholar", [dict(shared)])
        agent = SearchAgent(tools=[tool_a, tool_b])

        papers = agent.search_all("query")
        assert len(papers) == 1

    def test_single_source_failure_degrades(self):
        """单源抛异常时不影响其他源。"""
        good = _MockTool("pubmed", [{"title": "Good", "abstract": "abs"}])
        bad = _MockTool("arxiv", [], exc=RuntimeError("rate limited"))
        agent = SearchAgent(tools=[good, bad])

        papers = agent.search_all("query")
        assert len(papers) == 1
        assert papers[0]["title"] == "Good"

    def test_all_sources_fail_returns_empty(self):
        """全部源失败时返回空列表而非崩溃。"""
        t1 = _MockTool("a", [], exc=RuntimeError("down"))
        t2 = _MockTool("b", [], exc=RuntimeError("down"))
        agent = SearchAgent(tools=[t1, t2])

        assert agent.search_all("query") == []

# ── 工具构造测试 ─────────────────────────────────────────────────────────


class TestToolConstruction:
    """测试默认 5 数据源工具构造。"""

    def test_default_has_five_sources(self):
        """默认应构造 5 个数据源工具（PubMed/Arxiv/S2S/bioRxiv/medRxiv）。"""
        agent = SearchAgent()
        assert len(agent.tools) == 5
        sources = [t.name for t in agent.tools]
        assert "pubmed_search" in sources
        assert "arxiv_search" in sources
        assert "semantic_scholar_search" in sources
        assert "biorxiv_search" in sources
        assert "medrxiv_search" in sources
