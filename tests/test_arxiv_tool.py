"""Tests for Arxiv Search Tool."""

from unittest.mock import MagicMock, patch

import pytest

from src.tools.arxiv_tool import ArxivSearchInput, ArxivSearchTool


class TestArxivSearchInput:
    """Pydantic 输入模型验证。"""

    def test_valid_input_minimal(self):
        inp = ArxivSearchInput(query="deep learning bioinformatics")
        assert inp.query == "deep learning bioinformatics"
        assert inp.max_results == 10
        assert inp.sort_by == "relevance"

    def test_valid_input_full(self):
        inp = ArxivSearchInput(
            query="protein folding AI",
            max_results=20,
            sort_by="submittedDate",
        )
        assert inp.max_results == 20
        assert inp.sort_by == "submittedDate"

    def test_max_results_out_of_range(self):
        with pytest.raises(Exception):
            ArxivSearchInput(query="test", max_results=0)
        with pytest.raises(Exception):
            ArxivSearchInput(query="test", max_results=101)


class TestArxivSearchTool:
    """ArxivSearchTool 单元测试（Mock arxiv.Client）。"""

    def test_tool_creation(self):
        tool = ArxivSearchTool()
        assert tool.name == "arxiv_search"
        assert "arXiv" in tool.description
        assert tool.args_schema == ArxivSearchInput

    @patch("arxiv.Client")
    def test_search_returns_unified_format(self, mock_client_cls):
        """验证返回格式与 PubMed Tool 一致。"""
        # Arrange
        mock_result = MagicMock()
        mock_result.entry_id = "http://arxiv.org/abs/2401.00001v2"
        mock_result.title = "Test Arxiv Paper on Biomedical NLP"
        mock_result.authors = [MagicMock(__str__=lambda s: "Smith John")]
        mock_result.published = MagicMock()
        mock_result.updated = MagicMock()
        mock_result.categories = ["cs.CL", "q-bio.QM"]
        mock_result.summary = "This is a test paper about biomedical NLP."
        mock_result.pdf_url = "http://arxiv.org/pdf/2401.00001"
        mock_result.comment = "Accepted at ACL 2024"

        mock_client = MagicMock()
        mock_client.results.return_value = [mock_result]
        mock_client_cls.return_value = mock_client

        # Act
        tool = ArxivSearchTool()
        result = tool._run(query="biomedical NLP", max_results=5)

        # Assert — top-level
        assert result["source"] == "arxiv"
        assert result["total_count"] == 1
        assert result["query"] == "biomedical NLP"
        assert len(result["papers"]) == 1

        # Assert — paper fields
        paper = result["papers"][0]
        assert paper["arxiv_id"] == "2401.00001v2"
        assert "Test Arxiv Paper" in paper["title"]
        assert "Smith John" in paper["authors"]
        assert "cs.CL" in paper["categories"]
        assert "q-bio.QM" in paper["categories"]
        assert "test paper" in paper["summary"]
        assert "arxiv.org/pdf" in paper["pdf_url"]
        assert "ACL 2024" in paper["comment"]

    def test_sort_mapping(self):
        """验证排序参数映射正确。"""
        tool = ArxivSearchTool()
        import arxiv

        # 通过 _run 间接测试 sort_by 参数
        with patch("arxiv.Client") as mock_cls:
            mock_client = MagicMock()
            mock_client.results.return_value = []
            mock_cls.return_value = mock_client

            # Test relevance (default)
            with patch("arxiv.Search") as mock_search:
                tool._run("test", max_results=5, sort_by="relevance")
                call_kwargs = mock_search.call_args.kwargs
                assert call_kwargs["sort_by"] == arxiv.SortCriterion.Relevance

            with patch("arxiv.Search") as mock_search:
                tool._run("test", max_results=5, sort_by="submittedDate")
                call_kwargs = mock_search.call_args.kwargs
                assert call_kwargs["sort_by"] == arxiv.SortCriterion.SubmittedDate

    def test_results_preserved_in_order(self):
        """验证结果保序。"""
        with patch("arxiv.Client") as mock_cls:
            r1 = MagicMock()
            r1.entry_id = "http://arxiv.org/abs/0001.00001"
            r1.title = "Paper A"
            r1.authors = []
            r1.published = MagicMock()
            r1.updated = MagicMock()
            r1.categories = ["cs.AI"]
            r1.summary = "Summary A"
            r1.pdf_url = "http://arxiv.org/pdf/0001.00001"
            r1.comment = ""

            r2 = MagicMock()
            r2.entry_id = "http://arxiv.org/abs/0002.00002"
            r2.title = "Paper B"
            r2.authors = []
            r2.published = MagicMock()
            r2.updated = MagicMock()
            r2.categories = ["q-bio.QM"]
            r2.summary = "Summary B"
            r2.pdf_url = "http://arxiv.org/pdf/0002.00002"
            r2.comment = ""

            mock_client = MagicMock()
            mock_client.results.return_value = [r1, r2]
            mock_cls.return_value = mock_client

            tool = ArxivSearchTool()
            result = tool._run("test")

            assert result["papers"][0]["title"] == "Paper A"
            assert result["papers"][1]["title"] == "Paper B"
