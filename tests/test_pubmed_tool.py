"""Tests for PubMed Search Tool."""

from unittest.mock import MagicMock, patch

import pytest

from src.tools.pubmed_tool import PubMedSearchInput, PubMedSearchTool


class TestPubMedSearchInput:
    """Pydantic 输入模型验证。"""

    def test_valid_input_minimal(self):
        """最简输入 — 只有 query。"""
        inp = PubMedSearchInput(query="CRISPR")
        assert inp.query == "CRISPR"
        assert inp.max_results == 10  # 默认值
        assert inp.sort == "relevance"

    def test_valid_input_full(self):
        """完整输入 — 所有字段。"""
        inp = PubMedSearchInput(
            query="cancer immunotherapy",
            max_results=20,
            sort="date",
        )
        assert inp.max_results == 20
        assert inp.sort == "date"

    def test_max_results_out_of_range(self):
        """max_results 超出 1-100 范围应抛验证错误。"""
        with pytest.raises(Exception):
            PubMedSearchInput(query="test", max_results=0)
        with pytest.raises(Exception):
            PubMedSearchInput(query="test", max_results=101)


class TestPubMedSearchTool:
    """PubMedSearchTool 单元测试（Mock Bio.Entrez 外部调用）。"""

    def test_tool_creation(self):
        """Tool 创建 — name / description / args_schema 正确。"""
        tool = PubMedSearchTool()
        assert tool.name == "pubmed_search"
        assert "PubMed" in tool.description
        assert tool.args_schema == PubMedSearchInput

    @patch("Bio.Entrez.read")
    @patch("Bio.Entrez.esearch")
    def test_empty_results(self, mock_esearch, mock_read):
        """无结果时返回空列表而非异常。"""
        mock_read.return_value = {"IdList": [], "Count": "0"}

        tool = PubMedSearchTool()
        result = tool._search_pubmed("xyznonexistentquery1234567890", max_results=5)
        assert result["total_count"] == 0
        assert result["papers"] == []
        assert result["source"] == "pubmed"
        assert result["query"] == "xyznonexistentquery1234567890"

    @patch("Bio.Entrez.read")
    @patch("Bio.Entrez.efetch")
    @patch("Bio.Entrez.esearch")
    def test_search_returns_structured_results(
        self, mock_esearch, mock_efetch, mock_read
    ):
        """验证返回的 JSON 结构包含所有必需字段。"""
        # Arrange — ESearch 返回 2 个 PMID
        mock_read.side_effect = [
            {  # ESearch 返回值
                "IdList": ["12345", "67890"],
                "Count": "2",
            },
            {  # EFetch 返回值（1 篇论文）
                "PubmedArticle": [
                    {
                        "MedlineCitation": {
                            "PMID": "12345",
                            "Article": {
                                "ArticleTitle": "CRISPR-based Gene Therapy for Cancer",
                                "AuthorList": [
                                    {"LastName": "Smith", "ForeName": "John"},
                                    {"LastName": "Wang", "ForeName": "Li"},
                                ],
                                "Journal": {
                                    "Title": "Nature Medicine",
                                    "JournalIssue": {
                                        "PubDate": {"Year": "2024"}
                                    },
                                },
                                "Abstract": {
                                    "AbstractText": [
                                        "CRISPR technology enables precise gene editing.",
                                        "This study demonstrates efficacy in mouse models.",
                                    ]
                                },
                            },
                        },
                        "PubmedData": {
                            "ArticleIdList": [],
                        },
                    },
                    {
                        "MedlineCitation": {
                            "PMID": "67890",
                            "Article": {
                                "ArticleTitle": "",
                                "AuthorList": [],
                                "Journal": {
                                    "Title": "",
                                    "JournalIssue": {"PubDate": {}},
                                },
                            },
                        },
                        "PubmedData": {"ArticleIdList": []},
                    },
                ]
            },
        ]

        # Act
        tool = PubMedSearchTool()
        result = tool._run(query="CRISPR cancer", max_results=5)

        # Assert — 顶层结构
        assert result["total_count"] == 2
        assert result["source"] == "pubmed"
        assert len(result["papers"]) == 2

        # Assert — 有标题的论文
        paper = result["papers"][0]
        assert paper["pmid"] == "12345"
        assert "CRISPR-based" in paper["title"]
        assert len(paper["authors"]) == 2
        assert paper["authors"][0] == "Smith John"
        assert paper["year"] == "2024"
        assert paper["journal"] == "Nature Medicine"
        assert "gene editing" in paper["abstract"]
        assert paper["url"] == "https://pubmed.ncbi.nlm.nih.gov/12345/"

        # Assert — 无标题的论文 (边缘情况)
        paper2 = result["papers"][1]
        assert paper2["pmid"] == "67890"
        assert paper2["title"] == ""

    @patch("Bio.Entrez.read")
    @patch("Bio.Entrez.esearch")
    def test_sort_parameter_passed_correctly(
        self, mock_esearch, mock_read
    ):
        """验证 sort='date' 参数正确传递。"""
        mock_read.return_value = {"IdList": ["12345"], "Count": "1"}

        tool = PubMedSearchTool()
        tool._esearch("cancer", max_results=5, sort="date")

        # 验证 Entrez.esearch 被调用时的参数
        call_kwargs = mock_esearch.call_args.kwargs
        assert call_kwargs["sort"] == "date"

    @patch("Bio.Entrez.read")
    @patch("Bio.Entrez.esearch")
    def test_rate_limit_enforced(self, mock_esearch, mock_read):
        """验证限流：连续调用时 _last_request_time 被更新。"""
        mock_read.return_value = {"IdList": ["12345"], "Count": "1"}

        tool = PubMedSearchTool()
        initial_time = tool._last_request_time

        tool._esearch("test1", max_results=5, sort="relevance")
        after_first = tool._last_request_time
        assert after_first > initial_time

        tool._esearch("test2", max_results=5, sort="relevance")
        after_second = tool._last_request_time
        assert after_second > after_first

    def test_parse_articles_handles_missing_fields(self):
        """验证缺失字段不导致崩溃（防御性解析）。"""
        tool = PubMedSearchTool()

        # 最小合法记录 — 只有 PMID
        minimal_record = {
            "PubmedArticle": [
                {
                    "MedlineCitation": {
                        "PMID": "99999",
                        "Article": {
                            "ArticleTitle": "Minimal Paper",
                        },
                    },
                    "PubmedData": {"ArticleIdList": []},
                }
            ]
        }

        results = tool._parse_articles(minimal_record)
        assert len(results) == 1
        assert results[0]["pmid"] == "99999"
        assert results[0]["title"] == "Minimal Paper"
        assert results[0]["authors"] == []
        assert results[0]["journal"] == ""
        assert results[0]["abstract"] == ""

    def test_parse_articles_empty(self):
        """空 PubmedArticle 列表返回空列表。"""
        tool = PubMedSearchTool()
        results = tool._parse_articles({"PubmedArticle": []})
        assert results == []

    def test_min_interval_default(self):
        """默认限流间隔为 0.34 秒 (~3 req/s)。"""
        tool = PubMedSearchTool()
        assert tool._min_interval == 0.34


class TestRetryBehavior:
    """重试行为测试（验证 tenacity 装饰器生效）。"""

    @patch("Bio.Entrez.read")
    @patch("Bio.Entrez.esearch")
    def test_retry_on_esearch_failure(self, mock_esearch, mock_read):
        """ESearch 前两次失败、第三次成功 → 应重试 3 次。"""
        mock_esearch.side_effect = [
            Exception("Network error 1"),
            Exception("Network error 2"),
            MagicMock(),  # 第三次成功
        ]
        mock_read.return_value = {"IdList": ["12345"], "Count": "1"}

        # 还需要 mock efetch 的 Entrez.read
        tool = PubMedSearchTool()
        # 直接测 _esearch — 但 _run 会调 _efetch 也会失败
        # 所以我们 mock _efetch 让它返回空
        with patch.object(tool, "_efetch", return_value=[]):
            result = tool._run(query="cancer", max_results=5)

        assert mock_esearch.call_count == 3
        assert result["total_count"] == 1

    @patch("Bio.Entrez.esearch")
    def test_retry_exhausted_raises_exception(self, mock_esearch):
        """ESearch 全部失败 → 最终抛出异常。"""
        mock_esearch.side_effect = Exception("Persistent network failure")

        tool = PubMedSearchTool()
        with pytest.raises(Exception):
            tool._run(query="cancer", max_results=5)

        assert mock_esearch.call_count == 3  # 3 次重试
