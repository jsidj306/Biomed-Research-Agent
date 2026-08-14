"""
Semantic Scholar Search Tool — LangChain BaseTool 封装。

对 Semantic Scholar Academic Graph API 的封装，返回与 PubMed / Arxiv 统一的
{papers, total_count, query, source} 格式。相比 PubMed，Semantic Scholar 自带
「引用关系」和「影响力评分」，对找综述 / 高被引文献更友好。

特性:
- /graph/v1/paper/search 关键词搜索
- 匿名额度 100 req / 5min（有 Key 更高），内置限流保护
- 支持 relevance / citation_count 排序
- tenacity 指数退避重试

使用示例:
    tool = SemanticScholarSearchTool()
    result = tool._run(query="CAR-T therapy solid tumor", max_results=10)
"""

import logging
import os
import time

import requests
from dotenv import load_dotenv
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "title,authors,year,venue,abstract,externalIds,url,citationCount"


# ── Pydantic Input Schema ──────────────────────────────────────────────


class SemanticScholarSearchInput(BaseModel):
    """Semantic Scholar 搜索参数。"""

    query: str = Field(
        description="学术搜索查询词，英文。支持短语和布尔运算符。"
        '示例: "CAR-T cell therapy" solid tumor clinical trial'
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="返回结果数量上限 (1-100)，默认 10",
    )
    sort: str = Field(
        default="relevance",
        description="排序方式: 'relevance' (相关度) 或 'citation_count' (被引量降序)",
    )


# ── Tool 实现 ──────────────────────────────────────────────────────────


class SemanticScholarSearchTool(BaseTool):
    """Semantic Scholar 学术文献搜索工具。

    搜索 Semantic Scholar 学术图谱中的论文，返回统一格式结果。
    与 PubMedSearchTool / ArxivSearchTool 接口一致，便于多源合并。

    Attributes:
        name:        Tool 标识名
        description: Tool 功能描述
        args_schema: 输入参数 Schema
    """

    name: str = "semantic_scholar_search"
    description: str = (
        "Search the Semantic Scholar academic graph for research papers. "
        "Returns papers with titles, authors, abstracts, venues, DOIs and "
        "citation counts. Best for: finding highly-cited review papers, "
        "exploring citation networks, and cross-source deduplication via DOI."
    )
    args_schema: type[BaseModel] = SemanticScholarSearchInput

    # ── 内部状态 ──────────────────────────────────────────────────────

    _last_request_time: float = 0.0
    _min_interval: float = 1.2  # 匿名额度 ~100 req/5min，保守 ~50 req/min

    # ── 公开入口 ──────────────────────────────────────────────────────

    def _run(
        self,
        query: str,
        max_results: int = 10,
        sort: str = "relevance",
    ) -> dict:
        """LangChain Tool 的标准执行入口。

        Returns:
            {
                "papers": [{"title", "authors", "year", "journal", "abstract",
                            "doi", "pmid", "url", "citation_count"}, ...],
                "total_count": int,
                "query": str,
                "source": "semantic_scholar",
            }
        """
        logger.info(
            "SemanticScholar search: query='%s', max=%d, sort=%s",
            query, max_results, sort,
        )
        return self._search(query, max_results, sort)

    # ── 核心搜索逻辑 ──────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _search(self, query: str, max_results: int, sort: str) -> dict:
        """执行 Semantic Scholar 搜索并解析结果。"""
        self._rate_limit()

        params = {
            "query": query,
            "limit": max_results,
            "fields": _FIELDS,
        }
        if sort == "citation_count":
            params["sort"] = "citationCount:desc"

        api_key = os.getenv("S2_API_KEY", "")
        headers = {"x-api-key": api_key} if api_key else {}

        resp = requests.get(_BASE_URL, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        papers = [self._parse_paper(p) for p in data.get("data", [])]
        total = data.get("total", len(papers))

        logger.info("SemanticScholar: found %d papers (total=%s)", len(papers), total)
        return {
            "papers": papers,
            "total_count": total,
            "query": query,
            "source": "semantic_scholar",
        }

    # ── 数据解析 ──────────────────────────────────────────────────────

    def _parse_paper(self, paper: dict) -> dict:
        """解析单篇论文，字段对齐统一协议。"""
        external_ids = paper.get("externalIds") or {}
        authors = [a.get("name", "") for a in (paper.get("authors") or [])]
        authors = [a for a in authors if a][:5]

        return {
            "title": paper.get("title") or "",
            "authors": authors,
            "year": str(paper.get("year") or ""),
            "journal": paper.get("venue") or "",
            "abstract": (paper.get("abstract") or "")[:2000],
            "doi": external_ids.get("DOI") or "",
            "pmid": external_ids.get("PubMed") or "",
            "url": paper.get("url") or "",
            "citation_count": paper.get("citationCount") or 0,
        }

    # ── 限流控制 ──────────────────────────────────────────────────────

    def _rate_limit(self):
        """确保两次请求间隔 ≥ _min_interval 秒。"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()


# ── 快速测试入口 ──────────────────────────────────────────────────────

if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)

    tool = SemanticScholarSearchTool()
    result = tool._run(query="CAR-T therapy solid tumor", max_results=3)

    print(f"\nFound {result['total_count']} papers, showing {len(result['papers'])}:\n")
    for i, paper in enumerate(result["papers"], 1):
        print(f"{i}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'][:3])}")
        print(f"   Venue: {paper['journal']} ({paper['year']})")
        print(f"   Citations: {paper['citation_count']}")
        print(f"   DOI: {paper['doi']}")
        print()
