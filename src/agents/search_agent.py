"""
Search Agent — 多源并行文献检索。

并发调用 5 个数据源（PubMed / Arxiv / Semantic Scholar / bioRxiv / medRxiv），
把各自的返回归一化到统一 schema，再去重合并。单源失败自动降级（返回空列表
+ 记日志），不影响其余数据源。

统一 schema 字段::

    title / authors / journal / year / abstract / url / source / doi / pmid

多Agent协作的前提是「每个节点说同一种语言」——下游 Review / Summarize
无需区分来源即可处理，这正是 MVP 阶段统一返回格式的红利。

使用示例:
    from src.agents.search_agent import SearchAgent

    agent = SearchAgent()
    papers = agent.search_all("CAR-T therapy solid tumor", max_results=10)
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# 统一返回字段（下游 Agent 只依赖这些键）
UNIFIED_FIELDS = [
    "title", "authors", "journal", "year", "abstract", "url", "source", "doi", "pmid",
]


# ── 字段归一化 ───────────────────────────────────────────────────────────


def _extract_year(date_str: str) -> str:
    """从 ISO 日期字符串中提取年份（arxiv 的 published 字段）。"""
    m = re.match(r"(\d{4})", date_str or "")
    return m.group(1) if m else ""


def normalize_paper(raw: dict[str, Any], source: str) -> dict[str, Any]:
    """把某个数据源的论文原始字段归一化到统一 schema。

    Args:
        raw:    工具返回的单篇论文 dict（字段因源而异）
        source: 数据源标识（pubmed / arxiv / semantic_scholar / biorxiv / medrxiv）

    Returns:
        统一 schema 的论文 dict
    """
    title = raw.get("title") or ""
    authors = raw.get("authors") or []
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(",") if a.strip()]
    authors = list(authors)[:5]

    # arxiv 字段命名不同，需单独映射
    if source == "arxiv":
        year = _extract_year(raw.get("published") or "")
        journal = "arXiv preprint"
        abstract = raw.get("summary") or ""
        url = raw.get("pdf_url") or ""
        doi = ""
    else:
        year = raw.get("year") or ""
        journal = raw.get("journal") or ""
        abstract = raw.get("abstract") or ""
        url = raw.get("url") or ""
        doi = raw.get("doi") or ""

    return {
        "title": title,
        "authors": authors,
        "journal": journal,
        "year": str(year),
        "abstract": abstract,
        "url": url,
        "source": source,
        "doi": doi,
        "pmid": raw.get("pmid") or "",
    }


# ── 去重 ────────────────────────────────────────────────────────────────


def _normalize_title(title: str) -> str:
    """标题归一化：小写、去标点/空白，用于跨源软去重。"""
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def _dedup_key(paper: dict[str, Any]) -> str:
    """生成去重键，按 DOI → PMID → 归一化标题 的优先级。"""
    doi = (paper.get("doi") or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    pmid = (paper.get("pmid") or "").strip()
    if pmid:
        return f"pmid:{pmid}"
    return f"title:{_normalize_title(paper.get('title') or '')}"


def deduplicate_papers(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 DOI / PMID / 归一化标题去重（保留首次出现，稳定顺序）。"""
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for paper in papers:
        key = _dedup_key(paper)
        if key in seen:
            continue
        # 标题为空的兜底键，避免把多篇无标题论文误判为重复
        if key == "title:":
            key = f"title:{id(paper)}"
        seen.add(key)
        result.append(paper)
    return result


# ── Search Agent ─────────────────────────────────────────────────────────


class SearchAgent:
    """多源并行文献检索 Agent。

    并发调用 5 个数据源，归一化 + 去重后返回统一论文列表。

    Attributes:
        tools: LangChain BaseTool 实例列表（默认 5 个数据源）
    """

    def __init__(self, tools: list | None = None):
        """初始化 SearchAgent。

        Args:
            tools: 工具列表；缺省时构造 5 个默认数据源工具
        """
        self.tools = tools if tools is not None else self._default_tools()
        logger.info("SearchAgent ready: %d sources", len(self.tools))

    @staticmethod
    def _default_tools() -> list:
        """构造 5 个默认数据源工具。"""
        from src.tools.arxiv_tool import ArxivSearchTool
        from src.tools.biorxiv_tool import BioRxivSearchTool, MedRxivSearchTool
        from src.tools.pubmed_tool import PubMedSearchTool
        from src.tools.semantic_scholar_tool import SemanticScholarSearchTool

        return [
            PubMedSearchTool(),
            ArxivSearchTool(),
            SemanticScholarSearchTool(),
            BioRxivSearchTool(),
            MedRxivSearchTool(),
        ]

    # ── 公开接口 ────────────────────────────────────────────────────────

    def search_all(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        """同步入口：多源并发检索，返回归一化 + 去重后的论文列表。

        Args:
            query:       搜索查询词
            max_results: 每个数据源返回的结果上限

        Returns:
            List[dict] — 统一 schema 的论文列表（跨源去重后）
        """
        return asyncio.run(self.asearch_all(query, max_results))

    async def asearch_all(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        """异步入口：asyncio.gather 并发调用 5 个数据源。

        单源失败（限流/网络）时降级为跳过该源，不影响其余源。
        """
        tasks = [
            self._run_tool_async(tool, query, max_results)
            for tool in self.tools
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        papers: list[dict[str, Any]] = []
        for tool, res in zip(self.tools, results):
            if isinstance(res, BaseException):
                logger.warning(
                    "Source '%s' failed (degraded): %s",
                    getattr(tool, "name", tool.__class__.__name__), res,
                )
                continue
            source = res.get("source", getattr(tool, "name", "unknown"))
            for raw in res.get("papers", []):
                papers.append(normalize_paper(raw, source))

        merged = deduplicate_papers(papers)
        logger.info(
            "Search complete: %d sources → %d papers after dedup",
            len(self.tools), len(merged),
        )
        return merged

    # ── 内部实现 ────────────────────────────────────────────────────────

    @staticmethod
    async def _run_tool_async(tool, query: str, max_results: int):
        """在后台线程运行同步 Tool，避免阻塞事件循环。"""
        return await asyncio.to_thread(tool._run, query, max_results)
