"""
bioRxiv / medRxiv 预印本搜索工具 — LangChain BaseTool 封装。

bioRxiv 与 medRxiv 是「尚未正式发表」的最新研究预印本平台，做前沿综述必查。
两者共用一个检索后端（Europe PMC 聚合了 bioRxiv + medRxiv 预印本），通过
``server`` 参数区分数据源，并据 ``bookOrReportDetails.publisher`` 字段
精确过滤目标服务器。

特性:
- 统一返回 {papers, total_count, query, source} 格式
- server 前缀切换（"biorxiv" / "medrxiv"）
- 预印本过滤 (SRC:PPR) + 按 publisher 精确区分 bioRxiv / medRxiv

使用示例:
    biorxiv = BioRxivSearchTool()            # source = "biorxiv"
    medrxiv = MedRxivSearchTool()            # source = "medrxiv"
    result = biorxiv._run(query="mRNA vaccine cancer", max_results=10)
"""

import logging
import time

import requests
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_PREPRINT_FILTER = "SRC:PPR"


# ── Pydantic Input Schema ──────────────────────────────────────────────


class BioRxivSearchInput(BaseModel):
    """bioRxiv / medRxiv 预印本搜索参数。"""

    query: str = Field(
        description="生物医学预印本搜索查询词，英文。"
        '示例: "mRNA vaccine" cancer immunotherapy'
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="返回结果数量上限 (1-100)，默认 10",
    )


# ── Tool 实现 ──────────────────────────────────────────────────────────


class BioRxivSearchTool(BaseTool):
    """bioRxiv 预印本搜索工具（基类，供 MedRxivSearchTool 复用）。

    通过 Europe PMC 搜索预印本（bioRxiv / medRxiv），按 ``server`` 属性
    过滤并标注来源。子类只需覆盖 ``server`` 与 ``name`` 即可切换数据源。

    Attributes:
        server: "biorxiv" 或 "medrxiv"
        name:   Tool 标识名
    """

    name: str = "biorxiv_search"
    description: str = (
        "Search bioRxiv preprint server for the latest biomedical research "
        "that has not yet been formally peer-reviewed. Best for: cutting-edge "
        "findings, early results, and emerging topics ahead of formal publication."
    )
    args_schema: type[BaseModel] = BioRxivSearchInput

    # ── 数据源标识（子类覆盖） ────────────────────────────────────────

    server: str = "biorxiv"

    # ── 内部状态 ──────────────────────────────────────────────────────

    _last_request_time: float = 0.0
    _min_interval: float = 0.5  # Europe PMC 较宽松，保守 ~2 req/s

    # ── 公开入口 ──────────────────────────────────────────────────────

    def _run(self, query: str, max_results: int = 10) -> dict:
        """LangChain Tool 的标准执行入口。

        Returns:
            {
                "papers": [{"title", "authors", "year", "journal", "abstract",
                            "doi", "pmid", "url"}, ...],
                "total_count": int,
                "query": str,
                "source": "biorxiv" | "medrxiv",
            }
        """
        logger.info(
            "%s search: query='%s', max=%d", self.server, query, max_results,
        )
        return self._search(query, max_results)

    # ── 核心搜索逻辑 ──────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _search(self, query: str, max_results: int) -> dict:
        """执行 Europe PMC 预印本搜索，过滤到目标服务器。"""
        self._rate_limit()

        # 多取一些（3 倍）补偿按 publisher 过滤带来的损失
        fetch_size = min(max_results * 3, 100)
        full_query = f"({query}) AND ({_PREPRINT_FILTER})"

        params = {
            "query": full_query,
            "format": "json",
            "pageSize": fetch_size,
            "resultType": "core",
        }
        resp = requests.get(_BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        result_list = data.get("resultList", {}).get("result", [])
        total = int(data.get("hitCount", 0))

        # 精确过滤到目标服务器（bioRxiv / medRxiv）
        papers = [
            self._parse_paper(p)
            for p in result_list
            if self._is_target_server(p)
        ][:max_results]

        logger.info(
            "%s: %d papers after filter (total preprints=%s)",
            self.server, len(papers), total,
        )
        return {
            "papers": papers,
            "total_count": total,
            "query": query,
            "source": self.server,
        }

    def _is_target_server(self, paper: dict) -> bool:
        """判断该预印本是否属于当前 server（bioRxiv / medRxiv）。"""
        publisher = (
            (paper.get("bookOrReportDetails") or {}).get("publisher") or ""
        ).lower()
        return publisher == self.server.lower()

    # ── 数据解析 ──────────────────────────────────────────────────────

    def _parse_paper(self, paper: dict) -> dict:
        """解析单篇预印本，字段对齐统一协议。"""
        author_string = paper.get("authorString") or ""
        authors = [a.strip() for a in author_string.split(",") if a.strip()][:5]
        record_id = paper.get("id", "")

        return {
            "title": paper.get("title") or "",
            "authors": authors,
            "year": str(paper.get("pubYear") or ""),
            "journal": (paper.get("bookOrReportDetails") or {}).get("publisher")
            or self.server,
            "abstract": (paper.get("abstractText") or "")[:2000],
            "doi": paper.get("doi") or "",
            "pmid": "",
            "url": f"https://europepmc.org/article/PPR/{record_id}",
        }

    # ── 限流控制 ──────────────────────────────────────────────────────

    def _rate_limit(self):
        """确保两次请求间隔 ≥ _min_interval 秒。"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()


class MedRxivSearchTool(BioRxivSearchTool):
    """medRxiv 预印本搜索工具 — 复用 BioRxivSearchTool，切换数据源前缀。"""

    name: str = "medrxiv_search"
    description: str = (
        "Search medRxiv preprint server for the latest clinical, epidemiological "
        "and health-sciences research ahead of formal peer review. Best for: "
        "emerging clinical findings and public-health research."
    )
    server: str = "medrxiv"


# ── 快速测试入口 ──────────────────────────────────────────────────────

if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)

    for tool in (BioRxivSearchTool(), MedRxivSearchTool()):
        result = tool._run(query="mRNA vaccine cancer", max_results=3)
        print(f"\n[{tool.server}] Found {result['total_count']} preprints, "
              f"showing {len(result['papers'])}:")
        for i, paper in enumerate(result["papers"], 1):
            print(f"{i}. {paper['title']}")
            print(f"   Authors: {', '.join(paper['authors'][:3])}")
            print(f"   Journal: {paper['journal']} ({paper['year']})")
            print(f"   DOI: {paper['doi']}")
            print()
