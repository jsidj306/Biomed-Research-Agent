"""
PubMed Search Tool — LangChain BaseTool 封装。

对 NCBI E-utilities API 的标准封装，支持：
- ESearch (搜索论文ID列表) + EFetch (批量获取详情)
- 限流保护 (tenacity 指数退避重试)
- 结构化 JSON 输出 (统一格式，方便下游消费)
- 完整类型标注 + Google 风格 docstring

使用示例:
    tool = PubMedSearchTool()
    result = tool._run(query="CRISPR-Cas9 cancer therapy", max_results=10)
    for paper in result["papers"]:
        print(paper["title"])
"""

import logging
import os
import time
from typing import Optional, Type

from Bio import Entrez
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


# ── Pydantic Input Schema ──────────────────────────────────────────────


class PubMedSearchInput(BaseModel):
    """PubMed 搜索参数 — LangChain 会据此生成 tool 的参数 JSON Schema。

    Field description 直接影响 Agent 调用工具时的参数填充质量。
    """

    query: str = Field(
        description="生物医学文献搜索查询词，英文。支持 MeSH 术语、布尔运算符 (AND/OR/NOT)、"
        "字段限定 ([ti] 标题, [au] 作者, [dp] 出版日期)。"
        '示例: "CAR-T cell therapy[ti] AND solid tumors AND clinical trial"'
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="返回结果数量上限 (1-100)，默认 10",
    )
    sort: str = Field(
        default="relevance",
        description="排序方式: 'relevance' (相关度), 'date' (最新), 'pub_date' (出版日期)",
    )


# ── Tool 实现 ──────────────────────────────────────────────────────────


class PubMedSearchTool(BaseTool):
    """PubMed 生物医学文献搜索工具。

    封装 NCBI E-utilities API 两步查询流程:
    1. ESearch — 用查询词搜索 PMID 列表
    2. EFetch  — 按 PMID 批量获取论文元数据

    内置限流保护 (有 Key: 10 req/s) 和指数退避重试。

    Attributes:
        name: Tool 标识名 (Agent 引用用)
        description: Tool 功能描述 (Agent 判断是否调用)
        args_schema: 输入参数 Pydantic Schema (Agent 生成调用参数)
    """

    name: str = "pubmed_search"
    description: str = (
        "Search PubMed database for biomedical research papers. "
        "Input a biomedical query in English and get back a list of "
        "relevant papers with titles, authors, abstracts, and metadata. "
        "Best for: clinical trials, drug discovery, disease mechanisms, "
        "gene function studies, and any biomedical literature searches."
    )
    args_schema: Type[BaseModel] = PubMedSearchInput

    # ── 内部状态 ──────────────────────────────────────────────────────

    _last_request_time: float = 0.0
    _min_interval: float = 0.34  # 有 Key: 10 req/s，保守用 ~3 req/s

    # ── 公开入口 ──────────────────────────────────────────────────────

    def _run(
        self,
        query: str,
        max_results: int = 10,
        sort: str = "relevance",
    ) -> dict:
        """LangChain Tool 的标准执行入口。

        Args:
            query:       生物医学搜索查询词
            max_results: 返回结果数 (1-100)
            sort:        排序方式

        Returns:
            {
                "papers": [
                    {
                        "pmid": "...",
                        "title": "...",
                        "authors": ["...", ...],
                        "year": "...",
                        "journal": "...",
                        "abstract": "...",
                        "doi": "...",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/...",
                    },
                    ...
                ],
                "total_count": int,
                "query": str,
                "source": "pubmed",
            }
        """
        Entrez.email = os.getenv("NCBI_EMAIL", "")
        Entrez.api_key = os.getenv("NCBI_API_KEY", "")

        logger.info(
            "PubMed search: query='%s', max=%d, sort=%s",
            query, max_results, sort,
        )
        return self._search_pubmed(query, max_results, sort)

    # ── 核心搜索逻辑 ──────────────────────────────────────────────────

    def _search_pubmed(
        self,
        query: str,
        max_results: int = 10,
        sort: str = "relevance",
    ) -> dict:
        """执行 PubMed 搜索：ESearch → EFetch → 解析。

        使用 @retry 装饰器处理 NCBI 限流 (HTTP 429) 和临时网络故障。
        """
        # Step 1: ESearch — 搜索 PMID 列表
        id_list, total_count = self._esearch(query, max_results, sort)

        if not id_list:
            logger.info("PubMed search returned 0 results for '%s'", query)
            return {
                "papers": [],
                "total_count": 0,
                "query": query,
                "source": "pubmed",
            }

        logger.info(
            "ESearch: total=%s, fetched=%d IDs", total_count, len(id_list),
        )

        # Step 2: EFetch — 批量获取论文详情
        papers = self._efetch(id_list)

        logger.info("EFetch: retrieved %d papers", len(papers))
        return {
            "papers": papers,
            "total_count": total_count,
            "query": query,
            "source": "pubmed",
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _esearch(self, query: str, max_results: int, sort: str):
        """ESearch: 搜索 PubMed，返回 PMID 列表 + 总数。

        重试策略: 最多 3 次，等待 2s → 4s → 8s (指数退避)。
        """
        self._rate_limit()

        sort_map = {
            "relevance": "relevance",
            "date": "date",
            "pub_date": "pub_date",
        }

        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort=sort_map.get(sort, "relevance"),
        )
        record = Entrez.read(handle)
        handle.close()

        return record["IdList"], int(record["Count"])

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _efetch(self, id_list: list) -> list:
        """EFetch: 批量获取论文详情，返回结构化列表。

        按批次 (batch_size=50) 调用，避免单次请求 ID 过多。
        """
        batch_size = 50
        all_papers = []

        for i in range(0, len(id_list), batch_size):
            batch = id_list[i : i + batch_size]
            self._rate_limit()

            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(batch),
                rettype="xml",
                retmode="xml",
            )
            records = Entrez.read(handle)
            handle.close()

            papers = self._parse_articles(records)
            all_papers.extend(papers)

            # 批次间额外延迟
            if i + batch_size < len(id_list):
                time.sleep(0.5)

        return all_papers

    # ── 数据解析 ──────────────────────────────────────────────────────

    def _parse_articles(self, records: dict) -> list:
        """解析 NCBI EFetch XML 返回的 PubmedArticle 列表。

        提取: PMID, 标题, 作者, 年份, 期刊, 摘要, DOI, URL。
        摘要截断至 2000 字 (某些论文摘要极长)。
        """
        results = []

        for article in records.get("PubmedArticle", []):
            medline = article["MedlineCitation"]
            article_data = medline["Article"]

            # --- PMID ---
            pmid = str(medline["PMID"])

            # --- Title ---
            title = str(article_data.get("ArticleTitle", ""))

            # --- Authors (前 5 位) ---
            authors = []
            author_list = article_data.get("AuthorList", [])
            for author in author_list[:5]:
                last = author.get("LastName", "")
                fore = author.get("ForeName", "")
                if last:
                    authors.append(f"{last} {fore}".strip())

            # --- Year ---
            pub_date = (
                article_data.get("Journal", {})
                .get("JournalIssue", {})
                .get("PubDate", {})
            )
            year = str(pub_date.get("Year", ""))

            # --- Journal ---
            journal = str(
                article_data.get("Journal", {}).get("Title", "")
            )

            # --- Abstract (截断至 2000 字) ---
            abstract = ""
            abstract_obj = article_data.get("Abstract", {})
            if abstract_obj:
                abstract_parts = abstract_obj.get("AbstractText", [])
                if isinstance(abstract_parts, list):
                    abstract = " ".join(str(p) for p in abstract_parts)
                else:
                    abstract = str(abstract_parts)
            abstract = abstract[:2000]

            # --- DOI ---
            doi = ""
            for eid in article.get("PubmedData", {}).get("ArticleIdList", []):
                if hasattr(eid, "attributes"):
                    if eid.attributes.get("IdType") == "doi":
                        doi = str(eid)

            # --- URL ---
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            results.append({
                "pmid": pmid,
                "title": title,
                "authors": authors,
                "year": year,
                "journal": journal,
                "abstract": abstract,
                "doi": doi,
                "url": url,
            })

        return results

    # ── 限流控制 ──────────────────────────────────────────────────────

    def _rate_limit(self):
        """确保两次 API 请求间隔 ≥ _min_interval 秒。

        类级别 _last_request_time 保证同一个 Tool 实例不会超速。
        NCBI 限制: 有 Key = 10 req/s，无 Key = 3 req/s。
        本实现保守用 ~3 req/s (0.34s 间隔)。
        """
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()


# ── 快速测试入口 ──────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    tool = PubMedSearchTool()
    result = tool._run(
        query="mRNA vaccine cancer immunotherapy clinical trial 2024",
        max_results=3,
    )

    print(f"\nFound {result['total_count']} papers, showing {len(result['papers'])}:\n")
    for i, paper in enumerate(result["papers"], 1):
        print(f"{i}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'][:3])}")
        print(f"   Journal: {paper['journal']} ({paper['year']})")
        print(f"   PMID: {paper['pmid']}")
        print(f"   Abstract: {paper['abstract'][:150]}...")
        print()
