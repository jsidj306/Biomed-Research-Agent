"""
Arxiv Search Tool — LangChain BaseTool 封装。

对 arxiv 官方 Python 库的封装，与 PubMedSearchTool 接口统一：
- 返回 {"papers": [...], "total_count": N, "query": "...", "source": "arxiv"}
- 支持 relevance / lastUpdatedDate / submittedDate 三种排序
- 完全免费，无需 API Key

使用示例:
    tool = ArxivSearchTool()
    result = tool._run(query="large language model biomedical NLP", max_results=10)
    for paper in result["papers"]:
        print(paper["title"], paper["pdf_url"])
"""

import logging

import arxiv
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Pydantic Input Schema ──────────────────────────────────────────────


class ArxivSearchInput(BaseModel):
    """Arxiv 搜索参数。"""

    query: str = Field(
        description="学术搜索查询词，英文。支持布尔运算符和字段限定。"
        '分类限定示例: "cat:cs.CL AND transformer attention mechanism"'
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="返回结果数量上限 (1-100)，默认 10",
    )
    sort_by: str = Field(
        default="relevance",
        description="排序方式: 'relevance' (相关度), 'lastUpdatedDate' (最近更新), 'submittedDate' (提交日期)",
    )


# ── Tool 实现 ──────────────────────────────────────────────────────────


class ArxivSearchTool(BaseTool):
    """Arxiv 预印本学术搜索工具。

    搜索 arXiv 上最新的 AI/ML/NLP/生物信息学论文。
    与 PubMedSearchTool 返回统一格式，方便多源结果合并。

    Attributes:
        name: Tool 标识名
        description: Tool 功能描述 (Agent 据此判断是否调用)
        args_schema: 输入参数 Schema
    """

    name: str = "arxiv_search"
    description: str = (
        "Search arXiv preprint server for cutting-edge research papers "
        "in AI, machine learning, NLP, bioinformatics, and computational biology. "
        "Best for: finding the latest methods before formal publication, "
        "interdisciplinary ML+biomedicine papers, and preprints not yet in PubMed."
    )
    args_schema: type[BaseModel] = ArxivSearchInput

    # ── 公开入口 ──────────────────────────────────────────────────────

    def _run(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance",
    ) -> dict:
        """LangChain Tool 的标准执行入口。

        Args:
            query:       学术搜索查询词
            max_results: 返回结果数 (1-100)
            sort_by:     排序方式

        Returns:
            {
                "papers": [
                    {
                        "arxiv_id": "2401.00001",
                        "title": "...",
                        "authors": ["...", ...],
                        "published": "2024-01-01",
                        "categories": ["cs.CL", "q-bio.QM"],
                        "summary": "...",
                        "pdf_url": "https://arxiv.org/pdf/2401.00001",
                        "comment": "...",
                    },
                    ...
                ],
                "total_count": int,
                "query": str,
                "source": "arxiv",
            }
        """
        logger.info(
            "Arxiv search: query='%s', max=%d, sort=%s",
            query, max_results, sort_by,
        )

        sort_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastupdateddate": arxiv.SortCriterion.LastUpdatedDate,
            "submitteddate": arxiv.SortCriterion.SubmittedDate,
        }
        sort_criterion = sort_map.get(
            sort_by.lower().replace("_", ""),
            arxiv.SortCriterion.Relevance,
        )

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_criterion,
        )

        results = []
        for r in client.results(search):
            # Arxiv 的 entry_id 格式: "http://arxiv.org/abs/2401.00001v2"
            arxiv_id = r.entry_id.split("/")[-1]

            results.append({
                "arxiv_id": arxiv_id,
                "title": r.title,
                "authors": [str(a) for a in r.authors[:5]],
                "published": str(r.published),
                "updated": str(r.updated),
                "categories": r.categories,
                "summary": r.summary[:1500],
                "pdf_url": r.pdf_url,
                "comment": r.comment or "",
            })

        logger.info("Arxiv: found %d papers", len(results))
        return {
            "papers": results,
            "total_count": len(results),
            "query": query,
            "source": "arxiv",
        }


# ── 快速测试入口 ──────────────────────────────────────────────────────

if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)

    tool = ArxivSearchTool()
    result = tool._run(
        query="transformer attention mechanism biomedical NLP",
        max_results=3,
    )

    print(f"\nFound {result['total_count']} papers:\n")
    for i, paper in enumerate(result["papers"], 1):
        print(f"{i}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'][:3])}")
        print(f"   Categories: {', '.join(paper['categories'])}")
        print(f"   Published: {paper['published']}")
        print(f"   PDF: {paper['pdf_url']}")
        print(f"   Summary: {paper['summary'][:150]}...")
        print()
