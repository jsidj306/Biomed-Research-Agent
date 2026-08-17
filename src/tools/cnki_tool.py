"""
CNKI/万方 中文文献搜索工具（ChinaXiv 开放接口兜底）。

CNKI（中国知网）与万方数据均不对外提供公开 API（需机构订阅 + 反爬），因此本项目
采用 **ChinaXiv（中国科学院科技论文预发布平台）** 的 OAI-PMH 开放接口作为中文文献
的实际数据源，统一返回与 PubMed/arXiv 一致的 schema。

设计要点：
  - OAI-PMH 只支持按 set（学科/期刊集合）收割，不支持服务器端自由词检索；故采用
    「收割 + 客户端按 query 关键词过滤排序」的方式近似实现搜索
  - 纯函数 `_parse_oaidc_xml` / `_filter_by_query` 可独立测试
  - 网络失败 / 无结果时静默降级为空列表（与其他数据源一致的降级策略）

使用示例:
    tool = CNKISearchTool()
    result = tool._run(query="细胞治疗 肿瘤", max_results=10)
"""

from __future__ import annotations

import logging
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

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

# ChinaXiv OAI-PMH 端点（支持中文文献元数据收割，OAI-DC 格式）
CHINAXIV_OAI_URL = "http://www.chinaxiv.org/oai"

# OAI-DC 命名空间
_DC_NS = {"dc": "http://purl.org/dc/elements/1.1/"}
_OAI_NS = {"oai": "http://www.openarchives.org/OAI/2.0/"}


# ── Pydantic Input Schema ──────────────────────────────────────────────


class CNKISearchInput(BaseModel):
    """中文文献搜索参数。"""

    query: str = Field(
        description="中文生物医学文献搜索查询词，例如 '细胞治疗 肿瘤' 或 'CAR-T 实体瘤'"
    )
    max_results: int = Field(default=10, ge=1, le=100, description="返回结果数量上限")


# ── 纯函数：OAI-DC 解析 ─────────────────────────────────────────────────


def _parse_oaidc_xml(xml_bytes: bytes) -> list[dict[str, Any]]:
    """把 OAI-PMH ListRecords 的 OAI-DC XML 解析为论文列表。

    Args:
        xml_bytes: OAI-PMH 响应原始字节

    Returns:
        List[dict] — 统一 schema 的论文（title/authors/abstract/year/url/source）
    """
    root = ET.fromstring(xml_bytes)

    records: list[dict[str, Any]] = []
    # 遍历所有 <record>（无论外层是 <ListRecords> 还是 <GetRecord>）
    for record in root.iter():
        if record.tag.rsplit("}", 1)[-1] != "record":
            continue
        rec = _parse_record(record)
        if rec:
            records.append(rec)
    return records


def _parse_record(record: ET.Element) -> dict[str, Any] | None:
    """解析单个 <record>，提取 OAI-DC 元数据。

    只从 <metadata> 内提取 DC 字段，规避 <header> 里的 oai:identifier
    干扰 dc:identifier（真正的论文 URL）。
    """
    metadata = _first_child(record, "metadata")
    scope = metadata if metadata is not None else record

    # 用本地名匹配，规避命名空间前缀差异
    title = _first_text(scope, "title")
    if not title:
        return None

    creators = [el.text.strip() for el in _children(scope, "creator") if el.text]
    description = _first_text(scope, "description") or ""
    date = _first_text(scope, "date") or ""
    identifier = _first_text(scope, "identifier") or ""

    return {
        "title": title,
        "authors": creators[:5],
        "abstract": description[:2000],
        "year": (re.search(r"(\d{4})", date) or [None, ""])[1],
        "url": identifier,
        "source": "cnki",
        "doi": "",
        "pmid": "",
    }


def _first_child(root: ET.Element, local_name: str) -> ET.Element | None:
    """返回 root 下第一个匹配本地名的子元素，无则 None。"""
    for el in root.iter():
        if el.tag.rsplit("}", 1)[-1] == local_name:
            return el
    return None


def _children(root: ET.Element, local_name: str) -> list[ET.Element]:
    """按去除命名空间后的本地名，收集 root 下所有匹配子元素。"""
    out: list[ET.Element] = []
    for el in root.iter():
        if el.tag.rsplit("}", 1)[-1] == local_name:
            out.append(el)
    return out


def _first_text(root: ET.Element, local_name: str) -> str:
    """返回首个匹配子元素的文本（去空白），无则空串。"""
    for el in _children(root, local_name):
        if el.text and el.text.strip():
            return el.text.strip()
    return ""


def _filter_by_query(
    records: list[dict[str, Any]],
    query: str,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """客户端按 query 关键词对记录打分排序（OAI-PMH 无服务器端检索）。

    用「标题+摘要中命中查询词（中文逐字 / 英文逐词）的次数」作为相关度分，
    命中越多越靠前；无命中的记录被丢弃。
    """
    tokens = _query_tokens(query)
    if not tokens:
        return records[:max_results]

    scored: list[tuple[int, dict[str, Any]]] = []
    for rec in records:
        text = (rec.get("title", "") + " " + rec.get("abstract", "")).lower()
        score = sum(text.count(t) for t in tokens)
        if score > 0:
            scored.append((score, rec))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [rec for _, rec in scored[:max_results]]


def _query_tokens(query: str) -> list[str]:
    """把查询词拆成 token：含中文则逐字，否则按空白/标点分词。"""
    q = query.strip().lower()
    if not q:
        return []
    # 含中文字符 → 逐字切分（中文检索的最小粒度）
    if re.search(r"[一-鿿]", q):
        return list(re.findall(r"[一-鿿]", q))
    return [t for t in re.split(r"[^\w-]+", q) if t]


# ── Tool 实现 ──────────────────────────────────────────────────────────


class CNKISearchTool(BaseTool):
    """中文文献搜索工具（CNKI/万方 → ChinaXiv 兜底）。

    CNKI/万方无公开 API，实际通过 ChinaXiv OAI-PMH 收割中文预印本文献，
    再客户端按查询词过滤排序。
    """

    name: str = "cnki_search"
    description: str = (
        "Search Chinese biomedical literature (CNKI/Wanfang, via the ChinaXiv "
        "open repository). Input a Chinese or English biomedical query and get "
        "a list of Chinese-language papers with titles, authors, abstracts, and "
        "URLs. Best for: Chinese clinical research, traditional medicine, and "
        "Chinese-language biomedical preprints."
    )
    args_schema: type[BaseModel] = CNKISearchInput

    oai_url: str = CHINAXIV_OAI_URL
    set_spec: str = ""

    def _run(self, query: str, max_results: int = 10) -> dict:
        """执行中文文献搜索（收割 + 客户端过滤）。"""
        logger.info("CNKI/ChinaXiv search: query='%s', max=%d", query, max_results)
        try:
            records = self._fetch_records()
        except Exception as e:
            logger.warning("ChinaXiv harvest failed (degraded): %s", e)
            records = []

        papers = _filter_by_query(records, query, max_results)
        return {
            "papers": papers,
            "total_count": len(papers),
            "query": query,
            "source": "cnki",
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _fetch_records(self) -> list[dict[str, Any]]:
        """通过 OAI-PMH 收割 ChinaXiv 记录（ListRecords，OAI-DC）。"""
        params = {
            "verb": "ListRecords",
            "metadataPrefix": "oai_dc",
        }
        if self.set_spec:
            params["set"] = self.set_spec
        url = f"{self.oai_url}?{urllib.parse.urlencode(params)}"

        req = urllib.request.Request(url, headers={"User-Agent": "biomed-research-agent/2.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            xml_bytes = resp.read()

        records = _parse_oaidc_xml(xml_bytes)
        logger.info("ChinaXiv harvested %d records", len(records))
        return records


# ── 快速测试入口 ──────────────────────────────────────────────────────

if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    tool = CNKISearchTool()
    result = tool._run(query="细胞治疗 肿瘤", max_results=5)
    print(f"\nFound {result['total_count']} papers:\n")
    for i, p in enumerate(result["papers"], 1):
        print(f"{i}. {p['title']}")
        print(f"   {p['url']}")
        print()
