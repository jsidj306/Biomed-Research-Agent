"""
CNKI/万方 中文文献搜索工具 — 单元测试。

覆盖 src/tools/cnki_tool.py 的纯函数（OAI-DC 解析 / 查询过滤 / 分词）
与 CNKISearchTool 的降级行为。不发起真实网络请求。
"""

from unittest.mock import patch

from src.tools.cnki_tool import (
    CNKISearchTool,
    _filter_by_query,
    _parse_oaidc_xml,
    _query_tokens,
)

# ── 分词 ───────────────────────────────────────────────────────────────────


class TestQueryTokens:
    def test_chinese_query_splits_to_chars(self):
        tokens = _query_tokens("细胞治疗 肿瘤")
        assert tokens == ["细", "胞", "治", "疗", "肿", "瘤"]

    def test_english_query_splits_to_words(self):
        assert _query_tokens("CAR-T cell therapy") == ["car-t", "cell", "therapy"]

    def test_empty_query_returns_empty(self):
        assert _query_tokens("  ") == []


# ── 客户端过滤 ─────────────────────────────────────────────────────────────


def _rec(title: str, abstract: str = "") -> dict:
    return {
        "title": title,
        "authors": [],
        "abstract": abstract,
        "year": "2024",
        "url": "http://example.com",
        "source": "cnki",
        "doi": "",
        "pmid": "",
    }


class TestFilterByQuery:
    def test_filter_scores_and_orders(self):
        records = [
            _rec("肿瘤免疫治疗", "本文讨论 CAR-T 细胞治疗实体肿瘤"),
            _rec("细胞治疗", "细胞治疗在血液肿瘤中的进展"),
            _rec("无关标题", "无关内容"),
        ]
        result = _filter_by_query(records, "细胞治疗 肿瘤", max_results=10)
        # 命中者保留，未命中者丢弃；命中多的排前
        assert len(result) == 2
        assert result[0]["title"] == "肿瘤免疫治疗"

    def test_respects_max_results(self):
        records = [_rec("细胞治疗", "肿瘤") for _ in range(10)]
        result = _filter_by_query(records, "细胞治疗", max_results=3)
        assert len(result) == 3

    def test_empty_query_returns_top_n(self):
        records = [_rec(f"论文 {i}") for i in range(5)]
        assert len(_filter_by_query(records, "", max_results=3)) == 3


# ── OAI-DC 解析 ────────────────────────────────────────────────────────────

_OAIDC_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"
         xmlns:dc="http://purl.org/dc/elements/1.1/">
  <ListRecords>
    <record>
      <header>
        <identifier>oai:chinaxiv.org:202401.00123</identifier>
        <datestamp>2024-01-15</datestamp>
      </header>
      <metadata>
        <dc:dc xmlns:dc="http://purl.org/dc/elements/1.1/">
          <dc:title>细胞治疗在实体肿瘤中的临床应用</dc:title>
          <dc:creator>张三</dc:creator>
          <dc:creator>李四</dc:creator>
          <dc:description>本文综述了 CAR-T 细胞治疗的临床进展与挑战。</dc:description>
          <dc:date>2024-01-15</dc:date>
          <dc:identifier>http://www.chinaxiv.org/abs/202401.00123</dc:identifier>
        </dc:dc>
      </metadata>
    </record>
  </ListRecords>
</OAI-PMH>
"""


class TestParseOaiDc:
    def test_parse_extracts_fields(self):
        records = _parse_oaidc_xml(_OAIDC_SAMPLE.encode("utf-8"))
        assert len(records) == 1
        rec = records[0]
        assert rec["title"] == "细胞治疗在实体肿瘤中的临床应用"
        assert rec["authors"] == ["张三", "李四"]
        assert rec["year"] == "2024"
        assert rec["source"] == "cnki"
        # 关键：url 应为 dc:identifier（论文 URL），而非 header 里的 oai:identifier
        assert rec["url"] == "http://www.chinaxiv.org/abs/202401.00123"
        assert "CAR-T" in rec["abstract"]

    def test_parse_ignores_record_without_title(self):
        xml = b"""<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
          <ListRecords><record><metadata></metadata></record></ListRecords>
        </OAI-PMH>"""
        assert _parse_oaidc_xml(xml) == []


# ── Tool 降级行为 ──────────────────────────────────────────────────────────


class TestCNKISearchTool:
    def test_run_degrades_to_empty_on_harvest_failure(self):
        tool = CNKISearchTool()
        with patch.object(tool, "_fetch_records", side_effect=RuntimeError("boom")):
            result = tool._run(query="细胞治疗", max_results=5)
        assert result["papers"] == []
        assert result["total_count"] == 0
        assert result["source"] == "cnki"

    def test_run_filters_harvested_records(self):
        tool = CNKISearchTool()
        harvested = [
            _rec("细胞治疗与肿瘤免疫", "CAR-T 进展"),
            _rec("量子计算", "无关内容"),
        ]
        with patch.object(tool, "_fetch_records", return_value=harvested):
            result = tool._run(query="细胞治疗", max_results=10)
        assert result["total_count"] == 1
        assert result["papers"][0]["title"] == "细胞治疗与肿瘤免疫"

    def test_run_returns_unified_schema(self):
        tool = CNKISearchTool()
        with patch.object(tool, "_fetch_records", return_value=[_rec("标题", "摘要")]):
            result = tool._run(query="标题", max_results=10)
        paper = result["papers"][0]
        for key in ("title", "authors", "abstract", "year", "url", "source"):
            assert key in paper
