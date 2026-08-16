"""
Knowledge Graph — 单元测试。

运行: uv run pytest tests/test_knowledge_graph.py -v

验证三元组抽取（LLM + 词典兜底）、Cypher 生成、Mermaid 导出和空输入降级。
mock LLM 与 mock Neo4j driver，不触发真实 API / 数据库调用。
"""

import json
import os
from unittest.mock import MagicMock, patch

from src.knowledge_graph.extractor import (
    KnowledgeGraphExtractor,
    to_mermaid,
)
from src.knowledge_graph.neo4j_writer import Neo4jWriter

# ── Fixtures ───────────────────────────────────────────────────────────────


def _make_extractor(mock_llm):
    with patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "https://test.api.com/v1"},
    ), patch("src.knowledge_graph.extractor.create_chat_model", return_value=mock_llm):
        extractor = KnowledgeGraphExtractor()
    extractor.llm = mock_llm
    return extractor


def _mock_response(content):
    resp = MagicMock()
    resp.content = content
    return resp


# ── 词典兜底抽取测试（确定性）────────────────────────────────────────────


class TestDictionaryExtraction:
    """测试词典规则匹配兜底。"""

    def test_drug_treats_disease(self):
        triples = KnowledgeGraphExtractor._extract_dictionary(
            "CAR-T therapy is used to treat leukemia"
        )
        keys = {(t["head_name"], t["relation"], t["tail_name"]) for t in triples}
        assert ("car-t", "TREATS", "leukemia") in keys

    def test_drug_targets_target(self):
        triples = KnowledgeGraphExtractor._extract_dictionary(
            "Rituximab targets CD19 on B cells"
        )
        keys = {(t["head_name"], t["relation"], t["tail_name"]) for t in triples}
        assert ("rituximab", "TARGETS", "cd19") in keys

    def test_no_entities_no_triples(self):
        assert KnowledgeGraphExtractor._extract_dictionary("a completely unrelated sentence") == []


# ── LLM 抽取测试 ─────────────────────────────────────────────────────────


class TestLLMExtraction:
    """测试 LLM 主抽取 + 归一化。"""

    def test_llm_extraction(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_response(json.dumps({
            "triples": [
                {"head": "drug", "head_name": "CAR-T", "relation": "TREATS",
                 "tail": "disease", "tail_name": "cancer"}
            ]
        }))
        extractor = _make_extractor(mock_llm)

        triples = extractor.extract_triples("CAR-T therapy treats cancer")

        assert len(triples) >= 1
        assert triples[0]["head_type"] == "drug"
        assert triples[0]["relation"] == "TREATS"

    def test_llm_failure_falls_back_to_dictionary(self):
        """LLM 失败时应回退到词典兜底，仍能抽出三元组。"""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ValueError("LLM down")
        extractor = _make_extractor(mock_llm)

        triples = extractor.extract_triples("CAR-T therapy treats leukemia")

        keys = {(t["head_name"], t["relation"], t["tail_name"]) for t in triples}
        assert ("car-t", "TREATS", "leukemia") in keys

    def test_invalid_triple_dropped(self):
        """非法实体类型的三元组应被丢弃。"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_response(json.dumps({
            "triples": [
                {"head": "planet", "head_name": "X", "relation": "TREATS",
                 "tail": "disease", "tail_name": "Y"}
            ]
        }))
        extractor = _make_extractor(mock_llm)

        triples = extractor.extract_triples("irrelevant text with no dictionary terms")

        assert triples == []


# ── 空输入降级测试 ───────────────────────────────────────────────────────


class TestEmptyInput:
    def test_short_text_returns_empty(self):
        mock_llm = MagicMock()
        extractor = _make_extractor(mock_llm)

        assert extractor.extract_triples("short") == []


# ── Cypher 生成测试 ──────────────────────────────────────────────────────


class TestBuildCypher:
    """测试 MERGE Cypher 语句生成。"""

    def test_cypher_merge_statement(self):
        triples = [{
            "head_type": "drug", "head_name": "CAR-T",
            "relation": "TREATS", "tail_type": "disease", "tail_name": "cancer",
        }]
        statements = Neo4jWriter.build_cypher(triples)
        assert len(statements) == 1
        assert "MERGE" in statements[0]
        assert ":Drug" in statements[0]
        assert ":Disease" in statements[0]
        assert ":TREATS" in statements[0]

    def test_empty_triples(self):
        assert Neo4jWriter.build_cypher([]) == []


# ── 写入测试（mock driver）──────────────────────────────────────────────


class TestWriteTriples:
    """测试写入 Neo4j（mock driver，不真实连接）。"""

    def test_write_triples_uses_driver(self):
        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        writer = Neo4jWriter(driver=mock_driver)
        triples = [{
            "head_type": "drug", "head_name": "CAR-T",
            "relation": "TREATS", "tail_type": "disease", "tail_name": "cancer",
        }]
        count = writer.write_triples(triples)

        assert count == 1
        assert mock_session.run.called

    def test_write_empty_returns_zero(self):
        writer = Neo4jWriter(driver=MagicMock())
        assert writer.write_triples([]) == 0


# ── Mermaid 可视化测试 ───────────────────────────────────────────────────


class TestToMermaid:
    def test_mermaid_contains_triples(self):
        triples = [{
            "head_type": "drug", "head_name": "CAR-T",
            "relation": "TREATS", "tail_type": "disease", "tail_name": "cancer",
        }]
        mermaid = to_mermaid(triples)
        assert "CAR-T" in mermaid
        assert "cancer" in mermaid
        assert "TREATS" in mermaid
        assert mermaid.startswith("graph TD")
