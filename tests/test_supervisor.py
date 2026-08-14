"""
Supervisor Agent — 单元测试。

运行: uv run pytest tests/test_supervisor.py -v

通过 mock LLM 返回固定 JSON，验证意图解析、计划生成和 JSON 容错，
不触发真实 API 调用。
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from src.agents.supervisor import (
    INTENT_PROMPT,
    PLAN_PROMPT,
    SupervisorAgent,
)

# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def search_query():
    return "Find the latest clinical trials on CAR-T therapy for solid tumors"


@pytest.fixture
def summarize_query():
    return "Summarize this abstract about CRISPR gene editing"


def _make_supervisor(mock_llm):
    """构造注入 mock LLM 的 SupervisorAgent。"""
    with patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "https://test.api.com/v1"},
    ), patch("src.agents.supervisor.create_chat_model", return_value=mock_llm):
        sup = SupervisorAgent()
    sup.llm = mock_llm
    return sup


def _mock_response(content):
    resp = MagicMock()
    resp.content = content
    return resp


# ── 意图解析测试 ───────────────────────────────────────────────────────────


class TestClassifyIntent:
    """测试意图解析（检索类 vs 总结类）。"""

    def test_search_intent(self, search_query):
        """检索类问题应返回 'search'。"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_response(
            json.dumps({"intent": "search", "reason": "needs literature lookup"})
        )
        sup = _make_supervisor(mock_llm)

        assert sup.classify_intent(search_query) == "search"

    def test_summarize_intent(self, summarize_query):
        """总结类问题应返回 'summarize'。"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_response(
            json.dumps({"intent": "summarize", "reason": "content already provided"})
        )
        sup = _make_supervisor(mock_llm)

        assert sup.classify_intent(summarize_query) == "summarize"

    def test_unknown_intent_falls_back_to_search(self, search_query):
        """LLM 返回未知 intent 时应保守回退到 'search'。"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_response(
            json.dumps({"intent": "gibberish", "reason": "?"})
        )
        sup = _make_supervisor(mock_llm)

        assert sup.classify_intent(search_query) == "search"


# ── 计划生成测试 ───────────────────────────────────────────────────────────


class TestGeneratePlan:
    """测试研究步骤计划生成。"""

    def test_generate_plan_returns_list(self, search_query):
        """plan 应为 list[str]。"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_response(
            json.dumps({
                "plan": [
                    "Search PubMed for CAR-T solid tumor clinical trials",
                    "Review clinical outcome data",
                    "Identify adverse events",
                ]
            })
        )
        sup = _make_supervisor(mock_llm)

        plan = sup.generate_plan(search_query)

        assert isinstance(plan, list)
        assert len(plan) == 3
        assert all(isinstance(s, str) and s for s in plan)

    def test_plan_prompt_contains_query(self, search_query):
        """prompt 应包含用户问题。"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_response(json.dumps({"plan": ["s1"]}))
        sup = _make_supervisor(mock_llm)

        sup.generate_plan(search_query)

        call_args = mock_llm.invoke.call_args[0][0]
        assert search_query in call_args

    def test_plan_json_with_markdown_fence(self, search_query):
        """带 markdown 围栏的 JSON 应能容错解析。"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_response(
            '```json\n{"plan": ["step A", "step B"]}\n```'
        )
        sup = _make_supervisor(mock_llm)

        plan = sup.generate_plan(search_query)

        assert plan == ["step A", "step B"]

    def test_plan_missing_key_returns_empty(self, search_query):
        """LLM 漏掉 plan 键时应返回空列表而非抛异常。"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_response(json.dumps({"steps": ["x"]}))
        sup = _make_supervisor(mock_llm)

        assert sup.generate_plan(search_query) == []

    def test_plan_string_wrapped(self, search_query):
        """plan 是单个字符串时应包装成 list。"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_response(
            json.dumps({"plan": "search PubMed for CAR-T"})
        )
        sup = _make_supervisor(mock_llm)

        assert sup.generate_plan(search_query) == ["search PubMed for CAR-T"]


# ── JSON 容错与错误处理测试 ────────────────────────────────────────────────


class TestErrorHandling:
    """测试 LLM 异常输出的降级处理。"""

    def test_unparseable_response_raises(self):
        """完全无法解析的响应应抛出 ValueError。"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_response("not json at all")
        sup = _make_supervisor(mock_llm)

        with pytest.raises(ValueError):
            sup._call_llm.__wrapped__(sup, "prompt")


# ── Prompt 模板测试 ────────────────────────────────────────────────────────


class TestPromptTemplates:
    """验证 Prompt 模板正确性。"""

    def test_intent_prompt_placeholders(self):
        assert "{query}" in INTENT_PROMPT
        assert "search" in INTENT_PROMPT
        assert "summarize" in INTENT_PROMPT

    def test_plan_prompt_placeholders(self):
        assert "{query}" in PLAN_PROMPT
        assert "plan" in PLAN_PROMPT
