"""
Planner Agent — 单元测试。

运行: uv run pytest tests/test_planner_agent.py -v

验证研究方向推荐结构、假设因果结构、评分字段合法性和空输入降级，
通过 mock LLM 返回固定 JSON，不触发真实 API 调用。
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from src.agents.planner_agent import PLANNER_PROMPT, PlannerAgent

# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def gaps():
    return [
        {"gap": "CAR-T 在实体瘤中的持久性不足"},
        {"gap": "缺乏对肿瘤微环境长期影响的纵向研究"},
    ]


@pytest.fixture
def papers():
    return [{"title": "Paper A", "abstract": "CAR-T clinical trial"}]


def _make_planner(mock_llm):
    with patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "https://test.api.com/v1"},
    ), patch("src.agents.planner_agent.create_chat_model", return_value=mock_llm):
        planner = PlannerAgent()
    planner.llm = mock_llm
    return planner


def _mock_response(content):
    resp = MagicMock()
    resp.content = content
    return resp


# ── 方向推荐测试 ─────────────────────────────────────────────────────────


class TestGenerateDirections:
    """测试研究方向生成。"""

    def test_generate_directions_structure(self, gaps, papers):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_response(json.dumps({
            "directions": [
                {
                    "title": "增强 CAR-T 在实体瘤中的持久性",
                    "rationale": "持久性不足是核心瓶颈",
                    "hypothesis": "若共表达 IL-15，则 CAR-T 持久性提高，因 IL-15 促进记忆 T 细胞存活",
                    "feasibility": "high",
                    "novelty": "medium",
                }
            ]
        }))
        planner = _make_planner(mock_llm)

        directions = planner.generate_directions(gaps, papers)

        assert len(directions) == 1
        d = directions[0]
        assert d["title"]
        assert d["hypothesis"]
        assert d["feasibility"] in {"high", "medium", "low"}
        assert d["novelty"] in {"high", "medium", "low"}

    def test_hypothesis_causal_structure(self, gaps, papers):
        """假设应含因果结构（若…则…）。"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_response(json.dumps({
            "directions": [
                {"title": "T", "hypothesis": "若 X 则 Y，因 Z",
                 "feasibility": "high", "novelty": "high"}
            ]
        }))
        planner = _make_planner(mock_llm)

        directions = planner.generate_directions(gaps, papers)

        assert "若" in directions[0]["hypothesis"]

    def test_invalid_level_normalized_to_medium(self, gaps, papers):
        """非法 feasibility/novelty 取值应归一化为 medium。"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_response(json.dumps({
            "directions": [
                {"title": "T", "feasibility": "极高", "novelty": "unknown"}
            ]
        }))
        planner = _make_planner(mock_llm)

        directions = planner.generate_directions(gaps, papers)

        assert directions[0]["feasibility"] == "medium"
        assert directions[0]["novelty"] == "medium"

    def test_missing_title_dropped(self, gaps, papers):
        """缺 title 的方向应被丢弃。"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_response(json.dumps({
            "directions": [
                {"title": "", "hypothesis": "h"},
                {"title": "Valid", "hypothesis": "h2", "feasibility": "high", "novelty": "high"},
            ]
        }))
        planner = _make_planner(mock_llm)

        directions = planner.generate_directions(gaps, papers)

        assert len(directions) == 1
        assert directions[0]["title"] == "Valid"


# ── 降级测试 ─────────────────────────────────────────────────────────────


class TestDegradation:
    """测试空输入与 LLM 失败降级。"""

    def test_empty_gaps_returns_empty(self, papers):
        mock_llm = MagicMock()
        planner = _make_planner(mock_llm)

        assert planner.generate_directions([], papers) == []

    def test_llm_failure_returns_empty(self, gaps, papers):
        """LLM 抛异常时应返回空列表而非崩溃。"""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ValueError("LLM down")
        planner = _make_planner(mock_llm)

        assert planner.generate_directions(gaps, papers) == []


# ── Prompt 模板测试 ──────────────────────────────────────────────────────


class TestPrompt:
    def test_prompt_placeholders(self):
        assert "{gaps}" in PLANNER_PROMPT
        assert "{papers}" in PLANNER_PROMPT
        assert "directions" in PLANNER_PROMPT
        assert "hypothesis" in PLANNER_PROMPT
