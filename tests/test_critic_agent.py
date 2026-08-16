"""
Critic Agent — 单元测试。

运行: uv run pytest tests/test_critic_agent.py -v

通过 mock LLM 返回固定 JSON，验证交叉验证、引用核查、结果合并，
不触发真实 API 调用。
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from src.agents.critic_agent import CRITIC_PROMPT, CriticAgent

# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def sources():
    """4 篇可引用的论文（编号 1-4）。"""
    return [
        {"title": "Paper A", "abstract": "CAR-T for solid tumors"},
        {"title": "Paper B", "abstract": "mRNA vaccine"},
        {"title": "Paper C", "abstract": "CRISPR delivery"},
        {"title": "Paper D", "abstract": "immune checkpoint"},
    ]


def _make_critic(mock_llm):
    """构造注入 mock LLM 的 CriticAgent（两模型共用同一 mock）。"""
    with patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "https://test.api.com/v1"},
    ), patch("src.agents.critic_agent.create_chat_model", return_value=mock_llm):
        critic = CriticAgent()
    critic.llm_a = mock_llm
    critic.llm_b = mock_llm
    return critic


def _mock_response(content):
    resp = MagicMock()
    resp.content = content
    return resp


# ── 引用核查测试（确定性）────────────────────────────────────────────────


class TestCheckCitations:
    """测试确定性引用核查：不依赖 LLM，编造引用必被抓。"""

    def test_fabricated_citation_detected(self, sources):
        """报告引用 [5]，但 sources 只有 4 篇，应被识别为 high 严重度。"""
        issues = CriticAgent.check_citations("结论 [5] 表明……", sources)
        assert len(issues) == 1
        assert issues[0]["problem"] == "引用不存在"
        assert issues[0]["severity"] == "high"

    def test_valid_citations_pass(self, sources):
        """报告只引用 [1][2][3][4] 应无问题。"""
        issues = CriticAgent.check_citations("结论 [1] 与 [2] 一致，见 [3][4]", sources)
        assert issues == []

    def test_zero_citation_detected(self, sources):
        """引用 [0] 应被识别为无效。"""
        issues = CriticAgent.check_citations("[0] 是无意义引用", sources)
        assert len(issues) == 1

    def test_no_citations_no_issues(self, sources):
        """报告无任何引用标记应无问题。"""
        assert CriticAgent.check_citations("纯文本，无引用", sources) == []


# ── 结果合并测试 ─────────────────────────────────────────────────────────


class TestMerge:
    """测试两模型交叉验证结果的合并逻辑。"""

    def test_merge_union_issues(self):
        """两模型的 issues 应取并集（去重）。"""
        a = {"issues": [{"claim": "X", "problem": "无引用"}], "overall_score": 8}
        b = {"issues": [{"claim": "Y", "problem": "与原文矛盾"}], "overall_score": 9}
        merged = CriticAgent._merge(a, b)
        assert len(merged["issues"]) == 2

    def test_merge_dedup_same_issue(self):
        """两模型报同一问题应去重为 1 条。"""
        a = {"issues": [{"claim": "X", "problem": "无引用"}], "overall_score": 8}
        b = {"issues": [{"claim": "X", "problem": "无引用"}], "overall_score": 8}
        merged = CriticAgent._merge(a, b)
        assert len(merged["issues"]) == 1

    def test_merge_takes_stricter_score(self):
        """评分取更严格的一方（min）。"""
        merged = CriticAgent._merge({"overall_score": 8}, {"overall_score": 3})
        assert merged["overall_score"] == 3

    def test_merge_empty_verdict(self):
        """一个模型失败（空 dict）时，结果来自另一模型。"""
        merged = CriticAgent._merge({}, {"issues": [{"claim": "Z", "problem": "夸大"}], "overall_score": 5})
        assert merged["overall_score"] == 5
        assert len(merged["issues"]) == 1


# ── verify 全流程测试 ────────────────────────────────────────────────────


class TestVerify:
    """测试 verify() 全流程：引用核查 + 交叉验证 + 评分压制。"""

    def test_verify_merges_citation_and_llm_issues(self, sources):
        """LLM 报一条 issue + 报告里编造引用，最终 issues 应包含两者。"""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            _mock_response(json.dumps({"issues": [{"claim": "X", "problem": "无引用", "severity": "medium", "fix": "补引用"}], "overall_score": 7})),
            _mock_response(json.dumps({"issues": [], "overall_score": 8})),
        ]
        critic = _make_critic(mock_llm)

        result = critic.verify("结论 [9] 与 [1] 一致", sources)

        problems = {i["problem"] for i in result["issues"]}
        assert "引用不存在" in problems
        assert "无引用" in problems

    def test_fabricated_citation_caps_score(self, sources):
        """编造引用（high）应把评分压到不及格（< 6）。"""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            _mock_response(json.dumps({"issues": [], "overall_score": 10})),
            _mock_response(json.dumps({"issues": [], "overall_score": 10})),
        ]
        critic = _make_critic(mock_llm)

        result = critic.verify("完美报告但引用了 [99]", sources)

        assert result["overall_score"] <= 4.0

    def test_verify_llm_failure_degrades(self, sources):
        """两模型都失败时，verify 仍返回引用核查结果 + 中性评分。"""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ValueError("LLM down")
        critic = _make_critic(mock_llm)

        result = critic.verify("结论 [2] 有效", sources)

        assert result["overall_score"] >= 10.0  # 两模型失败不误判
        assert result["issues"] == []  # 无引用问题


# ── Prompt 模板测试 ──────────────────────────────────────────────────────


class TestPrompt:
    """验证 Critic Prompt 模板正确性。"""

    def test_prompt_placeholders(self):
        assert "{report}" in CRITIC_PROMPT
        assert "{sources}" in CRITIC_PROMPT
        assert "issues" in CRITIC_PROMPT
        assert "overall_score" in CRITIC_PROMPT
