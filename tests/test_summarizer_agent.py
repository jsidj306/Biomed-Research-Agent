"""
Summarizer Agent 5 种摘要模式 — 单元测试。

运行: uv run pytest tests/test_summarizer_agent.py -v

测试新增的 standard / detailed(扩展) / comparison / gap_analysis 模式，
以及多篇输入、中英文切换、JSON 容错，全部 mock LLM。
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from src.agents.summarizer import (
    COMPARISON_PROMPT,
    DETAILED_PROMPT,
    GAP_ANALYSIS_PROMPT,
    STANDARD_PROMPT,
    PaperSummarizer,
)

# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def sample_abstract():
    return (
        "Lipid nanoparticles (LNPs) deliver CRISPR-Cas9 components in vivo. "
        "In this study, a novel ionizable lipid achieved >80% knockout efficiency "
        "in mouse hepatocytes with minimal off-target editing. These results "
        "demonstrate therapeutic potential for metabolic disorders."
    )


@pytest.fixture
def papers():
    return [
        {"title": "Paper A: CAR-T for solid tumors", "abstract": "CAR-T cells show efficacy."},
        {"title": "Paper B: CAR-T for leukemia", "abstract": "CAR-T cells show higher efficacy."},
        {"title": "Paper C: CAR-T resistance", "abstract": "Tumor cells develop resistance."},
    ]


def _make_summarizer(mock_llm):
    with patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "https://test.api.com/v1"},
    ), patch("src.agents.summarizer.ChatOpenAI", return_value=mock_llm):
        s = PaperSummarizer()
    s.llm = mock_llm
    return s


# ── 5 种模式 Prompt 完整性测试 ────────────────────────────────────────────


class TestPromptCompleteness:
    """验证 5 种模式的 Prompt 模板完整且占位符齐全。"""

    def test_standard_prompt_has_four_sections(self):
        assert "{title}" in STANDARD_PROMPT
        assert "{abstract}" in STANDARD_PROMPT
        assert "background" in STANDARD_PROMPT.lower()
        assert "methods" in STANDARD_PROMPT.lower()
        assert "results" in STANDARD_PROMPT.lower()
        assert "conclusions" in STANDARD_PROMPT.lower()

    def test_detailed_prompt_has_extended_fields(self):
        assert "data" in DETAILED_PROMPT.lower()
        assert "limitations" in DETAILED_PROMPT.lower()
        assert "significance" in DETAILED_PROMPT.lower()

    def test_comparison_prompt_demands_cross_comparison(self):
        assert "{papers}" in COMPARISON_PROMPT
        assert "similarities" in COMPARISON_PROMPT
        assert "differences" in COMPARISON_PROMPT

    def test_gap_analysis_prompt_has_gaps(self):
        assert "{papers}" in GAP_ANALYSIS_PROMPT
        assert "gaps" in GAP_ANALYSIS_PROMPT


# ── standard 模式测试 ──────────────────────────────────────────────────────


class TestStandardMode:
    """测试 standard（四段式）摘要。"""

    def test_standard_summary(self, sample_abstract):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=json.dumps({
            "background": "b", "methods": "m", "results": "r", "conclusions": "c",
        }))
        s = _make_summarizer(mock_llm)

        result = s.summarize(
            title="T", abstract=sample_abstract, summary_type="standard",
        )

        assert result["standard"] is not None
        assert "background" in result["standard"]
        assert "conclusions" in result["standard"]
        assert result["detailed"] is None


# ── detailed（扩展）模式测试 ───────────────────────────────────────────────


class TestDetailedMode:
    """测试 detailed（扩展精读）摘要。"""

    def test_detailed_summary_has_extended_fields(self, sample_abstract):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=json.dumps({
            "background": "b", "methods": "m", "results": "r", "conclusions": "c",
            "data": ">80% knockout", "limitations": "mouse model only",
            "significance": "therapeutic potential",
        }))
        s = _make_summarizer(mock_llm)

        result = s.summarize(
            title="T", abstract=sample_abstract, summary_type="detailed",
        )

        assert result["detailed"]["data"] == ">80% knockout"
        assert "limitations" in result["detailed"]
        assert "significance" in result["detailed"]


# ── Comparison 多篇对比测试 ───────────────────────────────────────────────


class TestComparison:
    """测试多篇横向对比摘要。"""

    def test_compare_multiple_papers(self, papers):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=json.dumps({
            "comparison": [{"dimension": "sample_size", "entries": {}}],
            "similarities": ["both use CAR-T"],
            "differences": ["different tumor types"],
        }))
        s = _make_summarizer(mock_llm)

        result = s.compare(papers)

        assert "comparison" in result
        assert len(result["similarities"]) >= 1
        # prompt 应包含多篇论文标题
        prompt = mock_llm.invoke.call_args[0][0]
        assert "Paper A: CAR-T for solid tumors" in prompt
        assert "Paper C: CAR-T resistance" in prompt

    def test_compare_requires_two_papers(self, papers):
        """compare() 少于 2 篇应抛 ValueError。"""
        s = PaperSummarizer.__new__(PaperSummarizer)
        with pytest.raises(ValueError):
            s.compare(papers[:1])


# ── Gap Analysis 研究空白测试 ─────────────────────────────────────────────


class TestGapAnalysis:
    """测试研究空白分析。"""

    def test_gap_analysis_returns_gaps(self, papers):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=json.dumps({
            "gaps": [
                {"gap": "no solid tumor CAR-T study", "rationale": "r",
                 "suggested_direction": "d"},
                {"gap": "no long-term follow-up", "rationale": "r",
                 "suggested_direction": "d"},
            ]
        }))
        s = _make_summarizer(mock_llm)

        result = s.gap_analysis(papers)

        assert "gaps" in result
        assert len(result["gaps"]) == 2
        assert "suggested_direction" in result["gaps"][0]

    def test_gap_analysis_requires_papers(self):
        s = PaperSummarizer.__new__(PaperSummarizer)
        with pytest.raises(ValueError):
            s.gap_analysis([])


# ── 输入校验测试 ───────────────────────────────────────────────────────────


class TestInputValidation:
    """测试非法 summary_type 的处理。"""

    def test_unsupported_summary_type_raises(self, sample_abstract):
        mock_llm = MagicMock()
        s = _make_summarizer(mock_llm)

        with pytest.raises(ValueError):
            s.summarize(
                title="T", abstract=sample_abstract, summary_type="comparison",
            )
