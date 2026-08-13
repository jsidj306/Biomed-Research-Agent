"""
Paper Summarizer — 单元测试。

运行: uv run pytest tests/test_summarizer.py -v
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from src.agents.summarizer import (
    BRIEF_PROMPT,
    DETAILED_PROMPT,
    KEY_FINDINGS_PROMPT,
    PaperSummarizer,
)


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def sample_title():
    return "CRISPR-Cas9 delivery via lipid nanoparticles for in vivo genome editing"


@pytest.fixture
def sample_abstract():
    return (
        "Lipid nanoparticles (LNPs) have emerged as a promising delivery "
        "vehicle for CRISPR-Cas9 components. In this study, we developed "
        "a novel ionizable lipid formulation that achieves >80% knockout "
        "efficiency in hepatocytes following systemic administration in mice. "
        "The optimized LNPs showed minimal off-target editing as assessed by "
        "whole-genome sequencing, and no significant liver toxicity was "
        "observed. Histological analysis confirmed successful editing of the "
        "PCSK9 gene with concomitant reduction in serum cholesterol levels. "
        "These results demonstrate the therapeutic potential of LNP-mediated "
        "CRISPR delivery for metabolic disorders."
    )


@pytest.fixture
def mock_llm_response_brief():
    """模拟 LLM 的简短摘要 JSON 响应。"""
    return json.dumps({
        "one_liner": "A novel ionizable lipid nanoparticle achieves >80% CRISPR knockout efficiency in mouse hepatocytes with minimal off-target effects.",
        "takeaway": "LNP-mediated CRISPR delivery shows therapeutic potential for metabolic disorders like hypercholesterolemia.",
    })


@pytest.fixture
def mock_llm_response_detailed():
    """模拟 LLM 的详细摘要 JSON 响应。"""
    return json.dumps({
        "background": "CRISPR-Cas9 gene editing holds immense therapeutic potential, but safe and efficient in vivo delivery remains a major bottleneck.",
        "methods": "The authors developed a novel ionizable lipid nanoparticle (LNP) formulation and tested it for systemic delivery of CRISPR-Cas9 components targeting the PCSK9 gene in mice.",
        "results": "The optimized LNPs achieved >80% knockout efficiency in hepatocytes with minimal off-target editing and no significant liver toxicity. PCSK9 editing led to reduced serum cholesterol levels.",
        "conclusions": "LNP-mediated CRISPR delivery is a viable strategy for treating metabolic disorders and warrants further preclinical development.",
    })


@pytest.fixture
def mock_llm_response_key_findings():
    """模拟 LLM 的关键发现 JSON 响应。"""
    return json.dumps({
        "findings": [
            "Novel ionizable lipid achieves >80% CRISPR knockout in vivo",
            "Minimal off-target editing confirmed by whole-genome sequencing",
            "No significant liver toxicity observed",
            "PCSK9 editing reduces serum cholesterol in mice",
            "LNP formulation enables systemic administration of CRISPR components",
        ],
        "significance": "This study provides strong preclinical evidence for LNP-mediated CRISPR delivery as a therapeutic approach for metabolic diseases.",
    })


# ── 初始化测试 ──────────────────────────────────────────────────────────


class TestPaperSummarizerInit:
    """测试 PaperSummarizer 初始化。"""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "https://test.api.com/v1"})
    def test_default_init(self):
        """默认参数初始化。"""
        s = PaperSummarizer()
        assert s.model_name == "gpt-4o-mini"
        assert s.temperature == 0.3
        assert s.max_tokens == 1024

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "https://test.api.com/v1"})
    def test_custom_init(self):
        """自定义参数初始化。"""
        s = PaperSummarizer(model_name="gpt-4o", temperature=0.1, max_tokens=512)
        assert s.model_name == "gpt-4o"
        assert s.temperature == 0.1
        assert s.max_tokens == 512

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "https://oa.api2d.net"})
    def test_api2d_config(self):
        """API2D 中转配置正确。"""
        s = PaperSummarizer()
        assert s.llm.openai_api_base == "https://oa.api2d.net"
        assert s.llm.openai_api_key.get_secret_value() == "test-key"


# ── 简短摘要测试 ────────────────────────────────────────────────────────


class TestBriefSummary:
    """测试 brief 摘要类型。"""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "https://test.api.com/v1"})
    @patch("src.agents.summarizer.ChatOpenAI")
    def test_brief_summary(self, MockChatOpenAI, sample_title, sample_abstract, mock_llm_response_brief):
        """简短摘要生成：返回 one_liner + takeaway。"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = mock_llm_response_brief
        mock_llm.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm

        s = PaperSummarizer()
        s.llm = mock_llm  # 注入 mock

        result = s.summarize(
            title=sample_title,
            abstract=sample_abstract,
            summary_type="brief",
        )

        assert result["paper_title"] == sample_title
        assert result["summary_type"] == "brief"
        assert result["brief"] is not None
        assert "one_liner" in result["brief"]
        assert "takeaway" in result["brief"]
        assert result["detailed"] is None
        assert result["key_findings"] is None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "https://test.api.com/v1"})
    @patch("src.agents.summarizer.ChatOpenAI")
    def test_brief_prompt_format(self, MockChatOpenAI, sample_title, sample_abstract, mock_llm_response_brief):
        """验证 prompt 模板正确格式化了标题和摘要。"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = mock_llm_response_brief
        mock_llm.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm

        s = PaperSummarizer()
        s.llm = mock_llm

        s.summarize(title=sample_title, abstract=sample_abstract, summary_type="brief")

        call_args = mock_llm.invoke.call_args[0][0]
        assert sample_title in call_args
        assert "CRISPR-Cas9" in call_args
        assert "Lipid nanoparticles" in call_args


# ── 详细摘要测试 ────────────────────────────────────────────────────────


class TestDetailedSummary:
    """测试 detailed 摘要类型。"""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "https://test.api.com/v1"})
    @patch("src.agents.summarizer.ChatOpenAI")
    def test_detailed_summary(self, MockChatOpenAI, sample_title, sample_abstract, mock_llm_response_detailed):
        """详细摘要生成：返回 background/methods/results/conclusions。"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = mock_llm_response_detailed
        mock_llm.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm

        s = PaperSummarizer()
        s.llm = mock_llm

        result = s.summarize(
            title=sample_title,
            abstract=sample_abstract,
            summary_type="detailed",
            language="english",
        )

        assert result["paper_title"] == sample_title
        assert result["brief"] is None
        assert result["detailed"] is not None
        assert "background" in result["detailed"]
        assert "methods" in result["detailed"]
        assert "results" in result["detailed"]
        assert "conclusions" in result["detailed"]
        assert result["key_findings"] is None


# ── 关键发现测试 ────────────────────────────────────────────────────────


class TestKeyFindings:
    """测试 key_findings 摘要类型。"""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "https://test.api.com/v1"})
    @patch("src.agents.summarizer.ChatOpenAI")
    def test_key_findings(self, MockChatOpenAI, sample_title, sample_abstract, mock_llm_response_key_findings):
        """关键发现摘要：返回 findings 列表 + significance。"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = mock_llm_response_key_findings
        mock_llm.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm

        s = PaperSummarizer()
        s.llm = mock_llm

        result = s.summarize(
            title=sample_title,
            abstract=sample_abstract,
            summary_type="key_findings",
        )

        assert result["key_findings"] is not None
        assert "findings" in result["key_findings"]
        assert len(result["key_findings"]["findings"]) >= 3
        assert "significance" in result["key_findings"]
        assert result["brief"] is None
        assert result["detailed"] is None


# ── All 类型测试 ────────────────────────────────────────────────────────


class TestAllSummary:
    """测试 'all' 摘要类型 — 同时生成三种摘要。"""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "https://test.api.com/v1"})
    @patch("src.agents.summarizer.ChatOpenAI")
    def test_all_summary_types(
        self, MockChatOpenAI, sample_title, sample_abstract,
        mock_llm_response_brief, mock_llm_response_detailed, mock_llm_response_key_findings,
    ):
        """summary_type='all' 应同时填充 brief/detailed/key_findings。"""
        mock_llm = MagicMock()
        mock_response = MagicMock()

        # 按调用顺序返回不同内容
        mock_response.content = mock_llm_response_brief
        mock_llm.invoke = MagicMock(side_effect=[
            MagicMock(content=mock_llm_response_brief),
            MagicMock(content=mock_llm_response_detailed),
            MagicMock(content=mock_llm_response_key_findings),
        ])
        MockChatOpenAI.return_value = mock_llm

        s = PaperSummarizer()
        s.llm = mock_llm

        result = s.summarize(
            title=sample_title,
            abstract=sample_abstract,
            summary_type="all",
        )

        assert result["brief"] is not None
        assert result["detailed"] is not None
        assert result["key_findings"] is not None
        assert mock_llm.invoke.call_count == 3


# ── 边界条件测试 ────────────────────────────────────────────────────────


class TestEdgeCases:
    """测试边界条件。"""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "https://test.api.com/v1"})
    @patch("src.agents.summarizer.ChatOpenAI")
    def test_short_abstract(self, MockChatOpenAI):
        """短摘要 (< 50 字) 应返回警告。"""
        s = PaperSummarizer()
        result = s.summarize(
            title="A short paper",
            abstract="Too short.",
            summary_type="brief",
        )
        assert "warning" in result
        assert "too short" in result["warning"].lower()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "https://test.api.com/v1"})
    @patch("src.agents.summarizer.ChatOpenAI")
    def test_chinese_language(self, MockChatOpenAI, sample_title, sample_abstract, mock_llm_response_brief):
        """中文输出 — prompt 中 language 参数应为 'chinese'。"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = mock_llm_response_brief
        mock_llm.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm

        s = PaperSummarizer()
        s.llm = mock_llm

        s.summarize(
            title=sample_title, abstract=sample_abstract,
            summary_type="brief", language="chinese",
        )

        call_args = mock_llm.invoke.call_args[0][0]
        assert "chinese" in call_args.lower()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "https://test.api.com/v1"})
    @patch("src.agents.summarizer.ChatOpenAI")
    def test_long_abstract_truncation(self, MockChatOpenAI, sample_title):
        """超长摘要 (> 3000 字) 应被截断。"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"one_liner": "test", "takeaway": "test"}'
        mock_llm.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm

        s = PaperSummarizer()
        s.llm = mock_llm

        long_abstract = "This is a biomedical research paper. " * 1000  # ~7000 chars

        s.summarize(title=sample_title, abstract=long_abstract, summary_type="brief")

        call_args = mock_llm.invoke.call_args[0][0]
        # 传给 LLM 的 abstract 不应超过 3000 chars
        assert len(call_args) < len(long_abstract) + 500  # prompt overhead

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "https://test.api.com/v1"})
    @patch("src.agents.summarizer.ChatOpenAI")
    def test_batch_summarize(self, MockChatOpenAI, mock_llm_response_brief):
        """批量摘要：输入 3 篇，输出 3 个结果。"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = mock_llm_response_brief
        mock_llm.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm

        s = PaperSummarizer()
        s.llm = mock_llm

        papers = [
            {"title": f"Paper {i}", "abstract": f"Abstract content for paper {i}. " * 5}
            for i in range(3)
        ]

        results = s.summarize_batch(papers, summary_type="brief")
        assert len(results) == 3
        assert mock_llm.invoke.call_count == 3


# ── JSON 解析测试 ────────────────────────────────────────────────────────


class TestJSONParsing:
    """测试 LLM 响应 JSON 解析的鲁棒性。"""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "https://test.api.com/v1"})
    @patch("src.agents.summarizer.ChatOpenAI")
    def test_json_with_markdown_fence(self, MockChatOpenAI, sample_title, sample_abstract):
        """LLM 返回带 ```json fences 的响应应能正确解析。"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '```json\n{"one_liner": "test.", "takeaway": "test."}\n```'
        mock_llm.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm

        s = PaperSummarizer()
        s.llm = mock_llm

        result = s.summarize(
            title=sample_title, abstract=sample_abstract, summary_type="brief",
        )
        assert result["brief"]["one_liner"] == "test."

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "https://test.api.com/v1"})
    @patch("src.agents.summarizer.ChatOpenAI")
    def test_json_with_extra_text(self, MockChatOpenAI, sample_title, sample_abstract):
        """LLM 返回 JSON 前后带额外文字时应能提取。"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = 'Here is the summary:\n{"one_liner": "result.", "takeaway": "implication."}\nHope this helps!'
        mock_llm.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm

        s = PaperSummarizer()
        s.llm = mock_llm

        result = s.summarize(
            title=sample_title, abstract=sample_abstract, summary_type="brief",
        )
        assert result["brief"]["one_liner"] == "result."


# ── Prompt 模板测试 ─────────────────────────────────────────────────────


class TestPromptTemplates:
    """验证 Prompt 模板正确性。"""

    def test_brief_prompt_contains_placeholders(self):
        assert "{title}" in BRIEF_PROMPT
        assert "{abstract}" in BRIEF_PROMPT
        assert "{language}" in BRIEF_PROMPT

    def test_detailed_prompt_contains_placeholders(self):
        assert "{title}" in DETAILED_PROMPT
        assert "{abstract}" in DETAILED_PROMPT
        assert "background" in DETAILED_PROMPT.lower()
        assert "methods" in DETAILED_PROMPT.lower()
        assert "results" in DETAILED_PROMPT.lower()
        assert "conclusions" in DETAILED_PROMPT.lower()

    def test_key_findings_prompt_contains_placeholders(self):
        assert "{title}" in KEY_FINDINGS_PROMPT
        assert "{abstract}" in KEY_FINDINGS_PROMPT
        assert "findings" in KEY_FINDINGS_PROMPT.lower()
        assert "significance" in KEY_FINDINGS_PROMPT.lower()
