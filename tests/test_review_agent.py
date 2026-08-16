"""
Review Agent — 单元测试。

运行: uv run pytest tests/test_review_agent.py -v

通过 mock LLM 验证评分解析、三级去重、矛盾检测和 Top-K 排序，
不触发真实 API 调用。
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from src.agents.review_agent import ReviewAgent

# ── Fixtures ───────────────────────────────────────────────────────────────


def _make_agent(mock_llm):
    with patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "https://test.api.com/v1"},
    ), patch("src.agents.review_agent.create_chat_model", return_value=mock_llm):
        agent = ReviewAgent()
    agent.llm = mock_llm
    return agent


def _paper(title, doi="", pmid="", abstract="abstract text here", source="pubmed"):
    return {
        "title": title,
        "doi": doi,
        "pmid": pmid,
        "abstract": abstract,
        "source": source,
    }


@pytest.fixture
def papers():
    return [
        _paper("CAR-T therapy for solid tumors", doi="10.1/a"),
        _paper("mRNA vaccine for influenza", doi="10.1/b"),
        _paper("Deep learning for drug design", doi="10.1/c"),
    ]


# ── 评分测试 ───────────────────────────────────────────────────────────────


class TestScorePapers:
    """测试 LLM 相关性评分。"""

    def test_score_papers_returns_score_and_reason(self, papers):
        """每篇论文应返回 score + reason。"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps({"score": 8, "reason": "highly relevant"})
        )
        agent = _make_agent(mock_llm)

        scored = agent.score_papers(papers, "CAR-T")

        assert len(scored) == 3
        for s in scored:
            assert s["score"] == 8
            assert s["reason"] == "highly relevant"

    def test_score_clamped_to_range(self):
        """score 超出 0-10 时应被夹紧。"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps({"score": 99, "reason": "x"})
        )
        agent = _make_agent(mock_llm)

        scored = agent.score_papers([_paper("T")], "query")
        assert scored[0]["score"] == 10


# ── 三级去重测试 ───────────────────────────────────────────────────────────


class TestDeduplicate:
    """测试三级去重。"""

    def test_hard_dedup_by_doi(self):
        """同 DOI（硬去重）。"""
        papers = [_paper("A", doi="10.1/x"), _paper("A2", doi="10.1/X")]
        agent = ReviewAgent.__new__(ReviewAgent)  # 绕过 __init__，无需 LLM
        assert len(agent.deduplicate_papers(papers)) == 1

    def test_cross_source_dedup_by_title(self):
        """跨源重复（标题一致但 ID 不同）。"""
        papers = [
            _paper("CAR-T for solid tumors", doi="", pmid="", source="pubmed"),
            _paper("CAR-T for solid tumors", doi="", pmid="", source="arxiv"),
        ]
        agent = ReviewAgent.__new__(ReviewAgent)
        assert len(agent.deduplicate_papers(papers)) == 1

    def test_soft_dedup_by_title_similarity(self):
        """软去重：标题相似度 > 0.9 判重（如版本号后缀差异）。"""
        papers = [
            _paper("CAR-T therapy for solid tumors", doi="", pmid=""),
            _paper("CAR-T therapy for solid tumors v2", doi="", pmid=""),
        ]
        agent = ReviewAgent.__new__(ReviewAgent)
        assert len(agent.deduplicate_papers(papers)) == 1

    def test_distinct_papers_not_deduped(self):
        """标题明显不同的论文不应被误删。"""
        papers = [
            _paper("CAR-T therapy for solid tumors", doi="", pmid=""),
            _paper("mRNA vaccine for influenza", doi="", pmid=""),
        ]
        agent = ReviewAgent.__new__(ReviewAgent)
        assert len(agent.deduplicate_papers(papers)) == 2


# ── 矛盾检测测试 ───────────────────────────────────────────────────────────


class TestConflictDetection:
    """测试结论矛盾检测。"""

    def test_detect_conflicts(self):
        """LLM 返回的冲突对应被解析。"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps({
                "conflicts": [
                    {
                        "paper_a": "Drug X is effective",
                        "paper_b": "Drug X is ineffective",
                        "description": "opposite conclusions",
                    }
                ]
            })
        )
        agent = _make_agent(mock_llm)

        papers = [
            _paper("Drug X is effective"),
            _paper("Drug X is ineffective"),
        ]
        conflicts = agent.detect_conflicts(papers)

        assert len(conflicts) == 1
        assert conflicts[0]["paper_a"] == "Drug X is effective"

    def test_conflict_flags_applied_in_review(self):
        """review() 应将冲突论文标记 conflict=True。"""
        title_a = "CAR-T cell therapy for solid tumors"
        title_b = "CAR-T cell therapy for hematological malignancies"
        mock_llm = MagicMock()
        # score_papers 与 detect_conflicts 都会调用 LLM，用 side_effect 区分
        mock_llm.invoke.side_effect = [
            MagicMock(content=json.dumps({"score": 8, "reason": "r"})),  # paper A
            MagicMock(content=json.dumps({"score": 6, "reason": "r"})),  # paper B
            MagicMock(content=json.dumps({
                "conflicts": [
                    {"paper_a": title_a, "paper_b": title_b,
                     "description": "opposite conclusions"}
                ]
            })),
        ]
        agent = _make_agent(mock_llm)

        papers = [_paper(title_a), _paper(title_b)]
        reviews = agent.review(papers, "query", top_k=10)

        conflict_flags = {r["title"]: r.get("conflict") for r in reviews}
        assert conflict_flags[title_a] is True
        assert conflict_flags[title_b] is True


# ── Top-K 排序测试 ─────────────────────────────────────────────────────────


class TestTopK:
    """测试按 score 降序输出 Top-K。"""

    def test_review_sorts_descending(self):
        """结果应按 score 降序排列。"""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            MagicMock(content=json.dumps({"score": 3, "reason": "low"})),
            MagicMock(content=json.dumps({"score": 9, "reason": "high"})),
            MagicMock(content=json.dumps({"score": 6, "reason": "mid"})),
            MagicMock(content=json.dumps({"conflicts": []})),
        ]
        agent = _make_agent(mock_llm)

        papers = [_paper("A"), _paper("B"), _paper("C")]
        reviews = agent.review(papers, "query", top_k=10)

        scores = [r["score"] for r in reviews]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == 9

    def test_review_limits_top_k(self):
        """review 应截断到 top_k 篇。"""
        mock_llm = MagicMock()
        responses = [MagicMock(content=json.dumps({"score": i, "reason": "r"})) for i in range(5)]
        responses.append(MagicMock(content=json.dumps({"conflicts": []})))
        mock_llm.invoke.side_effect = responses
        agent = _make_agent(mock_llm)

        papers = [_paper(f"P{i}") for i in range(5)]
        reviews = agent.review(papers, "query", top_k=2)

        assert len(reviews) == 2

    def test_review_empty_papers(self):
        """空输入应返回空列表。"""
        agent = ReviewAgent.__new__(ReviewAgent)
        assert agent.review([], "query") == []
