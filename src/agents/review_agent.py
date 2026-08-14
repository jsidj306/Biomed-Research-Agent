"""
Review Agent — 相关性评分 + 去重 + 排序 + 矛盾检测。

从 Search Agent 产出的几十篇候选论文中，挑出最相关的 Top-K，减少下游
Summarize Agent 的 token 消耗和用户的信息过载。核心能力:
  1. 相关性评分（LLM 0-10 打分，附理由）
  2. 三级去重（硬去重 DOI/PMID + 软去重标题相似度 + 跨源去重）
  3. 矛盾检测（识别「结论相悖」的论文对）
  4. Top-K 排序输出

使用示例:
    from src.agents.review_agent import ReviewAgent

    agent = ReviewAgent()
    reviews = agent.review(papers, query="CAR-T solid tumor", top_k=10)
    for r in reviews:
        print(r["score"], r["title"], r["reason"])
"""

from __future__ import annotations

import difflib
import logging
from typing import Any

from dotenv import load_dotenv
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from src.agents.search_agent import deduplicate_papers as _hard_dedup
from src.utils.llm_utils import create_chat_model, parse_json_response

load_dotenv()

logger = logging.getLogger(__name__)

# ── Prompt 模板 ───────────────────────────────────────────────────────────

SCORE_PROMPT = """你是资深生物医学研究员。给这篇论文与用户问题的相关性打分（0-10 整数）。

评分依据（综合考量）:
- 与问题的语义相关度（最重要）
- 方法学质量
- 发表时间（越新越好）
- 来源可信度（正式发表 > 预印本）

用户问题: {query}
论文标题: {title}
论文摘要: {abstract}
论文来源: {source}

返回 JSON 对象:
{{
  "score": 8,
  "reason": "一句话说明打分理由"
}}

只返回合法 JSON，不要 markdown 围栏或多余文字。"""

CONFLICT_PROMPT = """你是资深生物医学研究员。分析以下论文中是否存在「结论相互矛盾」的论文对。

例如：A 论文声称某药物有效，B 论文声称同一药物无效——这就是结论相悖。

论文列表（标题 | 摘要）:
{paper_list}

返回 JSON 对象:
{{
  "conflicts": [
    {{"paper_a": "论文A标题", "paper_b": "论文B标题", "description": "矛盾点描述"}}
  ]
}}

若没有发现矛盾，返回 {{"conflicts": []}}。
只返回合法 JSON，不要 markdown 围栏或多余文字。"""


# ── Review Agent ─────────────────────────────────────────────────────────


class ReviewAgent:
    """文献相关性评审 Agent。

    对检索到的论文做相关性评分、去重、矛盾检测，输出排序后的 Top-K。

    Attributes:
        model_name:  LLM 模型名
        temperature: 温度（0.2，评分稳定）
        llm:         ChatOpenAI 实例
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 512,
    ):
        """初始化 ReviewAgent。"""
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = create_chat_model(model_name, temperature, max_tokens)
        logger.info("ReviewAgent ready: model=%s, temp=%.2f", model_name, temperature)

    # ── 公开接口 ────────────────────────────────────────────────────────

    def review(
        self,
        papers: list[dict[str, Any]],
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """完整评审流水线：去重 → 评分 → 矛盾检测 → 排序 → Top-K。

        Args:
            papers: 待评审的论文列表（统一 schema）
            query:  用户问题
            top_k:  输出论文数（默认 10）

        Returns:
            List[dict] — 按 score 降序的 Top-K 论文，含 score/reason/conflict 字段
        """
        if not papers:
            logger.warning("No papers to review")
            return []

        deduped = self.deduplicate_papers(papers)
        scored = self.score_papers(deduped, query)
        conflicts = self.detect_conflicts(scored)

        merged = self._apply_conflicts(scored, conflicts)
        merged.sort(key=lambda x: x.get("score", 0), reverse=True)

        logger.info("Review: %d → %d papers (top-%d)", len(papers), len(deduped), top_k)
        return merged[:top_k]

    def score_papers(
        self,
        papers: list[dict[str, Any]],
        query: str,
    ) -> list[dict[str, Any]]:
        """对每篇论文做 LLM 相关性评分。

        Returns:
            每篇论文附上 score(0-10) 与 reason 字段的列表
        """
        scored = []
        for paper in papers:
            entry = dict(paper)
            try:
                score, reason = self._score_paper(paper, query)
            except Exception as e:
                logger.warning("Scoring failed for '%s': %s", paper.get("title", "")[:40], e)
                score, reason = 0, "scoring failed"
            entry["score"] = score
            entry["reason"] = reason
            entry["conflict"] = False
            scored.append(entry)
        return scored

    def detect_conflicts(
        self,
        papers: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """识别论文间「结论相悖」的论文对。

        Args:
            papers: 已评分的论文列表

        Returns:
            List[dict] — 每个元素为 {paper_a, paper_b, description}
        """
        if len(papers) < 2:
            return []

        paper_list = "\n".join(
            f"- {p.get('title', '')} | {(p.get('abstract', '') or '')[:200]}"
            for p in papers[:20]
        )
        prompt = CONFLICT_PROMPT.format(paper_list=paper_list)

        try:
            data = self._call_llm(prompt)
            conflicts = data.get("conflicts", [])
            if not isinstance(conflicts, list):
                conflicts = []
        except Exception as e:
            logger.warning("Conflict detection failed: %s", e)
            conflicts = []

        logger.info("Conflict detection: %d conflict pair(s)", len(conflicts))
        return conflicts

    # ── 三级去重 ────────────────────────────────────────────────────────

    def deduplicate_papers(
        self,
        papers: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """三级去重：硬去重(DOI/PMID) + 软去重(标题相似度) + 跨源去重。

        Args:
            papers: 原始论文列表

        Returns:
            去重后的论文列表（保持首次出现顺序）
        """
        # 第 1 级 + 第 3 级：硬去重（DOI/PMID）+ 跨源（归一化标题一致）
        papers = _hard_dedup(papers)

        # 第 2 级：软去重（标题相似度 > 0.9）
        return self._dedup_by_title_similarity(papers, threshold=0.9)

    @staticmethod
    def _dedup_by_title_similarity(
        papers: list[dict[str, Any]],
        threshold: float = 0.9,
    ) -> list[dict[str, Any]]:
        """用 difflib 标题相似度去重（阈值 0.9）。

        同一篇论文在 PubMed 和 arXiv 的标题可能略有差异（大小写、标点、
        版本号），硬去重会漏掉这类跨源重复，标题相似度能抓住它们。
        """
        result: list[dict[str, Any]] = []
        for paper in papers:
            title = (paper.get("title") or "").strip().lower()
            if not title:
                result.append(paper)
                continue
            is_dup = any(
                difflib.SequenceMatcher(None, title, (r.get("title") or "").lower()).ratio()
                > threshold
                for r in result
                if (r.get("title") or "").strip()
            )
            if not is_dup:
                result.append(paper)
        return result

    # ── 内部实现 ────────────────────────────────────────────────────────

    def _score_paper(self, paper: dict[str, Any], query: str) -> tuple[int, str]:
        """对单篇论文打分，返回 (score, reason)。"""
        prompt = SCORE_PROMPT.format(
            query=query,
            title=paper.get("title", ""),
            abstract=(paper.get("abstract") or "")[:1500],
            source=paper.get("source", "unknown"),
        )
        data = self._call_llm(prompt)
        score = int(data.get("score", 0))
        score = max(0, min(10, score))
        return score, str(data.get("reason", ""))

    @staticmethod
    def _apply_conflicts(
        papers: list[dict[str, Any]],
        conflicts: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """把矛盾检测结果标记到对应论文的 conflict 字段。"""
        if not conflicts:
            return papers

        conflict_titles = set()
        for c in conflicts:
            conflict_titles.add((c.get("paper_a") or "").strip().lower())
            conflict_titles.add((c.get("paper_b") or "").strip().lower())

        for paper in papers:
            title = (paper.get("title") or "").strip().lower()
            if title in conflict_titles:
                paper["conflict"] = True
        return papers

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _call_llm(self, prompt: str) -> dict[str, Any]:
        """调用 LLM 并解析 JSON（3 次重试 + 指数退避）。"""
        response = self.llm.invoke(prompt)
        return parse_json_response(response.content)
