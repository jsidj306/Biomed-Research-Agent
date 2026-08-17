"""
Planner Agent — 研究方向推荐 + 实验假设生成。

让系统从「查文献 / 读文献」升级到「给你下一个研究题」。它基于 Gap Analysis
产出的研究空白，做两件事:
  1. 研究方向推荐：从研究空白里挑出「有可行性 + 有创新性」的方向，给优先级
  2. 实验假设生成：为每个方向生成可验证的因果假设（"若 A 则 B，因 C"）

与 Summarizer 的 Gap Analysis 关系::
    Gap Analysis 回答「还有什么没被研究」→ Planner 回答「那我现在该做什么」。
    前者找空白，后者把空白变成可执行的选题和假设，是「从文献到选题」的最后一公里。

使用示例:
    from src.agents.planner_agent import PlannerAgent

    planner = PlannerAgent()
    directions = planner.generate_directions(gaps, papers)
    for d in directions:
        print(d["title"], d["hypothesis"], d["feasibility"], d["novelty"])
"""

from __future__ import annotations

import logging
from typing import Any

from dotenv import load_dotenv
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.llm_utils import create_chat_model, parse_json_response

load_dotenv()

logger = logging.getLogger(__name__)

# ── Prompt 模板 ───────────────────────────────────────────────────────────

PLANNER_PROMPT = """你是生物医学研究规划师，负责把研究空白转成可执行的研究选题。

基于下面给定的研究空白和相关文献，输出 3-5 个研究方向。对每个方向必须同时给出:
- title: 方向标题
- rationale: 为什么值得做（这个空白为何重要）
- hypothesis: 可验证的因果假设，必须写成「若 A 则 B，因 C」的形式
- feasibility: high | medium | low（实验可行性）
- novelty: high | medium | low（创新性）

研究空白:
{gaps}

相关文献:
{papers}

返回 JSON 对象，包含以下键:
{{
  "directions": [
    {{
      "title": "...",
      "rationale": "...",
      "hypothesis": "若 A 则 B，因 C",
      "feasibility": "high",
      "novelty": "high"
    }}
  ]
}}

hypothesis 字段必须含因果结构（若…则…因…），不要输出「值得进一步研究」这类空话。
只返回合法 JSON，不要 markdown 围栏或多余文字。"""

# 合法取值（归一化用）
_VALID_LEVELS = {"high", "medium", "low"}


# ── Planner 实现 ─────────────────────────────────────────────────────────


class PlannerAgent:
    """生物医学研究规划 Agent。

    把研究空白转成带可验证假设 + 可行性/创新性评分的研究方向。

    Attributes:
        model_name:  LLM 模型名
        temperature: 温度（0.3）
        llm:         ChatOpenAI 实例
    """

    def __init__(
        self,
        model_name: str = "glm-5.2",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ):
        """初始化 Planner。"""
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = create_chat_model(model_name, temperature, max_tokens)
        logger.info("PlannerAgent ready: model=%s, temp=%.2f", model_name, temperature)

    # ── 公开接口 ────────────────────────────────────────────────────────

    def generate_directions(
        self,
        gaps: list[Any],
        papers: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """基于研究空白生成 3-5 个研究方向（含假设与评分）。

        Args:
            gaps:   研究空白列表（list[str] 或 list[dict]，dict 取 "gap" 键）
            papers: 相关论文列表（统一 schema，作为背景证据）

        Returns:
            List[dict] — 每个方向含 title/rationale/hypothesis/feasibility/novelty
        """
        gaps_text = self._format_gaps(gaps)
        if not gaps_text:
            logger.warning("No gaps provided, cannot generate directions")
            return []

        prompt = PLANNER_PROMPT.format(
            gaps=gaps_text,
            papers=self._format_papers(papers),
        )
        try:
            data = self._call_llm(prompt)
        except Exception as e:
            logger.warning("Planner generation failed: %s", e)
            return []

        raw = data.get("directions", [])
        if not isinstance(raw, list):
            raw = []

        directions = [self._normalize(d) for d in raw]
        directions = [d for d in directions if d]
        logger.info("Planner: generated %d direction(s)", len(directions))
        return directions

    # ── 内部实现 ────────────────────────────────────────────────────────

    @staticmethod
    def _format_gaps(gaps: list[Any]) -> str:
        """把 gaps 列表格式化为文本（兼容 str / dict 两种输入）。"""
        lines = []
        for i, g in enumerate(gaps or [], 1):
            if isinstance(g, dict):
                text = g.get("gap") or g.get("description") or str(g)
            else:
                text = str(g)
            if text.strip():
                lines.append(f"{i}. {text.strip()}")
        return "\n".join(lines)

    @staticmethod
    def _format_papers(papers: list[dict[str, Any]], max_abstract: int = 200) -> str:
        """把论文列表格式化为编号文本块。"""
        if not papers:
            return "（无相关文献）"
        lines = []
        for i, p in enumerate(papers, 1):
            title = p.get("title") or f"Paper {i}"
            fulltext = p.get("fulltext")
            # 本地文档全文给更长上下文，检索结果仍按 max_abstract 截断
            abstract = fulltext[:3000] if fulltext else (p.get("abstract") or "")[:max_abstract]
            lines.append(f"[{i}] {title}\n    {abstract}")
        return "\n".join(lines)

    @staticmethod
    def _normalize(direction: Any) -> dict[str, Any] | None:
        """归一化单个方向，校验字段合法性，缺 title 的丢弃。"""
        if not isinstance(direction, dict):
            return None

        title = str(direction.get("title", "")).strip()
        if not title:
            return None

        feasibility = str(direction.get("feasibility", "")).strip().lower()
        novelty = str(direction.get("novelty", "")).strip().lower()
        if feasibility not in _VALID_LEVELS:
            feasibility = "medium"
        if novelty not in _VALID_LEVELS:
            novelty = "medium"

        return {
            "title": title,
            "rationale": str(direction.get("rationale", "")).strip(),
            "hypothesis": str(direction.get("hypothesis", "")).strip(),
            "feasibility": feasibility,
            "novelty": novelty,
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _call_llm(self, prompt: str) -> dict[str, Any]:
        """调用 LLM 并解析 JSON（3 次重试 + 指数退避）。"""
        response = self.llm.invoke(prompt)
        return parse_json_response(response.content)
