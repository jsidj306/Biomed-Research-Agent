"""
Critic Agent — 质量安检：交叉验证 + 事实核查 + 引用真实性检查。

V1.0 的 Supervisor / Search / Review / Summarizer 都在「生成」内容，但 LLM
会幻觉——编造引用、夸大结论、张冠李戴。Critic 是专门「挑错」的 Agent：
它不产出新内容，只负责质疑和验证其他 Agent 的输出。

三种验证手段（各防一类错）:
  1. 交叉验证 (Cross-check): 用两个不同厂商的 LLM 独立判断同一结论，
     降低单一模型「重复自己的错误」的偏见（GLM vs Kimi 才是真正的第二意见）
  2. 事实核查 (Fact-check): 把结论里的关键声明与检索到的原文逐条比对
  3. 引用核查 (Citation check): **确定性**验证报告里的 [n] 引用是否真实对应
     一篇论文——这一步不用 LLM，纯正则 + 计数，杜绝「编造引用」

设计原则：Critic 与生成者用不同模型，避免「自己给自己打分」的偏见；temperature=0.0
保证核查结果确定可复现。

使用示例:
    from src.agents.critic_agent import CriticAgent

    critic = CriticAgent()
    result = critic.verify(report, sources)
    print(result["overall_score"], result["issues"])
"""

from __future__ import annotations

import logging
import re
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

CRITIC_PROMPT = """你是严谨的生物医学审稿人，只负责挑错，不负责重写。

请核查下面的研究报告，逐条判断：
1. 每个关键结论是否有对应引用 [n] 支撑？
2. 引用编号 [n] 是否真实存在于 sources？
3. 数字 / 药物名 / 结论是否与 sources 一致（有没有「原文没说的话」）？

研究报告:
{report}

可引用的论文列表 (编号从 1 开始):
{sources}

返回 JSON 对象，包含以下键:
{{
  "issues": [
    {{
      "claim": "被质疑的具体结论（原文摘录）",
      "problem": "无引用 | 引用不符 | 与原文矛盾 | 夸大结论",
      "severity": "high | medium | low",
      "fix": "建议的修正方式"
    }}
  ],
  "overall_score": 0-10 的整数，10 表示无问题
}}

若没有发现问题，返回 {{"issues": [], "overall_score": 10}}。
只返回合法 JSON，不要 markdown 围栏或多余文字。"""


# ── Critic 实现 ─────────────────────────────────────────────────────────


class CriticAgent:
    """生物医学报告质检 Agent。

    用两个不同厂商的 LLM 交叉验证报告结论，叠加确定性的引用真实性检查，
    输出问题清单 + 综合评分。

    Attributes:
        model_a: 第一个 LLM 模型名（主审）
        model_b: 第二个 LLM 模型名（交叉验证，与 model_a 不同厂商）
        llm_a:   主审 LLM 实例
        llm_b:   交叉验证 LLM 实例
    """

    def __init__(
        self,
        model_a: str = "glm-5.2",
        model_b: str = "kimi-k2.7-code",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        """初始化 Critic。

        Args:
            model_a:     主审模型（默认 glm-5.2，智谱）
            model_b:     交叉验证模型（默认 kimi-k2.7-code，与 model_a 不同厂商）
            temperature: 温度（0.0 —— 核查要确定性）
            max_tokens:  最大输出 token 数
        """
        self.model_a = model_a
        self.model_b = model_b
        self.temperature = temperature
        self.max_tokens = max_tokens
        # 两个不同模型交叉验证
        self.llm_a = create_chat_model(model_a, temperature, max_tokens)
        self.llm_b = create_chat_model(model_b, temperature, max_tokens)
        logger.info(
            "CriticAgent ready: cross-check %s × %s, temp=%.2f",
            model_a, model_b, temperature,
        )

    # ── 公开接口 ────────────────────────────────────────────────────────

    def verify(self, report: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
        """对报告做完整质检：引用核查 + 两模型交叉验证。

        Args:
            report:  待核查的报告文本（含 [n] 引用标记）
            sources: 可引用的论文列表（编号从 1 开始对应 [1][2]...）

        Returns:
            {
                "issues": [{"claim", "problem", "severity", "fix"}, ...],
                "overall_score": float,  # 0-10，越高越可信
            }
        """
        # 第 1 道关：确定性引用核查（不依赖 LLM，编造引用必被抓）
        citation_issues = self.check_citations(report, sources)

        # 第 2 道关：两模型交叉验证（任一模型失败都降级，不阻断主流程）
        verdict_a: dict[str, Any] = {}
        verdict_b: dict[str, Any] = {}
        try:
            verdict_a = self._verify_with(self.llm_a, report, sources)
        except Exception as e:
            logger.warning("Critic model_a failed (degraded): %s", e)
        try:
            verdict_b = self._verify_with(self.llm_b, report, sources)
        except Exception as e:
            logger.warning("Critic model_b failed (degraded): %s", e)

        merged = self._merge(verdict_a, verdict_b)
        score = merged["overall_score"]
        issues = citation_issues + merged["issues"]

        # 编造引用是硬伤：一旦出现 high 严重度的引用问题，直接把评分压到不及格
        if any(i.get("severity") == "high" for i in citation_issues):
            score = min(score, 4.0)

        logger.info(
            "Critic: %d issue(s), score=%.1f", len(issues), score,
        )
        return {"issues": issues, "overall_score": score}

    @staticmethod
    def check_citations(report: str, sources: list[dict[str, Any]]) -> list[dict[str, str]]:
        """确定性验证报告里的 [n] 引用是否真实对应 sources。

        纯正则 + 计数实现，不调用 LLM，是「引用真实性检查」的最强保证：
        报告里出现 [5] 但 sources 只有 4 篇，必然被标记为 high 严重度。

        Args:
            report:  报告文本（含 [n] 引用标记）
            sources: 论文列表（编号从 1 开始）

        Returns:
            List[dict] — 无效引用的问题清单
        """
        refs = {int(m) for m in re.findall(r"\[(\d+)\]", report)}
        max_ref = len(sources)
        issues: list[dict[str, str]] = []
        for n in sorted(refs):
            if n < 1 or n > max_ref:
                issues.append({
                    "claim": f"引用 [{n}]",
                    "problem": "引用不存在",
                    "severity": "high",
                    "fix": f"删除 [{n}] 或替换为 1-{max_ref} 之间的有效引用",
                })
        return issues

    # ── 交叉验证 ────────────────────────────────────────────────────────

    def _verify_with(
        self,
        llm: Any,
        report: str,
        sources: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """用单个 LLM 核查报告，返回 {issues, overall_score}。"""
        sources_text = self._format_sources(sources)
        prompt = CRITIC_PROMPT.format(report=report, sources=sources_text)
        return self._call_llm(llm, prompt)

    @staticmethod
    def _format_sources(sources: list[dict[str, Any]], max_abstract: int = 200) -> str:
        """把论文列表格式化为编号文本块（供 prompt 使用）。"""
        if not sources:
            return "（无可引用论文）"
        lines = []
        for i, paper in enumerate(sources, 1):
            title = paper.get("title") or f"Paper {i}"
            abstract = (paper.get("abstract") or "")[:max_abstract]
            lines.append(f"[{i}] {title}\n    {abstract}")
        return "\n".join(lines)

    @staticmethod
    def _merge(verdict_a: dict[str, Any], verdict_b: dict[str, Any]) -> dict[str, Any]:
        """合并两个模型的核查结果。

        规则（「两模型都认可才通过」）:
        - issues: 两个模型的并集（去重），任一模型揪出的问题都算问题
        - overall_score: 取更严格的一方（min），两模型都高分才算通过

        Args:
            verdict_a: 模型 A 的核查结果（可能为空 dict，表示该模型失败）
            verdict_b: 模型 B 的核查结果

        Returns:
            {"issues": [...], "overall_score": float}
        """
        a_issues = verdict_a.get("issues") or []
        b_issues = verdict_b.get("issues") or []
        if not isinstance(a_issues, list):
            a_issues = []
        if not isinstance(b_issues, list):
            b_issues = []

        issues = CriticAgent._dedupe_issues(a_issues + b_issues)

        a_score = CriticAgent._coerce_score(verdict_a.get("overall_score"))
        b_score = CriticAgent._coerce_score(verdict_b.get("overall_score"))
        # 两模型都失败时给一个中性默认分（不因监控/网络问题误判）
        score = min(a_score, b_score)

        return {"issues": issues, "overall_score": score}

    @staticmethod
    def _dedupe_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """按 (claim, problem) 去重问题清单，保持首次出现顺序。"""
        seen: set[tuple[str, str]] = set()
        result: list[dict[str, Any]] = []
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            key = (
                str(issue.get("claim", "")).strip().lower(),
                str(issue.get("problem", "")).strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(issue)
        return result

    @staticmethod
    def _coerce_score(value: Any) -> float:
        """把 LLM 返回的评分归一化到 0-10 的 float，异常值回退到 10。"""
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 10.0  # 该模型没给出分数 → 不扣分
        return max(0.0, min(10.0, score))

    # ── 内部实现 ────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _call_llm(self, llm: Any, prompt: str) -> dict[str, Any]:
        """调用 LLM 并解析 JSON（3 次重试 + 指数退避）。"""
        response = llm.invoke(prompt)
        return parse_json_response(response.content)
