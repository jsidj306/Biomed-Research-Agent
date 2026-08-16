"""
Supervisor Agent — 多Agent架构的「项目经理」。

它不亲自搜文献 / 写摘要，而是负责:
  1. classify_intent — 意图解析（区分「检索类」vs「总结类」问题）
  2. generate_plan   — 把用户问题拆解为可执行的研究步骤计划

这是「规划-执行分离」的核心：Supervisor 产出计划，由 Search / Review /
Summarize 等专职 Agent 执行，让每个 Agent 职责单一。

使用示例:
    from src.agents.supervisor import SupervisorAgent

    sup = SupervisorAgent()
    intent = sup.classify_intent("查找CAR-T治疗实体瘤的最新临床试验")
    plan = sup.generate_plan("查找CAR-T治疗实体瘤的最新临床试验")
    print(intent, plan)
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

INTENT_PROMPT = """你是生物医学研究主管，负责判断用户问题的类型。

判断规则:
- 检索类 (search): 需要搜索 / 查找生物医学文献、临床试验、最新研究。例如"查找CAR-T治疗实体瘤的最新临床试验"、"有哪些关于mRNA疫苗的研究"
- 总结类 (summarize): 用户已提供内容，只需总结 / 分析 / 提炼。例如"总结这段摘要"、"帮我分析这篇论文"

用户问题: {query}

返回 JSON 对象，包含以下键:
{{
  "intent": "search" 或 "summarize",
  "reason": "一句话说明判断理由"
}}

只返回合法 JSON，不要 markdown 围栏或多余文字。"""

PLAN_PROMPT = """你是资深生物医学研究员，负责把用户问题拆解为可执行的研究步骤。

把以下用户问题拆解为 3-5 个可执行的研究步骤。要求:
- 步骤具体、无歧义，可直接交给文献检索 Agent 执行
- 每个步骤聚焦一个明确的研究目标
- 用英文写步骤（便于检索 Agent 直接使用）

用户问题: {query}

返回 JSON 对象，包含以下键:
{{
  "plan": ["步骤1", "步骤2", "步骤3", ...]
}}

只返回合法 JSON，不要 markdown 围栏或多余文字。"""


# ── Supervisor 实现 ───────────────────────────────────────────────────────


class SupervisorAgent:
    """生物医学研究主管 Agent。

    负责意图解析和研究步骤计划生成，是 LangGraph 编排中的首个节点。

    Attributes:
        model_name:  LLM 模型名
        temperature: 温度（0.2，保证计划稳定可复现）
        llm:         ChatOpenAI 实例
    """

    def __init__(
        self,
        model_name: str = "glm-5.2",
        temperature: float = 0.2,
        max_tokens: int = 512,
    ):
        """初始化 Supervisor。

        Args:
            model_name:  LLM 模型名
            temperature: 温度（低温度 → 计划输出稳定，便于测试与演示）
            max_tokens:  最大输出 token 数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = create_chat_model(model_name, temperature, max_tokens)
        logger.info(
            "SupervisorAgent ready: model=%s, temp=%.2f",
            model_name, temperature,
        )

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def classify_intent(self, query: str) -> str:
        """判断用户问题的意图类型。

        Args:
            query: 用户问题

        Returns:
            str — "search"（检索类）或 "summarize"（总结类）
        """
        prompt = INTENT_PROMPT.format(query=query)
        data = self._call_llm(prompt)
        intent = str(data.get("intent", "")).strip().lower()
        # 归一化：任何异常值回退到 search（保守默认检索）
        if intent not in ("search", "summarize"):
            logger.warning("Unrecognized intent '%s', fallback to 'search'", intent)
            intent = "search"
        logger.info("Intent for '%s': %s", query[:60], intent)
        return intent

    def generate_plan(self, query: str) -> list[str]:
        """把用户问题拆解为研究步骤计划。

        Args:
            query: 用户问题

        Returns:
            list[str] — 3-5 个可执行的研究步骤
        """
        prompt = PLAN_PROMPT.format(query=query)
        data = self._call_llm(prompt)
        plan = data.get("plan", [])
        if isinstance(plan, str):
            plan = [plan]
        steps = [str(s).strip() for s in plan if str(s).strip()]
        logger.info("Generated %d steps for '%s'", len(steps), query[:60])
        return steps

    # ── 内部实现 ──────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _call_llm(self, prompt: str) -> dict[str, Any]:
        """调用 LLM 并解析 JSON 响应（3 次重试 + 指数退避）。

        Args:
            prompt: 完整 prompt 文本

        Returns:
            解析后的 JSON 字典

        Raises:
            ValueError: JSON 解析失败
        """
        response = self.llm.invoke(prompt)
        return parse_json_response(response.content)
