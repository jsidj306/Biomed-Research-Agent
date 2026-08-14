"""
LLM 通用工具 — JSON 三级容错解析 + Chat 模型工厂。

MVP 阶段在 summarizer.py 中内联了一套「markdown fence 去除 → 直接 json.loads
→ 子串提取」的容错逻辑。V1.0 的 Supervisor / Review / Summarizer 多个 Agent
都要用同样的解析，故抽取到这里作为单一职责的共享工具。

使用示例:
    from src.utils.llm_utils import parse_json_response, create_chat_model

    llm = create_chat_model(model_name="gpt-4o-mini", temperature=0.2)
    content = llm.invoke(prompt).content
    data = parse_json_response(content)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


def parse_json_response(content: str) -> dict[str, Any]:
    """三级容错地把 LLM 输出解析为 JSON 字典。

    容错策略（由宽松到严格）:
        1. 去除 markdown 代码块围栏（```json ... ```）
        2. 直接 json.loads 整体解析
        3. 提取首个 '{' 到最后一个 '}' 的子串再解析

    Args:
        content: LLM 返回的原始文本

    Returns:
        解析后的 JSON 字典

    Raises:
        ValueError: 三种方式都无法解析时抛出
    """
    text = content.strip()

    # 第 1 级：去除 markdown 围栏
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:]) if len(lines) > 1 else text
        text = text.removesuffix("```").strip()

    # 第 2 级：整体解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 第 3 级：提取 JSON 子串
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse LLM response as JSON: {content[:200]}"
            ) from e

    raise ValueError(f"Failed to parse LLM response as JSON: {content[:200]}")


def create_chat_model(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> ChatOpenAI:
    """创建 ChatOpenAI 实例（走 API2D / OpenAI 兼容中转）。

    Args:
        model_name:  模型名
        temperature: 温度（越低越确定，适合计划/评分类任务）
        max_tokens:  最大输出 token 数

    Returns:
        配置好的 ChatOpenAI 实例
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
    )
