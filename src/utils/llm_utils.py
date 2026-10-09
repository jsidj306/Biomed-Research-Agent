"""
LLM 通用工具 — JSON 三级容错解析 + Chat 模型工厂。

MVP 阶段在 summarizer.py 中内联了一套「markdown fence 去除 → 直接 json.loads
→ 子串提取」的容错逻辑。V1.0 的 Supervisor / Review / Summarizer 多个 Agent
都要用同样的解析，故抽取到这里作为单一职责的共享工具。

使用示例:
    from src.utils.llm_utils import parse_json_response, create_chat_model

    llm = create_chat_model(model_name="glm-5.2", temperature=0.2)
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


# ── 服务商路由 ─────────────────────────────────────────────────────────

# 模型名前缀 → (base_url, API key 环境变量)。主审用智谱 GLM、
# 交叉验证用月之暗面 Kimi，各自独立网关与 key。
_PROVIDERS: tuple[tuple[str, str, str], ...] = (
    ("glm", "https://open.bigmodel.cn/api/paas/v4", "ZHIPU_API_KEY"),
    ("kimi", "https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
    ("moonshot", "https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
)

# 原生 OpenAI 模型前缀：仅这些模型才需要 langchain-openai 把 max_tokens 改写为
# max_completion_tokens（OpenAI 2024-09 的弃用改法）。第三方厂商的 OpenAI 兼容
# 网关（智谱 / Kimi 等）只认 max_tokens，传 max_completion_tokens 会报
# "max_completion_tokens: Extra inputs are not permitted"。
_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")


def _resolve_provider(model_name: str) -> tuple[str, str]:
    """按模型名前缀解析 (base_url, api_key)。

    Args:
        model_name: 模型名（如 "glm-5.2" / "kimi-k2.7-code"）

    Returns:
        (base_url, api_key) 二元组；未匹配的模型回退 OPENAI_* 环境变量
    """
    lower = model_name.lower()
    for prefix, base_url, key_env in _PROVIDERS:
        if lower.startswith(prefix):
            return base_url, os.getenv(key_env, "")
    return (
        os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        os.getenv("OPENAI_API_KEY", ""),
    )


def _is_native_openai(model_name: str) -> bool:
    """判断是否为原生 OpenAI 模型（仅这些才传 max_tokens → max_completion_tokens）。"""
    return model_name.lower().startswith(_OPENAI_PREFIXES)


def _normalize_temperature(model_name: str, temperature: float) -> float:
    """按模型归一化 temperature。

    Kimi K2.7 code 是思维链模型，只接受 temperature=1（推理模型限制），传其它值会报
    "invalid temperature: only 1 is allowed"。故对其强制置 1。
    """
    if model_name.lower().startswith("kimi-k2.7-code"):
        return 1.0
    return temperature


def create_chat_model(
    model_name: str = "glm-5.2",
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> ChatOpenAI:
    """创建 ChatOpenAI 实例（按模型名自动路由到智谱 GLM / Kimi / OpenAI 兼容）。

    Args:
        model_name:  模型名（glm-5.2 / kimi-k2.7-code / gpt-4o-mini ...）
        temperature: 温度（越低越确定，适合计划/评分类任务）
        max_tokens:  最大输出 token 数（仅原生 OpenAI 模型生效，其余用厂商默认）

    Returns:
        配置好的 ChatOpenAI 实例
    """
    base_url, api_key = _resolve_provider(model_name)

    kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": _normalize_temperature(model_name, temperature),
        "api_key": api_key,
        "base_url": base_url,
        # 单次调用超时（秒）。默认走底层 openai 客户端的 600s，API 挂起时整条
        # 流水线会「一动不动」卡满 10 分钟；显式收紧到 180s，卡住也能快速失败重试。
        "request_timeout": 180,
    }
    if _is_native_openai(model_name):
        kwargs["max_tokens"] = max_tokens

    return ChatOpenAI(**kwargs)
