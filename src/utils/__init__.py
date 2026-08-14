"""Biomedical Research Agent — 通用工具包。

存放跨模块复用的基础设施：LLM JSON 容错解析、Chat 模型工厂等。
"""

from src.utils.llm_utils import create_chat_model, parse_json_response

__all__ = ["create_chat_model", "parse_json_response"]
