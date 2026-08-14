"""Biomedical Research Agent — 可观测性（Observability）包。

LangFuse 调用链追踪：记录每个 Agent 节点的输入 / 输出 / 耗时 / 成本。
监控是「旁路」功能——LangFuse 不可用时必须静默降级，不能拖垮主流程。
"""

from src.observability.tracing import LangfuseTracer

__all__ = ["LangfuseTracer"]
