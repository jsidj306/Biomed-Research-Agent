"""
LangFuse 调用链追踪 — 多Agent系统的可观测性埋点。

LangFuse 是开源的 LLM 应用可观测性平台（类似 LangSmith），记录每一次
LLM 调用 / Agent 节点的输入、输出、耗时、token 和成本，还能看整个调用链
的「火焰图」。多Agent系统必配，否则线上出问题只能瞎猜。

关键设计原则——监控是「旁路」功能:
    - LangFuse 未安装 / 未配置 Key / 网络异常时，一律静默降级为 no-op，
      绝不让监控问题拖垮主流程。
    - 所有 trace 调用都用 try/except 包裹，失败只记日志。

使用示例:
    from src.observability.tracing import LangfuseTracer

    tracer = LangfuseTracer()

    with tracer.trace("search", input_data={"query": "CAR-T"}):
        papers = search_agent.search_all("CAR-T")
        # 在 span 内 set_output 记录产物
        ...

    # 或显式记录一次调用（含耗时）
    tracer.record("review", input_data={"n": 10}, output_data={"top_k": 5},
                  duration_ms=123.4)

环境变量（.env）:
    LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class LangfuseTracer:
    """LangFuse 调用链追踪器（缺省配置时静默降级为 no-op）。

    Attributes:
        enabled: 是否真正可用（配置了 Key 且客户端初始化成功）
    """

    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
        client: Any = None,
    ):
        """初始化追踪器。

        Args:
            public_key: LangFuse 公钥（缺省读环境变量 LANGFUSE_PUBLIC_KEY）
            secret_key: LangFuse 密钥（缺省读环境变量 LANGFUSE_SECRET_KEY）
            host:       LangFuse 服务地址（缺省读 LANGFUSE_HOST 或官方云端）
            client:     注入的 LangFuse 客户端（测试用，绕过环境变量初始化）
        """
        if client is not None:
            self._client = client
        else:
            pk = public_key or os.getenv("LANGFUSE_PUBLIC_KEY", "")
            sk = secret_key or os.getenv("LANGFUSE_SECRET_KEY", "")
            h = host or os.getenv(
                "LANGFUSE_HOST", "https://cloud.langfuse.com"
            )
            self._client = self._build_client(pk, sk, h) if (pk and sk) else None

        self.enabled = self._client is not None
        if not self.enabled:
            logger.info("LangFuse tracer disabled (no key / client unavailable)")

    # ── 公开接口 ────────────────────────────────────────────────────────

    def trace(self, name: str, input_data: dict | None = None) -> "_TraceSpan":
        """返回节点级追踪的上下文管理器，自动记录耗时。

        Args:
            name:       节点名（如 supervisor / search / review / summarize）
            input_data: 节点输入（可选）

        Returns:
            _TraceSpan — 上下文管理器，进入记录开始时间，退出记录耗时+输出

        Usage::

            with tracer.trace("search", {"query": q}) as span:
                result = agent.search_all(q)
                span.set_output({"n_papers": len(result)})
        """
        return _TraceSpan(self, name, input_data)

    def record(
        self,
        name: str,
        input_data: dict | None = None,
        output_data: dict | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """显式记录一次节点调用。

        Args:
            name:        节点名
            input_data:  输入数据
            output_data: 输出数据
            duration_ms: 耗时（毫秒）
        """
        if not self.enabled:
            return
        try:
            trace = self._client.trace(
                name=name,
                input=input_data,
                output=output_data,
            )
            if duration_ms is not None:
                trace.update(metadata={"duration_ms": round(duration_ms, 2)})
        except Exception as e:  # noqa: BLE001 — 监控降级，绝不影响主流程
            logger.warning("LangFuse record failed (degraded): %s", e)

    # ── 内部实现 ────────────────────────────────────────────────────────

    @staticmethod
    def _build_client(pk: str, sk: str, host: str):
        """尝试构建 LangFuse 客户端，任何异常都降级为 None。"""
        try:
            from langfuse import Langfuse  # 延迟导入，未安装时降级

            return Langfuse(public_key=pk, secret_key=sk, host=host)
        except Exception as e:  # noqa: BLE001
            logger.warning("LangFuse unavailable, tracing disabled: %s", e)
            return None


class _TraceSpan:
    """节点级追踪上下文管理器。"""

    def __init__(self, tracer: LangfuseTracer, name: str, input_data: dict | None):
        self._tracer = tracer
        self._name = name
        self._input = input_data
        self._start: float | None = None
        self._output: dict | None = None

    def __enter__(self) -> "_TraceSpan":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        duration_ms = (time.perf_counter() - self._start) * 1000 if self._start else 0.0
        output = self._output
        if exc_type is not None:
            output = {"error": f"{exc_type.__name__}: {exc_val}"}
        self._tracer.record(self._name, self._input, output, duration_ms)
        return False  # 不吞异常，让主流程照常传播

    def set_output(self, output: dict) -> None:
        """在 span 内设置节点输出（可选）。"""
        self._output = output
