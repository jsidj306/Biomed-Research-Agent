"""
LangFuse 调用链追踪 — 单元测试。

运行: uv run pytest tests/test_langfuse.py -v

通过注入 mock 客户端验证埋点、耗时记录和不可用时的降级，
不依赖真实 LangFuse 服务。
"""

from unittest.mock import MagicMock

import pytest

from src.observability.tracing import LangfuseTracer


# ── 降级测试 ─────────────────────────────────────────────────────────────


class TestDegradation:
    """测试 LangFuse 不可用时的静默降级。"""

    def test_disabled_without_keys(self, monkeypatch):
        """未配置 Key 时应禁用且不抛异常。"""
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

        tracer = LangfuseTracer()
        assert tracer.enabled is False

        # 降级模式下 record / trace 都是 no-op，不抛异常
        tracer.record("search", input_data={"q": "x"})
        with tracer.trace("search"):
            pass

    def test_client_build_failure_degrades(self):
        """langfuse 未安装（ImportError）时 _build_client 应返回 None。"""
        assert LangfuseTracer._build_client("pk", "sk", "host") is None


# ── 埋点测试 ─────────────────────────────────────────────────────────────


class TestTracing:
    """测试埋点与耗时记录（mock 客户端）。"""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.trace.return_value = MagicMock()  # trace() 返回可 update 的对象
        return client

    def test_record_calls_client(self, mock_client):
        """record 应调用客户端 trace 并记录数据。"""
        tracer = LangfuseTracer(client=mock_client)
        assert tracer.enabled is True

        tracer.record(
            "review", input_data={"n": 10}, output_data={"top_k": 5},
            duration_ms=42.5,
        )

        mock_client.trace.assert_called_once()
        args, kwargs = mock_client.trace.call_args
        assert kwargs["name"] == "review"
        assert kwargs["input"] == {"n": 10}

    def test_trace_measures_duration(self, mock_client):
        """trace 上下文管理器应记录正数耗时。"""
        tracer = LangfuseTracer(client=mock_client)

        with tracer.trace("search", input_data={"query": "CAR-T"}) as span:
            span.set_output({"n_papers": 3})

        # 检查 record 触发了 trace，且带 metadata duration_ms
        assert mock_client.trace.called
        # 返回的 trace 对象上 update 被调用（含 duration_ms）
        returned_trace = mock_client.trace.return_value
        returned_trace.update.assert_called_once()
        metadata = returned_trace.update.call_args[1]["metadata"]
        assert metadata["duration_ms"] >= 0

    def test_trace_does_not_break_flow(self, mock_client):
        """span 内抛异常不应被吞掉，但异常信息应被记录。"""
        tracer = LangfuseTracer(client=mock_client)

        with pytest.raises(ValueError):
            with tracer.trace("review"):
                raise ValueError("boom")

        # 异常仍被记录到 output
        assert mock_client.trace.called

    def test_record_swallows_client_error(self, mock_client):
        """客户端记录抛异常时不应向上传播（监控降级）。"""
        mock_client.trace.side_effect = RuntimeError("network down")
        tracer = LangfuseTracer(client=mock_client)

        # 不应抛异常
        tracer.record("search", input_data={"q": "x"})
