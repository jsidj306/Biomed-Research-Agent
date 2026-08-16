"""
LangFuse 调用链追踪 — 单元测试。

运行: uv run pytest tests/test_langfuse.py -v

通过注入 mock 客户端验证埋点、耗时记录和不可用时的降级，
不依赖真实 LangFuse 服务。mock 对象模拟 LangFuse Python SDK v3+ 的
``start_observation`` → ``update`` / ``end`` 调用链。
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

    def test_client_build_failure_degrades(self, monkeypatch):
        """langfuse 未安装（ImportError）时 _build_client 应返回 None。"""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "langfuse":
                raise ImportError("no langfuse")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert LangfuseTracer._build_client("pk", "sk", "host") is None


# ── 埋点测试 ─────────────────────────────────────────────────────────────


class TestTracing:
    """测试埋点与耗时记录（mock 客户端）。"""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.start_observation.return_value = MagicMock()  # 返回可 update/end 的对象
        return client

    def test_record_calls_client(self, mock_client):
        """record 应调用客户端 start_observation 并 end。"""
        tracer = LangfuseTracer(client=mock_client)
        assert tracer.enabled is True

        tracer.record(
            "review", input_data={"n": 10}, output_data={"top_k": 5},
            duration_ms=42.5,
        )

        mock_client.start_observation.assert_called_once()
        kwargs = mock_client.start_observation.call_args[1]
        assert kwargs["name"] == "review"
        assert kwargs["input"] == {"n": 10}
        assert kwargs["metadata"] == {"duration_ms": 42.5}
        # 一次性 span 应被 end
        mock_client.start_observation.return_value.end.assert_called_once()

    def test_trace_updates_output_and_ends(self, mock_client):
        """trace 上下文管理器应在退出时 update 输出并 end span。"""
        tracer = LangfuseTracer(client=mock_client)

        with tracer.trace("search", input_data={"query": "CAR-T"}) as span:
            span.set_output({"n_papers": 3})

        assert mock_client.start_observation.called
        obs = mock_client.start_observation.return_value
        obs.update.assert_called_once()
        assert obs.update.call_args[1]["output"] == {"n_papers": 3}
        obs.end.assert_called_once()

    def test_trace_does_not_break_flow(self, mock_client):
        """span 内抛异常不应被吞掉，但异常信息应被记录。"""
        tracer = LangfuseTracer(client=mock_client)

        with pytest.raises(ValueError), tracer.trace("review"):
            raise ValueError("boom")

        # 异常仍被记录到 output
        assert mock_client.start_observation.called
        obs = mock_client.start_observation.return_value
        obs.update.assert_called_once()
        assert "error" in obs.update.call_args[1]["output"]

    def test_record_swallows_client_error(self, mock_client):
        """客户端记录抛异常时不应向上传播（监控降级）。"""
        mock_client.start_observation.side_effect = RuntimeError("network down")
        tracer = LangfuseTracer(client=mock_client)

        # 不应抛异常
        tracer.record("search", input_data={"q": "x"})

    def test_flush_calls_client(self, mock_client):
        """flush 应透传到客户端。"""
        tracer = LangfuseTracer(client=mock_client)
        tracer.flush()
        mock_client.flush.assert_called_once()
