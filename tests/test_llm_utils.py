"""
LLM 工具 — 单元测试。

运行: uv run pytest tests/test_llm_utils.py -v

重点覆盖 create_chat_model 的多厂商路由：glm/kimi 各自网关与 key，且非原生
OpenAI 模型不传 max_tokens（避免 langchain-openai 改写为 max_completion_tokens
后被第三方网关拒绝）。
"""

import os
from unittest.mock import patch

from src.utils.llm_utils import create_chat_model, parse_json_response


class TestParseJsonResponse:
    """三级容错 JSON 解析。"""

    def test_plain_json(self):
        assert parse_json_response('{"a": 1}') == {"a": 1}

    def test_markdown_fence(self):
        assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}

    def test_substring_extraction(self):
        assert parse_json_response('前缀 {"a": 1} 后缀') == {"a": 1}


class TestCreateChatModel:
    """模型工厂：多厂商路由 + max_tokens 只在原生 OpenAI 模型传入。"""

    def _call(self, model_name, env=None):
        env = env or {}
        with patch.dict(os.environ, env, clear=False), patch(
            "src.utils.llm_utils.ChatOpenAI"
        ) as mock_cls:
            create_chat_model(model_name=model_name, temperature=0.0, max_tokens=1024)
            return mock_cls.call_args.kwargs

    def test_glm_routes_to_zhipu(self):
        """glm-5.2 → 智谱网关 + ZHIPU_API_KEY，且不传 max_tokens。"""
        kwargs = self._call("glm-5.2", {"ZHIPU_API_KEY": "zhipu-key"})
        assert kwargs["base_url"] == "https://open.bigmodel.cn/api/paas/v4"
        assert kwargs["api_key"] == "zhipu-key"
        assert "max_tokens" not in kwargs

    def test_kimi_routes_to_moonshot(self):
        """kimi-k2.7-code → 月之暗面网关 + MOONSHOT_API_KEY，且不传 max_tokens。"""
        kwargs = self._call("kimi-k2.7-code", {"MOONSHOT_API_KEY": "moonshot-key"})
        assert kwargs["base_url"] == "https://api.moonshot.cn/v1"
        assert kwargs["api_key"] == "moonshot-key"
        assert "max_tokens" not in kwargs

    def test_openai_model_keeps_max_tokens(self):
        """原生 OpenAI 模型应传入 max_tokens。"""
        kwargs = self._call("gpt-4o-mini", {"OPENAI_API_KEY": "openai-key"})
        assert kwargs["max_tokens"] == 1024
        assert kwargs["api_key"] == "openai-key"

    def test_kimi_code_forces_temperature_one(self):
        """Kimi K2.7 code 是推理模型，temperature 应被归一化为 1。"""
        kwargs = self._call("kimi-k2.7-code", {"MOONSHOT_API_KEY": "moonshot-key"})
        assert kwargs["temperature"] == 1.0

    def test_glm_keeps_temperature(self):
        """非推理模型保持传入的 temperature。"""
        kwargs = self._call("glm-5.2", {"ZHIPU_API_KEY": "zhipu-key"})
        assert kwargs["temperature"] == 0.0

    def test_unknown_model_falls_back_to_openai_env(self):
        """未匹配的模型回退 OPENAI_BASE_URL / OPENAI_API_KEY。"""
        kwargs = self._call(
            "some-unknown-model",
            {"OPENAI_BASE_URL": "https://custom/v1", "OPENAI_API_KEY": "openai-key"},
        )
        assert kwargs["base_url"] == "https://custom/v1"
        assert kwargs["api_key"] == "openai-key"
        assert "max_tokens" not in kwargs
