# 贡献指南（Contributing）

感谢你为 Biomedical Research Agent 贡献代码。本文档说明开发规范、测试约定与提交流程。

---

## 1. 开发环境

```bash
git clone git@github.com:jsidj306/Biomed-Research-Agent.git
cd Biomed-Research-Agent
uv sync                       # 安装依赖（含 dev 组：ruff）
uv run pytest                 # 跑测试确认环境正常
```

---

## 2. 代码规范

- **Python 3.11+**，类型标注覆盖所有函数签名（`typing` / 内置泛型）。
- **Google 风格 docstring**，模块顶部注明用途与使用示例。
- **LLM 调用必须带重试**——用 `src/utils/llm_utils.py` 的 `create_chat_model` +
  `parse_json_response`，并加 `tenacity @retry`。
- **降级优先**——外部依赖（LLM / Neo4j / LangFuse / Unstructured）一律延迟导入，
  失败降级为「空结果 + warning 日志」，不因单点失败崩溃。
- **中文字符串注释**：项目约定注释与 docstring 用中文，代码标识符用英文。

### Lint

```bash
uv run ruff check src/ tests/          # 检查
uv run ruff check --fix src/ tests/    # 自动修复
```

`ruff` 配置见 `pyproject.toml`（line-length=100，忽略 B017/BLE001/S110 —— 这三条是
降级策略下的有意为之，详见配置注释）。

---

## 3. 测试约定

- 测试文件放在 `tests/`，命名 `test_<模块>.py`。
- **Mock 一切外部依赖**：
  - LLM：`patch("src.agents.xxx.create_chat_model", return_value=mock_llm)`
  - 缺失的可选依赖：`patch.dict(sys.modules, {"fitz": ...})` 或 `{"weasyprint": None}`
  - 需要真实文件的测试用 `pytest` 的 `tmp_path` fixture 造占位文件。
- **确定性优先**：引用核查、Cypher 生成、词典兜底等纯函数逻辑必须直接测，不 mock。
- 标记 `@pytest.mark.slow` 给「需下载/加载大模型」的测试（如 PubMedBERT）。

```bash
uv run pytest                  # 全量
uv run pytest -m "not slow"    # 跳过慢测试（CI 默认）
uv run pytest tests/test_critic_agent.py -v   # 单文件
```

覆盖率门槛 **≥70%**（`pytest-cov`，`--cov=src`）。

---

## 4. 新增一个 Agent 的步骤

以新增 `FooAgent` 为例：

1. 在 `src/agents/foo_agent.py` 实现，遵循现有 Agent 结构（`__init__` 建 LLM、公开方法、
   `@retry` 的 `_call_llm`）。
2. 在 `src/agents/__init__.py` 导出。
3. 在 `src/agents/graph.py` 的 `build_graph()` 加参数 + 节点 + 边（保持测试可注入 mock）。
4. 在 `src/agents/graph.py` 的 `AgentState` 中声明新增状态字段（若需要）。
5. 写 `tests/test_foo_agent.py`（mock LLM），并更新 `tests/test_graph.py`。
6. 跑 `uv run ruff check` + `uv run pytest -m "not slow"`。

---

## 5. 提交规范

- 提交信息用 conventional commits：`feat:` / `fix:` / `docs:` / `test:` / `refactor:` /
  `chore:`，中文或英文均可。
- 一个提交做一件事，避免「大杂烩」提交。
- 涉及文档的变更同步更新 README 与 `docs/`。

---

## 6. 常见坑

| 坑 | 规避方式 |
|----|---------|
| 项目 1.xlsx 用 openpyxl save 会丢格式 | 见项目 memory；用 shared string 追加，写入前关闭 Excel |
| 大视频提交撑爆 git 仓库 | 录屏等大文件先用 ffmpeg 压缩或走 Git LFS |
| 扫描版 PDF 提取为空 | 全文加载器会明确报错，需提供文字版 PDF 或启用 OCR |
| 容器内连不上 chroma/neo4j | 用服务名（`chroma`/`neo4j`）而非 `localhost` 做 host |
