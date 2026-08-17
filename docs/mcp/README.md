# MCP 接入指南

本项目把 7-Agent 研究助手的四个核心能力封装成 **MCP（Model Context Protocol）** 工具，
让 Claude Desktop / Cursor 等 MCP 客户端可以直接调用，无需打开 Streamlit UI。

## 暴露的工具

| 工具 | 说明 | 参数 |
|------|------|------|
| `search_papers` | 多源并行检索文献（PubMed / arXiv / Semantic Scholar / bioRxiv / medRxiv），归一化去重 | `query`, `max_results=10` |
| `summarize_paper` | 单篇论文结构化深度摘要（背景/方法/结果/结论/局限/意义） | `title`, `abstract`, `language="english"` |
| `generate_report` | 完整 7-Agent 流水线：六段式报告 + 研究方向 + Critic 评分 | `query` |
| `query_kg` | 从文本抽取知识图谱三元组（药物-靶点-疾病），附 Cypher + Mermaid | `text` |

## 手动验证（stdio）

```bash
uv sync
uv run python src/mcp/server.py
```

启动后进入 stdio 模式，等待 MCP 客户端连接。也可用单元测试验证：

```bash
uv run pytest tests/test_mcp_server.py -v
```

## 接入 Claude Desktop

把 `docs/mcp/claude_desktop_config.json` 的内容合并进 Claude Desktop 的配置文件：

- **Windows**：`%APPDATA%\Claude\claude_desktop_config.json`
- **macOS**：`~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "biomed-research-agent": {
      "command": "uv",
      "args": ["run", "python", "src/mcp/server.py"],
      "cwd": "D:/QLDownload/Cursor/项目1/biomed-research-agent"
    }
  }
}
```

> `cwd` 须指向项目根目录（改成你本机的实际路径）。重启 Claude Desktop 后，
> 在工具列表中即可看到 `search_papers` 等 4 个工具。

## 接入 Cursor

项目根目录已自带 `.cursor/mcp.json`（Cursor 项目级 MCP 配置，自动加载）：

```json
{
  "mcpServers": {
    "biomed-research-agent": {
      "command": "uv",
      "args": ["run", "python", "src/mcp/server.py"]
    }
  }
}
```

在 Cursor 的 **Settings → MCP** 中确认 `biomed-research-agent` 显示为已连接
（绿色），即可在对话里让 Cursor 直接调用文献检索 / 摘要 / 报告生成 / 知识图谱。

## 前置条件

- 已执行 `uv sync`（安装 `fastmcp` 依赖）
- `.env` 已配置 `ZHIPU_API_KEY` / `MOONSHOT_API_KEY`（报告生成、摘要、知识图谱
  抽取需要 LLM）
- `NCBI_API_KEY` / `NCBI_EMAIL`（文献检索需要）
