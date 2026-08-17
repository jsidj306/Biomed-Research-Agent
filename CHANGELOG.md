# Changelog

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。所有值得记录的变化都在此文件。

## [2.1.0] - 2026-08-17

### Added

- **MCP Server 封装**（`src/mcp/server.py`）：用 FastMCP 把 `search_papers` /
  `summarize_paper` / `generate_report` / `query_kg` 四个能力封装为 MCP 工具，
  供 Claude Desktop / Cursor 直接调用（stdio 传输）；附 `.cursor/mcp.json`、
  `docs/mcp/claude_desktop_config.json` 配置示例与接入指南（`docs/mcp/README.md`）。
- **MiniMind LoRA 微调实验**（`src/finetune/minimind_model.py` + `train_lora_local.py` +
  `src/finetune/data.py`）：在实验课产出的本地 MiniMind 权重（2 层 / 12M，
  `pretrain_512.pth`，纯 `.pth` 离线加载，无外部下载）上手动实现 LoRA 微调生物医学
  指令数据；确定性生成 800 条「药物-靶点-疾病」指令（8:1:1 划分，
  `scripts/generate_biomed_instruct.py`）；实测验证集困惑度 **156 → 6.72（-95.7%）**，
  同时诚实记录「2 层模型容量不足、生成仍不连贯」的负面结论。附
  `tests/test_biomed_data.py`、`tests/test_minimind_model.py`、
  `scripts/eval_minimind_compare.py` 与实验记录 `docs/minimind_lora_experiment.md`。
- **多语言支持**（`src/ui/i18n.py`）：UI 文案中英双语字典 + `t(lang, key)` 纯函数，
  `render_sidebar` 新增语言切换，界面语言由 `st.session_state.lang` 控制。
- **CNKI/万方检索**（`src/tools/cnki_tool.py`）：中文文献搜索工具，经 ChinaXiv
  OAI-PMH 开放接口收割中文预印本文献（CNKI/万方无公开 API），客户端按查询词过滤
  排序，统一返回与 PubMed/arXiv 一致的 schema；已接入 UI 检索数据源。
- **中文 RAG**（`src/rag/embeddings.py`）：新增 `BAAI/bge-small-zh-v1.5` 中文嵌入
  模型（`bge_zh`），`embed_query` 侧自动加检索指令前缀。
- **多语言测试**：新增 `tests/test_i18n.py` + `tests/test_cnki_tool.py`（共 23 个用例）。

## [2.0.0] - 2026-08-17

### Added

- **7-Agent 编排**：新增 Critic / Writer / Planner 三个 Agent，组成
  `Supervisor → Search → Review → Summarizer ⇄ Critic → Writer → Planner` 状态图。
- **自纠错回环**：Critic 质检 + 条件路由，分数低于阈值（6.0）时打回 Summarizer 重写，
  最多返工 2 轮（`MAX_CRITIC_RETRIES`）。
- **Critic Agent**（`src/agents/critic_agent.py`）：两模型交叉验证（GLM + Kimi）、
  事实核查、确定性引用核查（纯正则，`check_citations`）。
- **Writer Agent**（`src/agents/writer_agent.py`）：六段式报告（Abstract/Methods/
  Findings/Gaps/Directions/References）+ BibTeX 生成（`build_bibtex`）。
- **Planner Agent**（`src/agents/planner_agent.py`）：研究方向推荐 + 可验证因果假设 +
  可行性/创新性评分。
- **知识图谱**（`src/knowledge_graph/`）：`KnowledgeGraphExtractor`（LLM + 词典兜底
  三元组抽取）+ `Neo4jWriter`（MERGE 幂等写入 + 纯函数 Cypher 生成）。
- **PDF 全文解析**（`src/rag/fulltext_loader.py`）：PyMuPDF 主提取 + Unstructured.io
  兜底 + 扫描版降级提示；`RAGPipeline.ingest_fulltext()`。
- **引用溯源**（`src/ui/components.py`）：报告 `[n]` 引用自动链接到 PMID，`render_report`
  组件。
- **报告导出**（`src/report/export.py`）：Markdown / PDF（weasyprint→pandoc 兜底）/
  Word（python-docx）。
- **Docker**：多阶段 `Dockerfile` + `docker-compose.yml` 三服务编排（app + chroma + neo4j）。
- **文档**：重写 README（中英双语）、`docs/architecture.md`、`docs/deployment.md`、
  `docs/CONTRIBUTING.md`。
- **演示资产**：V2.0 演示录屏（`docs/demo/v2_demo_part1.mp4` / `part2.mp4`，压缩后
  合计 36 MB）+ 6 张演示截图（`docs/screenshots/`：流程图 / 知识图谱 / Critic 评分 /
  报告导出 / Docker 部署 / 测试覆盖率）。

### Changed

- `AgentState` 扩展 `gaps` / `critic_score` / `critic_issues` / `critic_retries` /
  `report` / `directions` 字段。
- 测试从 178 增至 251（覆盖 87%），新增 8 个测试文件。

### Fixed

- 修复 ruff lint 33 处（import 排序）+ 2 处（PYI034 `Self` 返回类型、SIM117 嵌套 with）。

---

## [1.0.0] - 2026-08-14

### Added

- 4-Agent 线性流水线：Supervisor / Search / Review / Summarizer。
- 5 数据源并发检索（PubMed / arXiv / Semantic Scholar / bioRxiv / medRxiv）。
- 混合检索（BM25 + 语义 + RRF 融合 + BGE-Reranker），MRR=0.890 / Recall@10=1.0。
- LangFuse 调用链追踪（v4 API）。
- 检索评测集 + Streamlit UI。
- V1.0 演示录屏（`docs/demo/v1_demo.mp4`）。

---

## [0.1.0] - 2026-08-10

### Added

- MVP：搜索 → RAG 语义检索 → 摘要 → 结构化输出的线性流水线（86 tests，82% 覆盖）。
