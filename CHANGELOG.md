# Changelog

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。所有值得记录的变化都在此文件。

## [2.0.0] - 2026-08-16

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
