# V1.0 内部演示脚本（5–8 分钟）

> 用途：第3-4周 V1.0 交付演示。对比 MVP 的单链，重点展示**多 Agent 协作**
> （Supervisor 派单 → Search 并发 → Review 评分 → Summarize 输出）。
> 演示的核心是「编排」的价值，而不是「搜索」本身。

## 一键启动（一条命令跑完 7 个场景）

```bash
cd "D:/QLDownload/Cursor/项目1/biomed-research-agent"
uv run python notebooks/07_v1_demo.py
```

- 每个场景结束后按 **Enter** 进入下一场景，边跑边按下面台词讲解。
- 每个场景都已埋 LangFuse 点，脚本退出前自动 flush，可到后台看火焰图。
- 若想不交互自动跑完（演练用）：`printf '\n\n\n\n\n\n\n' | uv run python notebooks/07_v1_demo.py`

**演示前检查清单**

- [ ] `uv run pytest tests/ -q -k "not slow"` 全绿（当前 178 passed）
- [ ] `.env` 中 `NCBI_API_KEY` / `OPENAI_API_KEY` 有效（搜索和 LLM 会真实调用）
- [ ] 浏览器无痕窗口，网络稳定（备用问题见下）

**备用搜索问题**（如果主查询实时搜索慢，可改脚本里的 `QUERY`）

1. "CAR-T cell therapy for solid tumors"
2. "mRNA vaccine delivery systems"
3. "single-cell RNA sequencing cancer heterogeneity"

---

## 各场景台词要点（配合脚本输出讲解）

### 场景 1：多 Agent 流程图（1 分钟）

脚本打印 LangGraph 状态图的 Mermaid 流程图，指出 4 个节点（supervisor / search / review / summarize）和条件边。

**台词**：对比 MVP 的固定流水线，V1.0 是一张「有状态、可条件路由」的图——Supervisor 拆完计划后，根据是否需要检索，决定下一步去哪。这就是从「流水线」到「编排」的升级。

### 场景 2：Supervisor 意图解析 + 计划生成（1 分钟）

脚本调用 `classify_intent` + `generate_plan`，打印意图和 5 步研究计划。

**台词**：Supervisor 是「项目经理」——它不搜文献，只判断意图、拆解步骤。规划与执行分离，让每个 Agent 职责单一。

### 场景 3：5 数据源并发搜索（1.5 分钟）

脚本调用 `search_all`，打印去重后的论文数 + 前 5 篇（含 source 标记）。

**台词**：5 个数据源（PubMed / Arxiv / Semantic Scholar / bioRxiv / medRxiv）用 asyncio 并发检索，总耗时 ≈ 最慢的单源，不是 5 源之和。每篇带 source 标记，重复论文按 DOI/标题去重。

### 场景 4：Review 评分 + 去重 + 矛盾检测（1 分钟）

脚本打印 Top-10 论文的分数，相悖论文带 ⚠️ 标记。

**台词**：Review Agent 从几十篇里挑出最相关的 Top-10，LLM 打分（0-10）+ 三级去重 + 主动标注「结论相悖」的论文对。这是研究的「安全网」。

### 场景 5：5 种摘要模式（1.5 分钟）

脚本打印 `compare`（横向对比）和 `gap_analysis`（研究空白）的结构化输出。

**台词**：读单篇用 Brief/Standard/Detailed，读多篇用 Comparison，找选题用 Gap Analysis——覆盖「读单篇 / 读多篇 / 找空白」三大需求。

### 场景 6：混合检索 + LangFuse（1 分钟，技术彩蛋）

脚本打印混合检索的 MRR / Recall@10，并提示去 LangFuse 后台看火焰图。

**台词**：检索是「粗排 + 精排」两段式（BM25 + 语义 + RRF 融合 + BGE-Reranker）；LangFuse 让多 Agent 系统可观测——每个节点的输入/输出/耗时都能看到，出问题一眼定位是哪个节点慢。

### 场景 7：收尾（30 秒）

脚本打印项目结构、测试结果、下一阶段方向。

**收尾金句**：V1.0 把一条流水线升级成一个有主管的 Agent 团队——Supervisor 派单、Search 并发、Review 评审、Summarize 输出，还带混合检索和 LangFuse 监控。下一阶段（第5-6周）做 Critic Agent、知识图谱和报告生成。

---

## 录屏保存

- 录屏工具：`Win + G`（Xbox Game Bar）或 OBS Studio
- 保存路径：`docs/demo/v1_demo.mp4`（目录不存在则新建）
- 截图：`docs/screenshots/`（主界面 / 流程图 / 搜索结果 / 摘要 / 覆盖率 / LangFuse 火焰图）
