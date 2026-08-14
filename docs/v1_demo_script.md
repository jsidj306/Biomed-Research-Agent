# V1.0 内部演示脚本（5–8 分钟）

> 用途：第3-4周 V1.0 交付演示。对比 MVP 的单链，重点展示**多 Agent 协作**
> （Supervisor 派单 → Search 并发 → Review 评分 → Summarize 输出）。
> 演示的核心是「编排」的价值，而不是「搜索」本身。

**演示前检查清单**

- [ ] `uv run pytest tests/ -q -k "not slow"` 全绿（146+ tests）
- [ ] `uv run python -c "from src.agents.graph import build_graph; app = build_graph(); print(app.get_graph().draw_mermaid())"` 能打印流程图
- [ ] `.env` 中 `NCBI_API_KEY` / `OPENAI_API_KEY` 有效
- [ ] 浏览器无痕窗口，网络稳定
- [ ] 备用问题备好（见下）

**备用搜索问题**

1. "CAR-T cell therapy for solid tumors"
2. "mRNA vaccine delivery systems"
3. "single-cell RNA sequencing cancer heterogeneity"

---

## 场景 1：多 Agent 流程图（1 分钟）

1. 打印 LangGraph 状态图的 Mermaid 流程图：

   ```python
   from src.agents.graph import build_graph
   app = build_graph()
   print(app.get_graph().draw_mermaid())
   ```

2. 指出 4 个节点（supervisor / search / review / summarize）和条件边。

**台词要点**：对比 MVP 的固定流水线，V1.0 是一张「有状态、可条件路由」的图——Supervisor 拆完计划后，根据是否需要检索，决定下一步去哪。这就是从「流水线」到「编排」的升级。

---

## 场景 2：Supervisor 意图解析 + 计划生成（1 分钟）

```python
from src.agents.supervisor import SupervisorAgent
sup = SupervisorAgent()
print(sup.classify_intent("Find CAR-T trials for solid tumors"))   # search
print(sup.generate_plan("Find CAR-T trials for solid tumors"))     # [步骤1, 步骤2, ...]
```

**台词要点**：Supervisor 是「项目经理」——它不搜文献，只判断意图、拆解步骤。规划与执行分离，让每个 Agent 职责单一。

---

## 场景 3：5 数据源并发搜索（1.5 分钟）

```python
from src.agents.search_agent import SearchAgent
agent = SearchAgent()
papers = agent.search_all("CAR-T therapy solid tumors", max_results=5)
print(f"{len(papers)} papers (deduped)")
```

**台词要点**：5 个数据源（PubMed / Arxiv / Semantic Scholar / bioRxiv / medRxiv）用 asyncio 并发检索，总耗时 ≈ 最慢的单源，不是 5 源之和。每篇带 source 标记，重复论文按 DOI/标题去重。

---

## 场景 4：Review 评分 + 去重 + 矛盾检测（1 分钟）

```python
from src.agents.review_agent import ReviewAgent
reviewer = ReviewAgent()
reviews = reviewer.review(papers, "CAR-T solid tumor", top_k=10)
for r in reviews:
    print(r["score"], r["title"], "⚠️" if r.get("conflict") else "")
```

**台词要点**：Review Agent 从几十篇里挑出最相关的 Top-10，LLM 打分（0-10）+ 三级去重 + 主动标注「结论相悖」的论文对。这是研究的「安全网」。

---

## 场景 5：5 种摘要模式（1.5 分钟）

```python
from src.agents.summarizer import PaperSummarizer
s = PaperSummarizer()
s.compare(papers[:3])        # 横向对比表
s.gap_analysis(papers[:5])   # 研究空白分析
```

**台词要点**：读单篇用 Brief/Standard/Detailed，读多篇用 Comparison，找选题用 Gap Analysis——覆盖「读单篇 / 读多篇 / 找空白」三大需求。

---

## 场景 6：混合检索 + LangFuse（1 分钟，技术彩蛋）

1. 展示混合检索：`BM25 + 语义 + RRF 融合 + BGE-Reranker`，评测指标 MRR / Recall@10。
2. 展示 LangFuse 追踪：每个节点的输入/输出/耗时/成本，火焰图。

**台词要点**：检索是「粗排 + 精排」两段式，LangFuse 让多 Agent 系统可观测——出问题一眼定位是哪个节点慢。

---

## 场景 7：收尾（30 秒）

1. 展示测试结果 + 覆盖率 + CI。
2. 展示项目结构：`src/{tools,rag,agents,observability,utils}` 分层清晰。

**收尾金句**：V1.0 把一条流水线升级成一个有主管的 Agent 团队——Supervisor 派单、Search 并发、Review 评审、Summarize 输出，还带混合检索和 LangFuse 监控。下一阶段（第5-6周）做 Critic Agent、知识图谱和报告生成。
