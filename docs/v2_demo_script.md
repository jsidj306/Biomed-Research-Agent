# V2.0 最终演示脚本（8–10 分钟）

> 用途：第5-6周 V2.0 交付演示。对比 V1.0 的 4-Agent 线性流水线，重点展示
> **自纠错回环（Critic）**、**知识图谱（Neo4j）**、**全文解析**、**引用溯源**
> 与 **报告导出** 五大升级。核心叙事是「从『会查文献』到『会挑错、会建图、会出报告』」。

---

## 演示前检查清单（先手动确认）

- [ ] `uv run pytest -q -m "not slow"` 全绿（当前 251 passed，87% 覆盖）
- [ ] `uv run ruff check src/ tests/` 全绿
- [ ] `.env` 中 `NCBI_API_KEY` / `OPENAI_API_KEY` 有效（搜索和 LLM 会真实调用）
- [ ] （可选）本地 Neo4j 已启动（`docker-compose up neo4j -d`，或已配 `NEO4J_URI`）
- [ ] （可选）PDF 导出：本地 pandoc 已装（兜底），或 Docker 内 weasyprint 开箱即用
- [ ] 浏览器无痕窗口，网络稳定

---

## 场景与台词要点

### 场景 1：7-Agent 流程图（1 分钟）

打印 LangGraph 状态图，指出 7 个节点与**自纠错回环**。

```python
from src.agents.graph import build_graph
app = build_graph()
print(app.get_graph().draw_mermaid())
```

**台词**：V1.0 是 4 节点线性流水线；V2.0 是 7 节点、带**回环**的图——Critic 质检 Writer
产出的六段式报告后，按分数把报告打回 Writer 重写，形成「生成 → 质检 → 返工」的
self-correcting 闭环。这是从「流水线」到「会自我纠错的团队」的升级。

### 场景 2：Critic 交叉验证 + 自纠错（2 分钟，核心）

演示「编造引用必被抓」：给一段含 `[99]` 的报告，看 Critic 确定性引用核查 + 两模型
交叉验证，评分被压到不及格，图打回重写。

```python
from src.agents.critic_agent import CriticAgent
c = CriticAgent()
print(c.check_citations("结论 [99] 表明……", [{"title": "A"}, {"title": "B"}]))
```

**台词**：Critic 用两个不同厂商的模型（GPT + Claude）做第二意见；引用核查是**纯正则 +
计数**，不靠 LLM——只有 2 篇文献却引用 `[99]`，必然被抓。这是「对抗幻觉」的最强保证。

### 场景 3：知识图谱（2 分钟）

抽取三元组 → 生成 Cypher → 写入 Neo4j → 展示 Mermaid。

```python
from src.knowledge_graph.extractor import KnowledgeGraphExtractor, to_mermaid
from src.knowledge_graph.neo4j_writer import Neo4jWriter

triples = KnowledgeGraphExtractor().extract_triples("CAR-T therapy treats B-cell leukemia...")
print(to_mermaid(triples))
print("\n".join(Neo4jWriter.build_cypher(triples)))  # 无需连接即可看 Cypher
```

**台词**：把「药物-靶点-疾病」抽成三元组，写入 Neo4j 用 **MERGE**（幂等）——重复写不会
产生重复节点。`build_cypher` 是纯函数，不需要真实数据库就能测试。

### 场景 4：PDF 全文解析（1 分钟）

```python
from src.rag.fulltext_loader import FulltextLoader
docs = FulltextLoader().load("path/to/paper.pdf")
print(len(docs[0].page_content), "字符")
```

**台词**：V1.0 只读摘要；V2.0 用 PyMuPDF 读整篇论文的方法/结果/讨论。扫描版 PDF 会明确
报错提示 OCR，不静默产生垃圾数据。

### 场景 5：引用溯源 + 报告导出（1.5 分钟）

```python
result = app.invoke({"query": "CAR-T therapy for solid tumors"})
report = result["report"]            # 六段式报告，含 [n] 引用
directions = result["directions"]    # 研究方向 + 假设

from src.report.export import export_report
export_report(report, "out/report.md", fmt="markdown")
export_report(report, "out/report.docx", fmt="docx")
export_report(report, "out/report.pdf", fmt="pdf")
```

**台词**：Writer 组装六段式报告（摘要/方法/发现/空白/方向/参考文献），每个结论带 `[n]`
引用；Planner 把研究空白转成可验证假设 + 可行性/创新性评分。一次调用导出
Markdown/PDF/Word。

### 场景 6：Docker 一键部署（30 秒，收尾）

**台词**：`docker-compose up -d` 一条命令拉起三服务（app + ChromaDB + Neo4j），数据卷
持久化。这是「从演示到可部署」的最后一公里。

### 收尾金句

V2.0 把「会查文献的 Agent」升级成「会挑错、会建图、会出报告的研究助手」——7-Agent
编排 + Critic 自纠错 + 知识图谱 + 全文解析 + 引用溯源 + 报告导出，全链路可观测、可部署。

---

## 录屏保存（手动执行）

- 录屏工具：OBS Studio 或 `Win + G`（Xbox Game Bar）
- 保存路径：`docs/demo/v2_demo.mp4`（目录不存在则新建）
- 截图：`docs/screenshots/`（流程图 / Critic 评分 / 知识图谱 / 报告导出 / Docker / 覆盖率）
- **提交前压缩**：`ffmpeg -i v2_demo.mp4 -crf 28 -preset veryslow v2_demo_compressed.mp4`
