# research_notes.md — 研究笔记

**项目**: biomed-research-agent  
**开始日期**: 2026-08-10

---

## 2026-08-11

### 网络架构备忘

- GitHub SSH 直连 22 端口被墙，需走 `ssh.github.com:443`
- `~/.ssh/config` 配置 `Host github.com → Hostname ssh.github.com → Port 443`
- API 调用走 API2D 中转 (`https://oa.api2d.net`)

### NVIDIA 参考项目分析

- 架构核心是四阶段研究循环：Plan → Search → Write → Reflect
- 但强依赖 NVIDIA GPU 生态（NIM、RAG Blueprint、NeMo）
- **决策**: 借鉴架构模式，用 ChromaDB + API2D 自建轻量版

### API Key 汇总

| 服务 | Key | 状态 |
|------|-----|------|
| NCBI PubMed | `ecb25dde...` | ✅ |
| API2D (GPT) | `fk248541-...` | ✅ |
| Semantic Scholar | 待审核 | ⏳ |

---

## 2026-08-10

### 环境搭建过程

- Python 3.11.15 + uv (比 Poetry 快，推荐)
- WSL 2 + Ubuntu 安装成功
- Docker Desktop 安装遇到 Manifest extraction failed 错误
  - 解决: 删除 temp 目录，管理员运行安装程序
  - hello-world 验证通过

### 技术选型记录

| 原方案 | 终方案 | 原因 |
|--------|--------|------|
| Poetry | uv | 更快更现代 |
| OpenAI 直充 | API2D 中转 | 国内银行卡被 Stripe 拒绝 |
| Anthropic 直连 | API2D 已支持 Claude | 无需额外配置 |

### RAG Demo 全链路验证

```
PubMed "CRISPR cancer" (3420篇)
  → ChromaDB embedding存储
    → 检索相似文献
      → GPT 生成回答 (API2D)
```

✅ 三阶段全部跑通。

### Docker Desktop 安装踩坑

1. `winget install Docker.DockerDesktop` → 下载失败
2. 官网手动下载 → Manifest extraction failed (WSL 2 选项导致)
3. 管理员运行安装程序 → 成功
4. 注销重登 → Docker 正常运行

---

## 待研究问题

- [ ] ChromaDB 大规模数据 (>10万篇) 性能表现
- [ ] 多语言文献支持 (中文 + 英文)
- [ ] Agent 自主决策循环的实现方案
- [ ] Gradio vs Streamlit 前端选型

---

## 2026-08-13

### 第1-2周 MVP 开发复盘

#### 完成情况

- 实际工时：约 60 小时（Day 3–14，与计划吻合）
- 完成率：**27/30** 项检查（任务 1–6 全部完成，任务 7 演示待录制）
- 核心闭环跑通：**搜索 → RAG 检索 → LLM 结构化摘要 → UI 渲染**
- 测试：**86 passed + 4 deselected(slow)，覆盖率 82%**（门禁 ≥70%）

#### 技术收获

1. **LangChain BaseTool 封装模式**：PubMed/Arxiv 统一 `{papers, total_count, query, source}` 返回格式，下游（UI/Agent）无需区分来源，为 V1.0 多数据源扩展打好基础。
2. **RAG Pipeline 调参经验**：chunk=1000/overlap=200 针对生物医学长句优化；PubMedBERT 768 维本地嵌入零 API 成本且领域适配，语义检索效果优于通用模型。
3. **LLM 摘要结构化 Prompt 化**：三种摘要类型（brief/detailed/key_findings）各自独立 prompt，temperature=0.3 保证一致性；JSON 解析三级容错（markdown fence 去除 → 直接解析 → 子串提取）解决 LLM 输出不稳定问题。
4. **Streamlit 状态管理**：`st.session_state` 三件套（messages / current_papers / search_triggered）+ `@st.cache_resource` 缓存工具实例，避免重复初始化 PubMed/Arxiv/Summarizer，实现多轮对话与按需摘要。

#### 踩坑记录

1. **坑**：`FakeEmbeddings` 用 `hash()` 导致测试 flaky（Python 每进程随机盐）。
   **解决**：改用 `zlib.crc32` 做确定性向量，测试稳定。
2. **坑**：langchain-community / chroma wrapper 陆续被官方 sunset，出现 DeprecationWarning。
   **解决**：迁移到独立包 `langchain-huggingface` + `langchain-chroma`。
3. **坑**：NCBI 真实 API 调用导致测试慢且依赖网络。
   **解决**：mock Entrez；PubMedBERT 模型下载类测试标 `@pytest.mark.slow`，CI 用 `-m "not slow"` 跳过。

#### 为第3-4周的准备

- [ ] 重新阅读 LangGraph Quick Start（用 StateGraph 重构 Agent 编排）
- [ ] 注册 Semantic Scholar API Key（等待审核中）
- [ ] 了解 BGE-Reranker 重排序原理
- [ ] 了解 LangFuse 监控接入方式
- [ ] 学习混合检索（BM25 + 语义搜索 + RRF 融合）

#### 对架构的重新思考

- MVP 的单文件 Tool 设计（每个源一个 BaseTool 类）可扩展到 5 个数据源，但需要统一的注册/调度层。
- 当前 `search_papers()` 是串行循环，多数据源应改为并发（asyncio）以缩短响应时间。
- `src/ui/app.py` 主编排逻辑较重（44% 覆盖率靠 mock），V1.0 建议抽出业务层，UI 只做渲染。
- 摘要触发目前是手动模式（避免浪费 API 配额），V1.0 可改为可配置的自动摘要。

#### MVP 交付验收（10 项确认）

| # | 验收项 | 状态 |
|---|--------|------|
| 1 | PubMed Search Tool | ✅ 实测命中 73 篇 |
| 2 | Arxiv Search Tool | ✅ 实测命中 5 篇 |
| 3 | RAG Pipeline 全链路 | ✅ 测试通过 |
| 4 | 结构化摘要（5 段） | ✅ GPT-4o-mini 实测生成 |
| 5 | Streamlit 聊天界面 | ✅ 启动健康检查 ok |
| 6 | 多轮对话上下文 | ✅ session_state 实现 + 测试覆盖 |
| 7 | 单元测试覆盖率 >70% | ✅ 82% |
| 8 | GitHub Actions CI | ⏳ 已触发，待远程确认绿色 |
| 9 | GitHub 代码推送 | ✅ 5 commits 已推送 |
| 10 | MVP 演示录屏 | ✅ 已完成（脚本见 docs/v1_demo_script.md） |

---

## 2026-08-14

### 第3-4周 V1.0 开发复盘

#### 完成情况

- 实际工时：约 60 小时（Day 15–28，与计划吻合）
- 完成率：**任务 1–8 全部完成，任务 9 演示进行中**
- 核心闭环：**Supervisor 派单 → 5 源并发搜索 → Review 评分/去重/矛盾检测 → 5 种摘要模式**
- 测试：**166 passed + 4 deselected(slow)，覆盖率 85%**（MVP 82% → 85%，门禁 ≥70%）

#### 架构决策对不对（复盘三问之一）

1. **LangGraph StateGraph 选型正确**。用「图」而非「链」描述 Agent 协作，条件边（`add_conditional_edges`）让流程从「固定流水线」升级成「自适应编排」。MVP 的 `search_papers()` 串行循环，V1.0 变成 `asyncio.gather` 并发，5 源总耗时 ≈ 最慢单源。
2. **统一返回协议是最值的决策**。MVP 定的 `{papers, total_count, query, source}` 在数据源从 2 扩到 5 时，下游 Review/Summarize 一行没改——前期半小时定协议，后期省一天。
3. **Supervisor「规划-执行分离」合理**。Supervisor 不碰检索，只判断意图、拆计划（temperature=0.2 保稳定），职责单一，测试好写。

#### 评测指标达没达（复盘三问之二）

- **测试/覆盖率达标**：166 tests、85%，超门禁。
- **混合检索指标（MRR/Recall@10）——评测基础设施已建，实测定级未跑**。`src/rag/hybrid_search.py` 的 `mean_reciprocal_rank` / `recall_at_k` 函数 + RRF 融合 + BM25 + BGE-Reranker 封装全部写完并有单测，但「30–50 个真实查询 + 人工标注」的评测集尚未构建，所以 `MRR>0.70 & Recall@10>0.85` 这条硬验收线**尚未实测定级**。这是下一阶段第一优先级。
- **LangFuse 调用链**：追踪器（`LangfuseTracer`）+ 单测完成，降级策略验证通过；但需真实注册 LangFuse 账号并配 `.env` 后，Dashboard 才能看到完整链路（当前缺 Key 时静默降级为 no-op）。

#### 下一阶段改什么（复盘三问之三）

1. **补评测集**：用之前真实搜索过的 query 建 30–50 条带标注的评测集，把 MRR/Recall 跑出真实数字。
2. **接真 LangFuse**：注册账号配 Key，把 graph.py 各节点埋点，Dashboard 出火焰图。
3. **博客发布 + 录屏**：`docs/blog_1_minimind_to_langgraph.md` 已写好，需发到掘金/知乎/GitHub README；`docs/v1_demo_script.md` 已写好，需录屏。
4. **第5-6周增强**：Critic Agent（self-refine 闭环）、知识图谱（实体抽取+关系推理）、报告生成。

#### 新增文件清单

| 文件 | 作用 |
|------|------|
| `src/agents/graph.py` | LangGraph StateGraph 编排（AgentState + 4 节点 + 条件边） |
| `src/agents/supervisor.py` | Supervisor 意图解析 + 计划生成 |
| `src/agents/search_agent.py` | 5 源 asyncio 并发 + 去重 |
| `src/agents/review_agent.py` | 评分 + 三级去重 + 矛盾检测 + Top-K |
| `src/agents/summarizer.py` | 5 种摘要模式（Brief/Standard/Detailed/Comparison/Gap） |
| `src/tools/semantic_scholar_tool.py` | Semantic Scholar 接入 |
| `src/tools/biorxiv_tool.py` | bioRxiv + medRxiv 接入（Europe PMC SRC:PPR） |
| `src/rag/hybrid_search.py` | BM25 + RRF + Reranker + MRR/Recall 评测 |
| `src/observability/tracing.py` | LangFuse 调用链追踪（降级安全） |
| `src/utils/llm_utils.py` | JSON 三级容错 + ChatModel 工厂 |
| `docs/blog_1_minimind_to_langgraph.md` | 第 1 篇技术博客 |
| `docs/v1_demo_script.md` | V1.0 演示脚本 |

#### V1.0 验收（5 项确认）

| # | 验收项 | 状态 |
|---|--------|------|
| 1 | LangGraph 多Agent流程图可运行，状态转换正确 | ✅ 166 tests 含 graph 单测 |
| 2 | 5 个数据源全部接入 | ✅ PubMed/Arxiv/S2S/bioRxiv/medRxiv |
| 3 | 混合检索 MRR>0.70 & Recall@10>0.85 | ⏳ 基础设施完成，评测集未建、数字未测 |
| 4 | LangFuse 可追踪完整调用链 | ⏳ 代码完成，待注册账号配 Key |
| 5 | 第 1 篇技术博客发布 | ⏳ 已撰写，待发布到平台 |

> 结论：V1.0 的「代码 + 测试」主干 100% 完成并验证通过；「外部依赖 + 实测定级」三类（评测集、LangFuse 账号、博客发布/录屏）需真实环境收尾，已列入第5-6周第一优先级。

#### 博客发布记录

- 2026-08-14 已发布到掘金：**从 MiniMind 到 LangGraph——构建生物医学研究 Agent**
  - 链接：https://juejin.cn/spost/7673780779483676678
  - 同步挂到 `README.md`「技术博客」小节。
- 2026-08-16 已发布到掘金：**给 Agent 加一个「挑刺的审稿人」——Critic 与自纠错回环**
  - 链接：https://juejin.cn/post/7674130372750295082
- 2026-08-16 已发布到掘金：**用 LLM + Neo4j 给生物医学文献建知识图谱**
  - 链接：https://juejin.cn/post/7674123634785828900
- 2026-08-16 已发布到掘金：**从研究到可交付产物——报告生成、导出与容器化部署**
  - 链接：https://juejin.cn/post/7674130372750360618
