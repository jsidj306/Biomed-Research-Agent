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
| 10 | MVP 演示录屏 | ⏳ 待录制（脚本见 docs/demo_script.md） |
