# MVP 内部演示脚本（5–8 分钟）

> 用途：第1-2周 MVP 交付演示。按此脚本走完 6 个场景，可完整展示
> 「搜索 → 论文列表 → 结构化摘要」核心闭环，以及测试/CI 质量保障。

**演示前检查清单**

- [ ] `uv run streamlit run src/ui/app.py` 启动正常（默认 http://localhost:8501）
- [ ] `.env` 中 `NCBI_API_KEY` / `OPENAI_API_KEY` 有效（搜索与摘要不掉链子）
- [ ] 浏览器无缓存干扰（建议无痕窗口）
- [ ] 网络稳定
- [ ] 备用问题已备好（若实时搜索慢，改用预录截图）

**备用搜索问题**（演示时若主问题太慢可换用）

1. "CRISPR gene editing for sickle cell disease"
2. "mRNA vaccine delivery systems"
3. "single-cell RNA sequencing cancer heterogeneity"

---

## 场景 1：启动（30 秒）

1. 终端运行 `uv run streamlit run src/ui/app.py`，展示启动日志。
2. 浏览器打开 http://localhost:8501。
3. 指出界面三要素：
   - 左侧 **⚙️ Configuration 侧边栏**（搜索源 / 结果数 / 摘要类型 / 语言 / 模型）
   - 顶部标题 **🧬 Biomedical Research Agent**
   - 底部聊天输入框

**台词要点**：这是一个对话式生物医学文献研究助手，用 Streamlit 搭建，侧边栏可配置搜索源与摘要类型。

---

## 场景 2：基础搜索（1.5 分钟）

1. 输入主演示问题：

   > What are the latest advances in CAR-T cell therapy for solid tumors?

2. 展示返回结果（**实测：PubMed 命中 73 篇，Arxiv 命中 5 篇**）。
3. 强调论文卡片的来源标识：
   - 🔵 **PubMed**（含 PMID、期刊、作者）
   - 🟢 **Arxiv**（含 arxiv id、分类）
4. 展开第 1 篇论文的 **📄 Abstract** 折叠区，展示摘要内容。

**台词要点**：一次输入同时检索 PubMed 和 Arxiv 两个源，返回统一格式的论文列表；每篇卡片标明来源、作者、期刊/年份，摘要默认折叠可按需展开。

---

## 场景 3：论文结构化摘要（1.5 分钟）

1. 在下方 **📝 Generate Paper Summaries** 区选择第 1 篇论文。
2. 点击 **🤖 Generate Summary**（primary 按钮）。
3. 展示 3 个标签页：
   - **💡 Brief** —— 一句话结论 + 启示
   - **📋 Detailed** —— Background / Methods / Results / Conclusions 四段式
   - **🔑 Key Findings** —— 要点列表 + Significance
4. 指出 Findings 中的**定量数据**（若摘要含具体疗效/生存率数字，重点圈出）。

**台词要点**：这是核心价值——不是把摘要原文丢给你，而是用 GPT-4o-mini 生成**结构化摘要**，按背景/方法/结果/结论拆解，方便快速判断这篇论文值不值得精读。

---

## 场景 4：多轮追问（1 分钟）

1. 输入追问：

   > Compare the first two papers in terms of clinical outcomes

2. 展示聊天历史保留上一轮搜索结果，上下文不丢失。
3. 说明：连续 3 轮（搜索 → 追问 → 对比）上下文由 `st.session_state` 管理。

**台词要点**：对话式交互，上下文通过 session state 保存，支持连续追问，不是一次性查询。

---

## 场景 5：RAG 语义检索（1.5 分钟，可选）

> 若演示时已准备 PDF，可加这段；否则口述跳过，留作「技术彩蛋」。

1. 展示 `src/rag/rag_pipeline.py` 的 `ingest_pdf()` + `search()`。
2. 说明链路：**PDF → RecursiveCharacterTextSplitter 分块 → PubMedBERT 768 维嵌入 → ChromaDB 持久化 → 语义检索**。
3. 口述：这是后续 V1.0 多 Agent 的检索底座，MVP 阶段已跑通（`notebooks/02_rag_pipeline_demo.py`）。

**台词要点**（技术彩蛋）：检索不是关键词匹配，而是把文本变成 768 维向量做语义检索，用领域模型 PubMedBERT，比通用模型更懂生物医学术语。

---

## 场景 6：技术亮点收尾（1 分钟）

1. 终端展示测试结果：

   ```
   86 passed + 4 deselected(slow)  —  覆盖率 82% (门禁 ≥70%)
   ```

2. 展示 GitHub 仓库：5 个 commit 已推送，`.github/workflows/ci.yml` 在 push 时自动跑 lint + 覆盖率门禁。
3. 展示项目结构：`src/{tools,rag,agents,ui}` 分层清晰，10 个测试文件。

**台词要点**：质量保障是 MVP 的一部分——82% 覆盖率、ruff 0 报错、CI 自动化。演示的重点是「做了什么」，而不是「怎么做的」。

---

## 收尾金句

> 这是第1-2周的 MVP：搜索、RAG、结构化摘要、聊天界面四件事全部打通，且带 82% 测试覆盖率和 CI 门禁。下一阶段（第3-4周）将用 LangGraph 把它升级为多 Agent 编排。
