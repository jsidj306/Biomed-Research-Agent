# reuse_plan.md — NVIDIA Biomedical AI-Q 可复用分析

**参考项目**: NVIDIA Biomedical AI-Q Research Agent  
**目标项目**: biomed-research-agent  
**日期**: 2026-08-11

---

## 架构对比

| 维度 | NVIDIA 参考项目 | 本项目 | 可复用度 |
|------|----------------|--------|----------|
| Agent 框架 | LangGraph + NeMo Agent Toolkit | 自建 (Python) | ⚠️ 借鉴模式 |
| RAG | NVIDIA RAG Blueprint + NIM | ChromaDB | ❌ 架构不同 |
| LLM | Llama NIM 微服务 | API2D → GPT/Claude | ❌ 接口不同 |
| 部署 | Docker Compose + GPU | Windows + WSL 轻量 | ❌ 差异大 |
| 前端 | 闭源 Docker 前端 | Gradio / Streamlit | ❌ 不可用 |
| 虚拟筛选 | MolMIM + DiffDock | 暂不需要 | ❌ 跳过 |

---

## ✅ 可复用项 (5项)

### 1. 研究循环架构 (Research Loop)

**来源**: `aira/src/aiq_aira/nodes.py`

NVIDIA 的四阶段研究循环：

```
用户输入主题 → ①制定研究计划 → ②并行检索多数据源 
→ ③撰写报告草稿 → ④反思空白补全 → 生成最终报告
```

**复用方式**: 仿照这个 pattern，用我们自己的工具链重写：

```
用户输入主题 → ①LLM生成检索策略 → ②PubMed+CORD-19搜索 
→ ③ChromaDB存储+检索 → ④GPT生成报告 → ⑤自检空白循环
```

### 2. LLM-as-Judge 相关性判断

**来源**: `aira/src/aiq_aira/search_utils.py` 的 relevancy check 逻辑

对每个检索结果，用 LLM 判断是否与查询相关，过滤噪音。

**复用方式**: 直接用 API2D + GPT 实现，prompt 模板：

```
请判断以下文献是否与查询相关：
查询：{query}
文献摘要：{abstract}
仅回答 YES 或 NO，然后给出理由。
```

### 3. 并行搜索模式

**来源**: `aira/src/aiq_aira/functions/generate_queries.py`

将一个研究主题拆解为多个子查询，并行搜索以提高效率。

**复用方式**: 
- 用 LLM 生成 3-5 个子查询
- 并行调用 PubMed API
- 合并去重结果

### 4. 报告生成结构

**来源**: `aira/src/aiq_aira/report_gen_utils.py`

参考报告结构：摘要 → 背景 → 方法 → 结果 → 讨论 → 参考文献

**复用方式**: 直接采用这个结构，设计 prompt 模板引导 GPT 按此格式输出。

### 5. Prompt 工程参考

**来源**: `aira/src/aiq_aira/prompts.py` + `artifact_prompts.py`

高质量的系统提示词设计模式，包括：
- 报告规划 prompt 结构
- 研究问题生成 prompt
- 反思空白 prompt
- 文献引用格式规范

**复用方式**: 提取 prompt 模板的核心结构，替换模型调用为 API2D 接口。

---

## ❌ 不可复用项 (3项)

### 1. NVIDIA NIM 微服务
需要 NVIDIA GPU + CUDA + Container Toolkit，个人电脑无法运行。

### 2. MolMIM / DiffDock 虚拟筛选
药物发现专用，第0周阶段不需要。

### 3. Demo 前端
闭源 Docker 镜像，不可获取源码。

---

## 📐 本项目适配后的架构

```
┌─────────────────────────────────────────────────┐
│                 用户界面 (CLI/Gradio)              │
├─────────────────────────────────────────────────┤
│              研究循环编排器 (Orchestrator)          │
│  Plan → Search → Retrieve → Write → Reflect      │
├──────────┬───────────┬───────────┬───────────────┤
│ PubMed   │ ChromaDB  │ API2D     │ LLM-as-Judge  │
│ API      │ (RAG)     │ (GPT/     │ (Relevancy    │
│          │           │  Claude)  │  Check)       │
└──────────┴───────────┴───────────┴───────────────┘
```

---

## 📋 复用时序

| 顺序 | 复用项 | 依赖 | 预计工时 |
|------|--------|------|----------|
| 1 | Prompt 工程参考 | 无 | 0.5h |
| 2 | 并行搜索模式 | PubMed API | 1h |
| 3 | LLM-as-Judge 相关性判断 | API2D | 0.5h |
| 4 | 研究循环架构 | 前3项 | 2h |
| 5 | 报告生成结构 | 前4项 | 1h |

---

## 🔗 参考文件路径

```text
参考项目: D:\QLDownload\Cursor\项目1\biomedical-aiq-research-agent\
├── aira/src/aiq_aira/
│   ├── nodes.py          ← 研究循环节点
│   ├── prompts.py        ← 提示词模板
│   ├── search_utils.py   ← 搜索工具 + 相关性判断
│   ├── report_gen_utils.py ← 报告生成
│   ├── schema.py         ← 数据模型
│   └── tools.py          ← 工具定义
├── notebooks/
│   └── get_started_nvidia_api.ipynb ← API 调用示例
└── demo/
    └── example_report.md ← 报告输出样例
```
