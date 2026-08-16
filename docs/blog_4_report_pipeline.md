# 博客 4 大纲：从研究到可交付产物——报告生成、导出与容器化部署

> 定位：收官复盘。核心叙事：一个 Agent 系统的「最后一公里」不是模型更强，
> 而是「把结果变成别人能用的东西」——结构化报告、多格式导出、一键部署。

---

## 一、报告生成：从「一堆摘要」到「一份综述」

- Writer Agent：把 State 里的 plan / papers / summaries / gaps / directions 组装成
  **六段式报告**（Abstract / Methods / Findings / Gaps / Directions / References）。
- 关键点：每个结论带 `[n]` 引用（与 Critic 的引用核查闭环呼应）。
- BibTeX：`build_bibtex` 让报告能直接进论文引用管理器。

## 二、引用溯源：让「结论」能点到「原文」

- UI 组件 `render_report`：把报告里的 `[n]` 自动链接到对应论文的 PMID。
- 意义：报告不再是一堵文字墙，而是可追溯的证据链。

## 三、多格式导出：Markdown / PDF / Word

- Markdown 是「中间格式」，零成本；Word 用 python-docx 保留标题层级；PDF 用
  weasyprint（中文友好）→ pandoc 兜底。
- 坑：中文 PDF 需要中文字体（Docker 镜像内置 fonts-noto-cjk，本地要装 weasyprint）。

## 四、容器化：docker-compose 三服务编排

- 多阶段 Dockerfile：依赖层 / 源码层分离，缓存复用，构建更快。
- `docker-compose.yml`：app + chroma + neo4j，数据卷持久化。
- 坑：容器内服务间通信用**服务名**做 host（`chroma`/`neo4j`），不是 localhost。

## 五、Deploy 到 HuggingFace Spaces

- 选 Docker SDK，Secrets 面板配 API Key（别把 `.env` 提交进仓库）。
- 免费 Space 的限制：Neo4j 用 AuraDB 远程实例，PubMedBERT 冷启动慢。

## 六、收官总结
- 从 MiniMind（懂模型）→ LangGraph（懂编排）→ 评估/知识图谱/报告（懂交付）的成长线。
- 复盘 12 周：MVP → 多 Agent → 增强，178 → 251 测试、86% → 87% 覆盖。

*发布平台：掘金 / 公众号，2026-08（手动发布）*
