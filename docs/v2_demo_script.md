# V2.0 最终演示脚本（8–10 分钟）

> 用途：第5-6周 V2.0 交付演示。对比 V1.0 的 4-Agent 线性流水线，重点展示
> **自纠错回环（Critic）**、**知识图谱（Neo4j）**、**全文解析**、**引用溯源**
> 与 **报告导出** 五大升级。核心叙事是「从『会查文献』到『会挑错、会建图、会出报告』」。
>
> **一键脚本**：`notebooks/08_v2_demo.py` 已把 6 个场景串起来、自动打印分隔标题、
> 场景间按回车继续。直接跑它即可，下面的台词对应脚本输出的内容。

---

## 一、录屏前准备（一次性，约 3 分钟）

```bash
cd "D:/QLDownload/Cursor/项目1/biomed-research-agent"

# 1. 测试全绿（应显示 252 passed, 4 deselected, 0 warning）
uv run pytest -q -m "not slow"

# 2. lint 干净
uv run ruff check src/ tests/

# 3. 消中文乱码（终端 GBK 代码页 → UTF-8）
chcp 65001

# 4. 确认 API key（场景 2/5 会真实调用）
#    打开 .env，确认 OPENAI_API_KEY、NCBI_API_KEY 两行非空
```

- [ ] 场景 4 的 PDF：放一份**文字版**论文到 `data/papers/demo.pdf`（或改脚本顶部 `PDF_PATH`）；没有也不崩，会打印「跳过」提示
- [ ] 本地 PDF 导出已就绪：**pandoc 3.x + typst**（`winget install JohnMacFarlane.Pandoc Typst.Typst`）。场景 5 的 PDF 会是绿的；缺了也不崩，打印友好提示
- [ ] 浏览器无痕窗口，网络稳定

---

## 二、录屏工具（二选一）

| 工具 | 启动 | 适用 |
|------|------|------|
| **OBS Studio**（推荐） | 手动打开 | 可调分辨率、加解说音轨、暂停 |
| **Xbox Game Bar** | `Win + G` | 快速、零配置 |

建议：OBS 录「全屏」或「当前窗口」，1920×1080，30fps，麦克风开（要讲解的话）。

---

## 三、录制流程（核心，8–10 分钟）

### 开场（15 秒）

终端里先跑过 `chcp 65001`，然后一句话开场：

> 这是 Biomed-Research-Agent V2.0，把「会查文献」升级成「会挑错、会建图、会出报告」。

### 场景 1–6：跑一键脚本，按回车切场景

```bash
uv run python notebooks/08_v2_demo.py
```

脚本会**自动打印分隔标题 + 每个场景按回车继续**，按下面节奏讲解：

| # | 场景 | 时长 | 讲解重点（对着屏幕上的输出讲） |
|---|------|------|------|
| 1 | 7-Agent 流程图 | 1min | V1.0 是 4 节点线性；V2.0 是 7 节点带**回环**——Critic 质检后打回 Writer 重写 |
| 2 | Critic 抓编造引用 | 2min | 先看**确定性核查**（`[99]` 只有 2 篇文献必被抓，纯正则不靠 LLM）；再看两模型交叉验证把分压到 ≤4.0 |
| 3 | 知识图谱 | 2min | 三元组 `(drug)-[targets]->(target)` → Mermaid 可视化 → **MERGE** Cypher（幂等，不产生重复节点） |
| 4 | PDF 全文解析 | 1min | V1.0 只读摘要，V2.0 用 PyMuPDF 读整篇；扫描版会明确报错不产垃圾 |
| 5 | 全链路 + 报告导出 | 2min | 快跑预设（每源 3 篇）真实跑通；六段式报告带 `[n]` 引用 + 研究方向可行性/创新性评分；导出 Markdown/Word/PDF |
| 6 | Docker 部署 | 30s | 三服务编排（app+ChromaDB+Neo4j），`docker-compose up -d` 一条命令 |

### 各场景台词要点（脚本自动输出的内容）

#### 场景 1：7-Agent 流程图（1 分钟）

脚本打印 LangGraph 状态图（`app.get_graph().draw_mermaid()`）。

**台词**：V1.0 是 4 节点线性流水线；V2.0 是 7 节点、带**回环**的图——Critic 质检 Writer
产出的六段式报告后，按分数把报告打回 Writer 重写，形成「生成 → 质检 → 返工」的
self-correcting 闭环。这是从「流水线」到「会自我纠错的团队」的升级。

#### 场景 2：Critic 交叉验证 + 自纠错（2 分钟，核心）

演示「编造引用必被抓」：给一段含 `[99]` 的报告，看 Critic 确定性引用核查 + 两模型
交叉验证，评分被压到不及格。

**台词**：Critic 用两个不同厂商的模型（GPT + Claude）做第二意见；引用核查是**纯正则 +
计数**，不靠 LLM——只有 2 篇文献却引用 `[99]`，必然被抓。这是「对抗幻觉」的最强保证。

#### 场景 3：知识图谱（2 分钟）

抽取三元组 → 生成 Cypher → 展示 Mermaid。

**台词**：把「药物-靶点-疾病」抽成三元组，写入 Neo4j 用 **MERGE**（幂等）——重复写不会
产生重复节点。`build_cypher` 是纯函数，不需要真实数据库就能测试。

#### 场景 4：PDF 全文解析（1 分钟）

**台词**：V1.0 只读摘要；V2.0 用 PyMuPDF 读整篇论文的方法/结果/讨论。扫描版 PDF 会明确
报错提示 OCR，不静默产生垃圾数据。

#### 场景 5：引用溯源 + 报告导出（2 分钟）

快跑预设（每源 3 篇）真实跑通全链路，输出六段式报告 + 研究方向，导出三种格式。

**台词**：Writer 组装六段式报告（摘要/方法/发现/空白/方向/参考文献），每个结论带 `[n]`
引用；Planner 把研究空白转成可验证假设 + 可行性/创新性评分。一次调用导出
Markdown/Word/PDF（本地 pandoc+typst，无需 TeX 发行版）。

#### 场景 6：Docker 一键部署（30 秒，收尾）

**台词**：`docker-compose up -d` 一条命令拉起三服务（app + ChromaDB + Neo4j），数据卷
持久化。这是「从演示到可部署」的最后一公里。

### 收尾金句（脚本最后已自动打印）

> V2.0 把「会查文献的 Agent」升级成「会挑错、会建图、会出报告的研究助手」——7-Agent
> 编排 + Critic 自纠错 + 知识图谱 + 全文解析 + 引用溯源 + 报告导出，全链路可观测、可部署。

### 可选加拍：UI 界面（30 秒）

```bash
uv run streamlit run src/ui/app.py
```

录一下浏览器界面（`localhost:8501`），演示「搜索 → 摘要 → 报告」的聊天交互。

---

## 四、收尾（压缩 + 保存）

```bash
# 压缩（避免撑爆 git 仓库，见项目约定）
ffmpeg -i v2_demo.mp4 -crf 28 -preset veryslow v2_demo_compressed.mp4
```

- 录屏保存：`docs/demo/v2_demo.mp4`（压缩版）
- 截图：`docs/screenshots/`（流程图 / Critic 评分 / 知识图谱 / 报告导出 / Docker / 覆盖率）

---

## 已知可忽略的 ⚠️ 情况

1. **场景 4 没 PDF** → 打印「跳过」提示，正常。
2. **weasyprint 本地报「could not import external libraries」** → 正常（Windows 缺
   GTK/Pango），会自动兜底到 pandoc+typst，PDF 仍正常产出；这条正好可以顺势讲
   「容器内 weasyprint 开箱即用（Dockerfile 已补 Pango 原生库）」。
3. **终端中文乱码** → 跑 `chcp 65001` 即可，纯显示问题，不影响功能。
