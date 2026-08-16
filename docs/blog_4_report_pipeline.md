# 从研究到可交付产物——报告生成、导出与容器化部署

> 这是「Biomed-Research-Agent」项目的第四篇、也是收官篇技术复盘。
> 前面三篇讲了「多 Agent 编排」「Critic 自纠错」「知识图谱」，但一个 Agent 系统真正的
> 「最后一公里」，不是模型更强，而是**把结果变成别人能用的东西**。
> 这一篇，就讲这条最后一公里：结构化报告、多格式导出、一键部署。

---

## 一、报告生成：从「一堆摘要」到「一份综述」

前几轮的产物是散的：`plan`（研究步骤）、`papers`（检索结果）、`reviews`（相关性评分）、`summaries`（摘要）、`gaps`（研究空白）。这些零散的数据，用户是没法直接读的。

于是有了 **Writer Agent**，它把这些东西组装成一篇 **IMRaD 变体的六段式报告**：

```
Abstract / Methods / Findings / Gaps / Directions / References
```

关键设计有两点。

**第一，分节生成。** 报告很长，一次生成容易超 token 或前后不一致，所以按节分别调 LLM 再拼装：

```python
REPORT_SECTIONS = ["Abstract", "Methods", "Findings", "Gaps", "Directions", "References"]

sections = {}
for section in REPORT_SECTIONS[:-1]:   # References 最后统一代码生成
    sections[section] = self._generate_section(section, state)
sections["References"] = self.build_references(papers)
```

**第二，References 用代码拼接，不让 LLM 写。** 引用是幻觉的重灾区，所以参考文献列表干脆不经过 LLM，直接用论文元数据模板化拼接：

```python
CITATION_FMT = "[{i}] {title}. {authors}. {journal} ({year}). PMID:{pmid}"
```

正文里 LLM 用 `[n]` 标记引用，`[n]` 与 References 的编号一一对应。因为 References 是代码生成的，所以「引用 100% 真实」——这也正是上一篇 Critic 能做「确定性引用核查」的前提：正文里的 `[n]` 超出论文总数，必然是被编造的。

顺带一提，`build_bibtex` 还能把论文列表转成 BibTeX，报告里的引用能直接进 Zotero / EndNote。

---

## 二、引用溯源：让「结论」能点到「原文」

一篇报告如果只是文字，`[n]` 就只是个数字。但如果 `[n]` 能点一下跳转到 PubMed 原文，报告就从「一堵文字墙」变成「一条可追溯的证据链」。

这靠 UI 层一个纯函数实现——把报告里的 `[n]` 替换成可点击的 PubMed 链接：

```python
def _linkify_citations(report, papers):
    def _repl(match):
        n = int(match.group(1))
        if 1 <= n <= len(papers):
            pmid = str(papers[n - 1].get("pmid", "")).strip()
            if pmid:
                return f"[{n}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)"
        return match.group(0)   # 对应不上就原样保留，绝不瞎链
    return re.sub(r"\[(\d+)\]", _repl, report)
```

注意那个兜底分支：如果 `[n]` 对应的论文没有 PMID、或者编号越界，就**原样保留**，绝不链到一个错误的地方。溯源和引用核查一样，宁可「没链上」，不能「链错」。

---

## 三、多格式导出：Markdown / PDF / Word

Writer 产出的报告本身就是 Markdown，导出其实是「把 Markdown 转成别的格式」。三种格式，按成本从低到高：

| 格式 | 后端 | 说明 |
|------|------|------|
| Markdown | 无（直接写文件） | 零成本，也是 PDF/Word 的中间格式 |
| Word | python-docx | 保留标题层级，延迟导入 |
| PDF | weasyprint → pandoc 兜底 | 首选中文友好的 weasyprint |

PDF 这条链路最折腾，所以做了两级降级：

```python
def export_pdf(report, path):
    html = markdown_to_html(report)
    # 首选 weasyprint（中文支持好，纯 Python）
    try:
        from weasyprint import HTML
        HTML(string=html).write_pdf(str(path))
        return str(Path(path).resolve())
    except ImportError:
        pass
    # 兜底 pandoc（本地 CLI）
    subprocess.run(["pandoc", "-f", "markdown", "-t", "pdf", "-o", str(path)],
                   input=report.encode("utf-8"), check=True, capture_output=True)
    ...
```

两个后端都缺失时，抛一个**明确的 RuntimeError**告诉你装哪个，而不是神秘崩溃。

> 踩坑记录：weasyprint 是纯 Python 包，但它底层依赖 GTK/Pango 这些**系统原生库**。Windows 本地 `uv add weasyprint` 只是装了 Python 包，原生库得另外装——这也是为什么我在 Docker 镜像里内置了中文字体（fonts-noto-cjk），容器里 PDF 导出开箱即用，本地则用 pandoc 兜底。

---

## 四、容器化：docker-compose 三服务编排

一个 Agent 项目依赖一堆东西：Streamlit UI、ChromaDB 向量库、Neo4j 图库。手动一个个装，换台机器就崩。所以把它容器化，用一条命令拉起整个环境。

**多阶段 Dockerfile**——依赖层和源码层分离，源码改动不触发依赖重装：

```dockerfile
# ---- 依赖层（先拷贝清单，充分利用 Docker 缓存）----
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev

# ---- 源码层 ----
COPY . .

# 中文字体（PDF 导出用，避免中文变方框）
RUN apt-get update && apt-get install -y --no-install-recommends fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

CMD ["uv", "run", "streamlit", "run", "src/ui/app.py",
     "--server.address=0.0.0.0", "--server.headless=true"]
```

**docker-compose 三服务**——app + chroma + neo4j 一起编排：

```yaml
services:
  app:
    build: .
    ports: ["8501:8501"]
    env_file: [".env"]
    depends_on: [chroma, neo4j]
    environment:
      CHROMA_HOST: chroma       # 容器内用服务名，不是 localhost
      NEO4J_URI: bolt://neo4j:7687
  chroma:
    image: chromadb/chroma
    ports: ["8000:8000"]
  neo4j:
    image: neo4j:5
    ports: ["7474:7474", "7687:7687"]
    environment: ["NEO4J_AUTH=neo4j/password"]
```

> 踩坑记录：**容器内服务间通信用服务名做 host（`chroma` / `neo4j`），不是 `localhost`。** 这是容器网络和本机网络最根本的区别，写错这个，`app` 就永远连不上向量库和图库。compose 文件里已经用 `CHROMA_HOST` / `NEO4J_URI` 环境变量注入了正确地址，代码里读环境变量而不是硬编码 `localhost`。

数据持久化靠两个命名卷 `chroma_data` / `neo4j_data`：`docker-compose down` 不删卷，`down -v` 才会清空。

---

## 五、部署落地：本地 Docker（附一个云端收费变更）

原计划把 demo 免费部署到 HuggingFace Spaces 上，但 2026 年 7 月起，HF 取消了免费层：

> ⚠️ HF Spaces 的 Docker / Streamlit SDK 现在都要求 **PRO 订阅（$9/月）**，免费层只剩静态站点和限制极严的 ZeroGPU。本项目是实时 Streamlit 应用、又依赖 Dockerfile（中文字体 + uv），**免费部署已不可行**。

所以最终的交付策略是**本地 Docker**——`docker-compose up -d` 一条命令拉起完整环境，录屏演示也完全可以展示给面试官/同事看，而且不占用云端资源、不受第三方收费变更影响。

这件事本身也是个心得：**依赖第三方平台做「免费部署」，等于把交付的可控性交到别人手里。** 本地容器化才是真正属于自己的、可复现的交付产物。

---

## 六、收官总结

回头看这 12 周，一条清晰的成长线：

1. **MiniMind** —— 手搓小模型，搞懂「模型」是怎么训练出来的；
2. **LangGraph 多 Agent** —— 学会「编排」，让多个模型像团队一样协作；
3. **Critic / 知识图谱 / 报告导出 / 容器化** —— 学会「交付」，把 Agent 的能力变成别人能用的东西。

从「懂模型」到「懂编排」再到「懂交付」，每一步都在补一块短板。

数据上也有一条线：测试从 MVP 的 **86 个**（82% 覆盖）一路涨到 **251 个**（**87% 覆盖**），ruff 全绿。而最难的不是写出这些代码，是**始终保持每个 Agent 都能被单独 mock 测试**——这是多 Agent 系统不失控的底线。

这个项目到这里就告一段落了。但「评估」——怎么证明一个 Agent 真的比另一个好——还留着一个更大的坑，留给下一个项目。

---

*项目地址：[github.com/jsidj306/Biomed-Research-Agent](https://github.com/jsidj306/Biomed-Research-Agent)*
*2026-08*
