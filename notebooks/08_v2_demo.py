"""
V2.0 演示脚本（一键跑通 6 个场景，用于录屏）。

用法:
    uv run python notebooks/08_v2_demo.py

相比 docs/v2_demo_script.md 里散落的代码片段，这个脚本把 6 个场景串起来、自动打印
分隔标题，并在场景之间暂停（按回车继续），方便边录边讲解。

关键设计:
  - 场景 5 用「快跑预设」：FastSearchAgent 限制每源 3 篇（各 Agent 默认已是 glm-5.2），
    缩短真实检索 + LLM 调用的耗时
  - 所有外部调用（LLM / PDF / 检索）都包 try/except，单点失败只降级提示、不中断录屏

注意:
  - 场景 2/5 会真实调 LLM 与 PubMed，需 .env 里 ZHIPU_API_KEY / MOONSHOT_API_KEY / NCBI_API_KEY 有效
  - 场景 4/5 需要一份文字版 PDF，把顶部 PDF_PATH 改成你的文件路径（场景 5 会读全文作为首要证据）
  - 场景 5 的 PDF 导出：本地需 pandoc 3.x + typst（winget 安装）或 Docker 内
    weasyprint，后端缺失时会打印友好提示而非崩溃
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Windows 控制台默认 GBK 代码页，print 含 ⏸/🎉 等 emoji 时会抛 UnicodeEncodeError；
# 强制 stdout/stderr 走 UTF-8，避免录屏时因终端编码崩溃（显示乱码可用 chcp 65001 解决）。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

# 让各 Agent 节点的 logger.info 进度打印到终端。默认 logging 无 handler + WARNING 级别，
# 全链路 invoke 期间会「一片寂静」，误以为卡死；这里显式打开 INFO 并降噪第三方库。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
for _noisy in ("httpx", "httpcore", "openai", "langchain", "langchain_core",
               "langchain_openai", "urllib3", "tenacity"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# 直接 `python notebooks/08_v2_demo.py` 运行时，Python 只把 notebooks/ 加入 sys.path，
# 导致 `from src...` 找不到模块；把项目根目录前置，保证任意目录下都能跑。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.agents.critic_agent import CriticAgent
from src.agents.graph import build_graph
from src.agents.search_agent import SearchAgent
from src.knowledge_graph.extractor import KnowledgeGraphExtractor, to_mermaid
from src.knowledge_graph.neo4j_writer import Neo4jWriter
from src.rag.fulltext_loader import FulltextLoader
from src.report.export import export_report

# ── 可调参数 ─────────────────────────────────────────────────────────────

QUERY = "brain-computer interface decoding algorithm"
PDF_PATH = "data/papers/demo.pdf"   # 场景 4/5 用：改成你备好的文字版 PDF 路径（场景 5 会读它的全文）

# ── 工具函数 ─────────────────────────────────────────────────────────────


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def pause() -> None:
    try:
        input("\n  ⏸  按回车继续下一场景... ")
    except EOFError:
        pass


def extract_title(text: str, fallback: str = "本地文档") -> str:
    """从 PDF 全文开头提取一个可读标题（取首个非空短行），失败回退文件名。"""
    for line in text.splitlines():
        line = line.strip()
        if line and len(line) <= 120:
            return line
    return fallback


# ── 场景 1：7-Agent 流程图 ───────────────────────────────────────────────

section("场景 1：7-Agent 流程图（含自纠错回环）")
app = build_graph()
print(app.get_graph().draw_mermaid())
pause()

# ── 场景 2：Critic 抓编造引用 ────────────────────────────────────────────

section("场景 2：Critic 抓编造引用（确定性核查 + 评分压制）")
critic = CriticAgent()
sources = [{"title": "Paper A"}, {"title": "Paper B"}]
bad_report = "结论 [99] 表明 CAR-T 对实体瘤疗效显著，参考 [1] 与 [2]。"

print("【第 1 道关】确定性引用核查（纯正则 + 计数，不调 LLM）：")
for issue in critic.check_citations(bad_report, sources):
    print(f"  ⚠️  {issue['claim']} — {issue['problem']} (severity={issue['severity']})")

print("\n【第 2+3 道关】两模型交叉验证 + 评分压制（真实调 GLM + Kimi，约 5-10 秒）：")
verdict = critic.verify(bad_report, sources)
print(f"  overall_score = {verdict['overall_score']}（编造引用 → 一票否决压到 ≤ 4.0）")
print(f"  issues 共 {len(verdict['issues'])} 条")
pause()

# ── 场景 3：知识图谱 ─────────────────────────────────────────────────────

section("场景 3：知识图谱（三元组抽取 → MERGE Cypher → Mermaid）")
text = "CAR-T therapy targets CD19 and treats B-cell leukemia in patients."
triples = KnowledgeGraphExtractor().extract_triples(text)

print(f"抽取到 {len(triples)} 个三元组：")
for t in triples:
    print(f"  ({t['head_type']}:{t['head_name']}) -[{t['relation']}]-> ({t['tail_type']}:{t['tail_name']})")

print("\nMermaid 可视化：")
print(to_mermaid(triples))

print("\nMERGE Cypher（纯函数，无需连库即可预览）：")
for stmt in Neo4jWriter.build_cypher(triples):
    print("  " + stmt)
pause()

# ── 场景 4：PDF 全文解析 ─────────────────────────────────────────────────

section("场景 4：PDF 全文解析（PyMuPDF 读全文）")
pdf = Path(PDF_PATH)
if pdf.exists():
    docs = FulltextLoader().load(str(pdf))
    print(f"{PDF_PATH} → 提取 {len(docs[0].page_content)} 字符")
    print("  前 200 字预览：", docs[0].page_content[:200].replace("\n", " "))
else:
    print(f"⚠️  未找到 {PDF_PATH}，跳过（请先放一份文字版 PDF，或改脚本顶部 PDF_PATH）")
pause()

# ── 场景 5：全链路 + 报告导出（快跑预设）────────────────────────────────

section("场景 5：全链路 invoke → 六段式报告 → 导出（快跑预设：每源 3 篇）")


class FastSearchAgent(SearchAgent):
    """演示预设：限制每源结果数，加速全链路（仍是真实检索，只是量小）。"""

    def search_all(self, query: str, max_results: int = 3) -> list:
        return super().search_all(query, max_results=max_results)


# 把本地 demo.pdf 全文接进流水线：全文作为「第 1 篇证据」，报告基于它生成。
# 读不到 PDF 就降级为纯检索模式，不中断录屏。
document_text = ""
document_title = Path(PDF_PATH).name
if Path(PDF_PATH).exists():
    try:
        docs = FulltextLoader().load(str(PDF_PATH))
        document_text = docs[0].page_content
        document_title = extract_title(document_text, fallback=Path(PDF_PATH).name)
        print(f"✅ 已读取本地文档全文：{document_title}（{len(document_text)} 字符）")
    except Exception as exc:
        print(f"⚠️  本地 PDF 读取失败（{exc}），降级为纯检索模式")
else:
    print(f"⚠️  未找到 {PDF_PATH}，降级为纯检索模式（报告仅基于检索结果）")

demo_app = build_graph(search_agent=FastSearchAgent())
print(f"开始全链路（query: {QUERY}，本地文档: {document_title}）... 真实检索 + 7 Agent，约 1-2 分钟")
result = demo_app.invoke({
    "query": QUERY,
    "document_text": document_text,
    "document_title": document_title,
})

report = result["report"]
directions = result["directions"]
print(f"\n✅ 完成：critic_score={result.get('critic_score')}，报告 {len(report)} 字符，"
      f"研究方向 {len(directions)} 条")
print("\n报告预览（前 1200 字）：")
print(report[:1200])

print("\n研究方向：")
for d in directions[:5]:
    print(f"  • {d.get('title', '')}（可行性={d.get('feasibility', '?')}，创新性={d.get('novelty', '?')}）")

print("\n导出三种格式：")
for fmt, path in [("markdown", "out/report.md"), ("docx", "out/report.docx"), ("pdf", "out/report.pdf")]:
    try:
        out = export_report(report, path, fmt=fmt)
        print(f"  ✅ {fmt:8s} → {out}")
    except Exception as exc:  # 演示脚本：单格式失败只降级提示，不中断录屏
        print(f"  ⚠️  {fmt:8s} 导出失败（{exc}）")
pause()

# ── 场景 6：Docker 一键部署 ──────────────────────────────────────────────

section("场景 6：Docker 一键部署（三服务编排）")
compose = Path(__file__).resolve().parent.parent / "docker-compose.yml"
print(compose.read_text(encoding="utf-8"))
pause()

print("\n" + "=" * 72)
print("  🎉 全部 6 个场景跑完。收尾金句：")
print("  V2.0 把「会查文献的 Agent」升级成「会挑错、会建图、会出报告的研究助手」。")
print("=" * 72)
