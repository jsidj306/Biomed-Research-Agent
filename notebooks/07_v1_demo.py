"""
V1.0 演示 — 一条命令跑完 7 个场景（替代逐段敲 REPL）。

用途：第3-4周 V1.0 交付演示。按 Enter 逐场景推进，边跑边按台词讲解。
每个场景都埋了 LangFuse 点，退出前统一 flush。

运行:
    uv run python notebooks/07_v1_demo.py

说明:
    - 每场景结束后按 Enter 进入下一场景（便于录屏时控制节奏）；
      若不想交互，可用 `printf '\\n\\n\\n\\n\\n\\n\\n' | uv run python notebooks/07_v1_demo.py` 自动跑完。
    - 场景 2-5 会真实调用 LLM / 5 个数据源（消耗 API 配额，搜索约 10-30 秒）。
    - 任一数据源失败会降级跳过，不影响整体演示。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

QUERY = "CAR-T cell therapy for solid tumors"


def hr(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def pause() -> None:
    try:
        input("\n  [Enter] 进入下一场景 … ")
    except EOFError:
        pass


def dump(obj, limit: int = 1200) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    print(text[:limit] + ("\n  …(截断)" if len(text) > limit else ""))


def main() -> None:
    from src.observability.tracing import LangfuseTracer

    tracer = LangfuseTracer()

    print("=" * 70)
    print("  V1.0 演示 — 多 Agent 编排（Supervisor → Search → Review → Summarize）")
    print("=" * 70)
    print(f"  查询: {QUERY}")
    print(f"  LangFuse 追踪: {'✅ 已启用' if tracer.enabled else '🔶 未启用（缺 Key，降级）'}")
    pause()

    # ── 场景 1：流程图 ──────────────────────────────────────────────
    hr("场景 1：多 Agent 流程图（LangGraph 状态图）")
    from src.agents.graph import build_graph

    app = build_graph(tracer=tracer)
    print(app.get_graph().draw_mermaid())
    print("\n  ▶ 4 个节点 + 条件边：supervisor 拆完计划后，决定去 search 还是直接 summarize。")
    pause()

    # ── 场景 2：Supervisor ──────────────────────────────────────────
    hr("场景 2：Supervisor 意图解析 + 计划生成")
    from src.agents.supervisor import SupervisorAgent

    sup = SupervisorAgent()
    with tracer.trace("supervisor", {"query": QUERY}):
        intent = sup.classify_intent(QUERY)
        plan = sup.generate_plan(QUERY)
    print(f"  意图: {intent}")
    print("  计划:")
    for i, step in enumerate(plan, 1):
        print(f"    {i}. {step}")
    print("\n  ▶ Supervisor 只规划不执行：规划与执行分离，Agent 职责单一。")
    pause()

    # ── 场景 3：Search ─────────────────────────────────────────────
    hr("场景 3：5 数据源并发搜索")
    from src.agents.search_agent import SearchAgent

    agent = SearchAgent()
    with tracer.trace("search", {"query": QUERY}):
        papers = agent.search_all(QUERY, max_results=5)
    print(f"  检索到 {len(papers)} 篇（去重后）")
    for p in papers[:5]:
        title = (p.get("title") or "")[:64]
        print(f"    - [{p.get('source', '?')}] {title}")
    print("\n  ▶ PubMed / Arxiv / Semantic Scholar / bioRxiv / medRxiv 并发，耗时≈最慢单源。")
    pause()

    # ── 场景 4：Review ─────────────────────────────────────────────
    hr("场景 4：Review 评分 + 去重 + 矛盾检测")
    from src.agents.review_agent import ReviewAgent

    reviewer = ReviewAgent()
    with tracer.trace("review", {"query": QUERY, "n_papers": len(papers)}):
        reviews = reviewer.review(papers, QUERY, top_k=10)
    for r in reviews:
        flag = "  ⚠️结论相悖" if r.get("conflict") else ""
        title = (r.get("title") or "")[:60]
        print(f"    [{r.get('score', '?')}分] {title}{flag}")
    print("\n  ▶ LLM 打分(0-10) + 三级去重 + 主动标注相悖论文对——研究的「安全网」。")
    pause()

    # ── 场景 5：Summarize ──────────────────────────────────────────
    hr("场景 5：多篇摘要（Comparison 横向对比 + Gap 空白分析）")
    from src.agents.summarizer import PaperSummarizer

    s = PaperSummarizer()
    if len(papers) >= 2:
        with tracer.trace("summarize_compare", {"n": min(3, len(papers))}):
            cmp = s.compare(papers[:3])
        print("  ◆ 横向对比 (compare):")
        dump(cmp)
        print()
    else:
        print("  （检索结果不足 2 篇，跳过对比）")
    if papers:
        with tracer.trace("summarize_gap", {"n": min(5, len(papers))}):
            gap = s.gap_analysis(papers[:5])
        print("  ◆ 研究空白 (gap_analysis):")
        dump(gap)
    print("\n  ▶ 读单篇 Brief/Detailed，读多篇 Comparison，找选题 Gap Analysis。")
    pause()

    # ── 场景 6：混合检索 + LangFuse ────────────────────────────────
    hr("场景 6：混合检索评测 + LangFuse 火焰图")
    from src.rag.eval_set import CORPUS, TfidfSemanticRanker, build_eval_set
    from src.rag.hybrid_search import HybridRetriever, mean_reciprocal_rank, recall_at_k

    corpus = [f"{d['title']}. {d['abstract']}" for d in CORPUS]
    ranker = TfidfSemanticRanker(corpus)
    retriever = HybridRetriever(corpus)
    eval_set = build_eval_set()
    search_fn = lambda q: [r["index"] for r in retriever.search(q, ranker.rank, k=10)]  # noqa: E731
    mrr = mean_reciprocal_rank(eval_set, search_fn)
    recall = recall_at_k(eval_set, search_fn)
    print(f"  混合检索（BM25 + 语义 + RRF）:  MRR={mrr:.3f} / Recall@10={recall:.3f}")
    print(f"  LangFuse: 以上每个场景都已埋点，退出后到后台看节点火焰图。")
    pause()

    # ── 场景 7：收尾 ───────────────────────────────────────────────
    hr("场景 7：收尾")
    print("  项目结构:  src/{tools, rag, agents, observability, utils}")
    print("  测试:      178 passed / 覆盖率 85%")
    print("  下一阶段:  第5-6周 → Critic Agent + 知识图谱 + 报告生成")
    print("\n  ▶ V1.0 = 从「流水线」升级为「有主管的 Agent 团队」。")
    print()

    tracer.flush()
    print("已 flush LangFuse 埋点数据。演示结束。")


if __name__ == "__main__":
    main()
