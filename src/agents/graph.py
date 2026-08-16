"""
LangGraph 多Agent编排 — V2.0「7-Agent 协作」状态图。

把 7 个专职 Agent 串成一张有状态、可条件路由、可自纠错的图。相比 V1.0 的
线性流水线，这里的关键升级是「回环」——Critic 质检 Writer 产出的六段式报告
（含 [n] 引用），分数不达标就打回 Writer 重写，形成「生成→质检→返工」的
自纠错闭环。

图结构::

    __start__ → supervisor ─(条件边)─┬─→ search → review → summarize ─┐
                                     └────────────→ summarize ───────┤
                                                                     ↓
                                                                  writer ─(六段式报告，含 [n] 引用)
                                                                     ↓
                                                                  critic ─(分数<阈值则打回 writer，最多返工 2 次)
                                                                     ↓
                                                                  planner → END

使用示例:
    from src.agents.graph import build_graph
    app = build_graph()
    result = app.invoke({"query": "CAR-T therapy for solid tumors"})
    print(result["report"])       # Writer 产出的六段式报告
    print(result["directions"])   # Planner 产出的研究方向
    print(app.get_graph().draw_mermaid())
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)

# ── 常量 ────────────────────────────────────────────────────────────────

# 分数低于该阈值时，Critic 判定报告不合格，打回 Summarize 重写
CRITIC_PASS_SCORE = 6.0
# 最多返工次数（critic 最多跑 MAX_CRITIC_RETRIES + 1 轮，防死循环）
MAX_CRITIC_RETRIES = 2


# ── Agent 共享状态 Schema ───────────────────────────────────────────────


class AgentState(TypedDict, total=False):
    """多Agent协作的共享状态（贯穿所有节点的交接界面）。

    Attributes:
        query:         用户原始问题
        plan:          研究步骤计划（Supervisor 生成）
        papers:        检索结果列表（Search 生成）
        reviews:       相关性评分结果（Review 生成）
        summaries:     摘要结果（Summarize 生成）
        gaps:          研究空白（Summarize 的 gap_analysis 生成）
        final_report:  初步报告（Summarize 汇总）
        critic_score:  Critic 综合评分（0-10）
        critic_issues: Critic 问题清单
        critic_retries: Critic 已执行的返工轮数
        report:        最终六段式报告（Writer 生成）
        directions:    研究方向推荐（Planner 生成）
    """

    query: str
    plan: list[str]
    papers: list[dict]
    reviews: list[dict]
    summaries: dict
    gaps: list[Any]
    final_report: str
    critic_score: float
    critic_issues: list[dict]
    critic_retries: int
    report: str
    directions: list[dict]


# ── 条件路由 ────────────────────────────────────────────────────────────


def route_after_supervisor(state: AgentState) -> str:
    """Supervisor 节点的条件边路由函数。

    - plan 非空（需要检索文献）→ 进入 search
    - plan 为空（无需检索，直接总结）→ 直接 summarize
    """
    if state.get("plan"):
        return "search"
    return "summarize"


def route_after_critic(state: AgentState) -> str:
    """Critic 节点的条件边路由函数。

    - 分数 < 阈值 且 尚未超返工上限 → 打回 writer 重写（报告不合格）
    - 否则 → 进入 planner 生成研究方向（报告已通过）

    Returns:
        str — "writer" 或 "planner"
    """
    score = state.get("critic_score", 10.0)
    retries = state.get("critic_retries", 0)
    if score < CRITIC_PASS_SCORE and retries <= MAX_CRITIC_RETRIES:
        return "writer"
    return "planner"


# ── 图构建 ──────────────────────────────────────────────────────────────


def build_graph(
    supervisor=None,
    search_agent=None,
    review_agent=None,
    summarizer=None,
    critic=None,
    writer=None,
    planner=None,
    tracer=None,
) -> Any:
    """构建并编译 LangGraph「7-Agent 协作」编排图。

    Args:
        supervisor:   SupervisorAgent 实例
        search_agent: SearchAgent 实例
        review_agent: ReviewAgent 实例
        summarizer:   PaperSummarizer 实例
        critic:       CriticAgent 实例
        writer:       WriterAgent 实例
        planner:      PlannerAgent 实例
        tracer:       LangfuseTracer 实例（缺省时惰性实例化，无 Key 静默降级）

    Returns:
        CompiledStateGraph — 编译后的图，可 .invoke() / .get_graph()

    Notes:
        测试时传入 mock 对象即可，避免真实 LLM / API 调用。
    """
    # 惰性实例化默认 Agent（测试时传入 mock，避免真实调用）
    if supervisor is None:
        from src.agents.supervisor import SupervisorAgent
        supervisor = SupervisorAgent()
    if search_agent is None:
        from src.agents.search_agent import SearchAgent
        search_agent = SearchAgent()
    if review_agent is None:
        from src.agents.review_agent import ReviewAgent
        review_agent = ReviewAgent()
    if summarizer is None:
        from src.agents.summarizer import PaperSummarizer
        summarizer = PaperSummarizer()
    if critic is None:
        from src.agents.critic_agent import CriticAgent
        critic = CriticAgent()
    if writer is None:
        from src.agents.writer_agent import WriterAgent
        writer = WriterAgent()
    if planner is None:
        from src.agents.planner_agent import PlannerAgent
        planner = PlannerAgent()
    if tracer is None:
        from src.observability.tracing import LangfuseTracer
        tracer = LangfuseTracer()

    # ── 节点定义（纯函数式：输入 State，返回部分 State 更新）────────────

    def supervisor_node(state: AgentState) -> dict:
        query = state["query"]
        logger.info("[supervisor] generating plan for: %s", query[:60])
        with tracer.trace("supervisor", {"query": query}) as span:
            plan = supervisor.generate_plan(query)
            span.set_output({"n_steps": len(plan)})
        return {"plan": plan}

    def search_node(state: AgentState) -> dict:
        query = state["query"]
        logger.info("[search] searching multi-source: %s", query[:60])
        with tracer.trace("search", {"query": query}) as span:
            papers = search_agent.search_all(query)
            span.set_output({"n_papers": len(papers)})
        return {"papers": papers}

    def review_node(state: AgentState) -> dict:
        papers = state.get("papers", [])
        query = state["query"]
        logger.info("[review] reviewing %d papers", len(papers))
        with tracer.trace("review", {"query": query, "n_papers": len(papers)}) as span:
            reviews = review_agent.review(papers, query)
            span.set_output({"n_reviews": len(reviews)})
        return {"reviews": reviews}

    def summarize_node(state: AgentState) -> dict:
        papers = state.get("papers", [])
        logger.info("[summarize] summarizing %d papers", len(papers))
        with tracer.trace("summarize", {"n_papers": len(papers)}) as span:
            summaries = summarizer.summarize_batch(
                [
                    {"title": p.get("title", ""), "abstract": p.get("abstract", "")}
                    for p in papers[:5]
                ],
                summary_type="brief",
            )
            final_report = _compose_report(state, summaries)
            gaps = _derive_gaps(summarizer, papers)
            span.set_output({"n_summaries": len(summaries), "n_gaps": len(gaps)})
        return {"summaries": summaries, "final_report": final_report, "gaps": gaps}

    def critic_node(state: AgentState) -> dict:
        # 核查 Writer 产出的六段式报告（含 [n] 引用），而非 Summarizer 的初步摘要
        report = state.get("report", "")
        papers = state.get("papers", [])
        retries = state.get("critic_retries", 0) + 1
        logger.info("[critic] reviewing report (round %d)", retries)
        with tracer.trace("critic", {"n_papers": len(papers)}) as span:
            verdict = critic.verify(report, papers)
            span.set_output({"score": verdict.get("overall_score")})
        return {
            "critic_score": verdict.get("overall_score", 10.0),
            "critic_issues": verdict.get("issues", []),
            "critic_retries": retries,
        }

    def writer_node(state: AgentState) -> dict:
        logger.info("[writer] generating six-section report")
        with tracer.trace("writer", {"n_papers": len(state.get("papers", []))}) as span:
            report = writer.write_report(state)
            span.set_output({"report_len": len(report)})
        return {"report": report}

    def planner_node(state: AgentState) -> dict:
        gaps = state.get("gaps", [])
        papers = state.get("papers", [])
        logger.info("[planner] generating directions from %d gap(s)", len(gaps))
        with tracer.trace("planner", {"n_gaps": len(gaps)}) as span:
            directions = planner.generate_directions(gaps, papers)
            span.set_output({"n_directions": len(directions)})
        return {"directions": directions}

    # ── 编排 ────────────────────────────────────────────────────────────

    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("search", search_node)
    graph.add_node("review", review_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("critic", critic_node)
    graph.add_node("writer", writer_node)
    graph.add_node("planner", planner_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"search": "search", "summarize": "summarize"},
    )
    graph.add_edge("search", "review")
    graph.add_edge("review", "summarize")
    graph.add_edge("summarize", "writer")
    graph.add_edge("writer", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"writer": "writer", "planner": "planner"},
    )
    graph.add_edge("planner", END)

    logger.info(
        "LangGraph compiled: supervisor→search→review→summarize→writer⇄critic→planner"
    )
    return graph.compile()


# ── 辅助函数 ────────────────────────────────────────────────────────────


def _compose_report(state: AgentState, summaries: Any) -> str:
    """把各阶段产物拼装成初步报告文本（轻量进度摘要，供 UI 预览）。

    Args:
        state:    当前共享状态
        summaries: Summarize 节点产出的摘要列表

    Returns:
        str — 面向用户的初步报告
    """
    plan = state.get("plan", [])
    papers = state.get("papers", [])
    reviews = state.get("reviews", [])

    lines: list[str] = []
    if plan:
        lines.append("研究步骤: " + " → ".join(plan))
    if papers:
        lines.append(f"检索到论文数: {len(papers)}")
    if reviews:
        lines.append(f"经相关性评审: {len(reviews)} 篇")
    if summaries:
        lines.append(f"生成摘要数: {len(summaries)}")

    return "\n".join(lines) if lines else "（未检索到论文，无最终报告）"


def _derive_gaps(summarizer: Any, papers: list[dict]) -> list[Any]:
    """从 Summarizer 的 gap_analysis 中派生研究空白（失败/无该能力时降级为空）。

    Args:
        summarizer: PaperSummarizer 实例（可能是测试里的 mock）
        papers:     论文列表

    Returns:
        List — 研究空白列表（gap 文本），失败时返回 []
    """
    if not papers:
        return []
    if not hasattr(summarizer, "gap_analysis"):
        return []
    try:
        result = summarizer.gap_analysis(papers)
        gaps = result.get("gaps", [])
        if not isinstance(gaps, list):
            return []
        return [g.get("gap", "") for g in gaps if isinstance(g, dict) and g.get("gap")]
    except Exception as e:
        logger.warning("gap_analysis failed (degraded): %s", e)
        return []
