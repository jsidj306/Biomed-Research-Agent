"""
LangGraph 多Agent编排 — 核心状态图骨架。

把 4 个专职 Agent（Supervisor / Search / Review / Summarize）用 StateGraph
串成一张有状态、可条件路由的图。对比 MVP 阶段的线性流水线，这里的关键升级是
「条件边」：Supervisor 产出研究计划后，根据是否有可执行的检索步骤，决定是
进入 search 节点还是直接跳到 summarize 节点。

图结构::

    __start__ → supervisor ──(有条件边)──┬─→ search → review → summarize → END
                                          └─→ summarize ────────────────→ END

使用示例:
    from src.agents.graph import build_graph
    app = build_graph()
    result = app.invoke({"query": "CAR-T therapy for solid tumors"})
    print(result["final_report"])
    print(app.get_graph().draw_mermaid())
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)


# ── Agent 共享状态 Schema ───────────────────────────────────────────────


class AgentState(TypedDict, total=False):
    """多Agent协作的共享状态（贯穿所有节点的交接界面）。

    用 TypedDict 而非普通 dict 是为了给字段加类型约束：
    IDE 能自动补全、拼错字段名会报错，避免多Agent协作时因字段名
    不一致而出的隐蔽 bug。

    Attributes:
        query:        用户原始问题
        plan:         研究步骤计划（Supervisor 生成）
        papers:       检索结果列表（Search 生成）
        reviews:      相关性评分结果（Review 生成）
        summaries:    摘要结果（Summarize 生成）
        final_report: 最终报告（Summarize 汇总）
    """

    query: str
    plan: list[str]
    papers: list[dict]
    reviews: list[dict]
    summaries: dict
    final_report: str


# ── 条件路由 ────────────────────────────────────────────────────────────


def route_after_supervisor(state: AgentState) -> str:
    """Supervisor 节点的条件边路由函数。

    Supervisor 产出的 plan 决定下一步去哪个节点：
    - plan 非空（需要检索文献）→ 进入 search
    - plan 为空（无需检索，直接总结）→ 直接 summarize

    Args:
        state: 当前共享状态

    Returns:
        str — 下一节点名 "search" 或 "summarize"
    """
    if state.get("plan"):
        return "search"
    return "summarize"


# ── 图构建 ──────────────────────────────────────────────────────────────


def build_graph(
    supervisor=None,
    search_agent=None,
    review_agent=None,
    summarizer=None,
    tracer=None,
) -> Any:
    """构建并编译 LangGraph 多Agent编排图。

    Args:
        supervisor:   SupervisorAgent 实例（缺省时惰性实例化真实对象）
        search_agent: SearchAgent 实例
        review_agent: ReviewAgent 实例
        summarizer:   PaperSummarizer 实例
        tracer:       LangfuseTracer 实例（缺省时惰性实例化，无 Key 静默降级）

    Returns:
        CompiledStateGraph — 编译后的图，可 .invoke() / .get_graph()

    Notes:
        测试时传入 mock 对象即可，避免真实 LLM / API 调用。
        每个节点用 tracer.trace 包裹，未配置 LangFuse Key 时自动降级为
        no-op，不影响主流程。
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
    if tracer is None:
        from src.observability.tracing import LangfuseTracer

        tracer = LangfuseTracer()

    # ── 节点定义（纯函数式：输入 State，返回部分 State 更新）────────────

    def supervisor_node(state: AgentState) -> dict:
        """Supervisor：意图解析 + 研究步骤计划生成。"""
        query = state["query"]
        logger.info("[supervisor] generating plan for: %s", query[:60])
        with tracer.trace("supervisor", {"query": query}) as span:
            plan = supervisor.generate_plan(query)
            span.set_output({"n_steps": len(plan)})
        return {"plan": plan}

    def search_node(state: AgentState) -> dict:
        """Search：多源并发检索。"""
        query = state["query"]
        logger.info("[search] searching multi-source: %s", query[:60])
        with tracer.trace("search", {"query": query}) as span:
            papers = search_agent.search_all(query)
            span.set_output({"n_papers": len(papers)})
        return {"papers": papers}

    def review_node(state: AgentState) -> dict:
        """Review：相关性评分 + 去重 + 排序。"""
        papers = state.get("papers", [])
        query = state["query"]
        logger.info("[review] reviewing %d papers", len(papers))
        with tracer.trace("review", {"query": query, "n_papers": len(papers)}) as span:
            reviews = review_agent.review(papers, query)
            span.set_output({"n_reviews": len(reviews)})
        return {"reviews": reviews}

    def summarize_node(state: AgentState) -> dict:
        """Summarize：对 Top 论文生成摘要并汇总最终报告。"""
        papers = state.get("papers", [])
        logger.info("[summarize] summarizing %d papers", len(papers))
        with tracer.trace("summarize", {"n_papers": len(papers)}) as span:
            summaries = summarizer.summarize_batch(
                [
                    {
                        "title": p.get("title", ""),
                        "abstract": p.get("abstract", ""),
                    }
                    for p in papers[:5]
                ],
                summary_type="brief",
            )
            final_report = _compose_report(state, summaries)
            span.set_output({"n_summaries": len(summaries)})
        return {"summaries": summaries, "final_report": final_report}

    # ── 编排 ────────────────────────────────────────────────────────────

    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("search", search_node)
    graph.add_node("review", review_node)
    graph.add_node("summarize", summarize_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"search": "search", "summarize": "summarize"},
    )
    graph.add_edge("search", "review")
    graph.add_edge("review", "summarize")
    graph.add_edge("summarize", END)

    logger.info("LangGraph compiled: supervisor→search→review→summarize")
    return graph.compile()


# ── 报告合成 ────────────────────────────────────────────────────────────


def _compose_report(state: AgentState, summaries: Any) -> str:
    """把各阶段产物拼装成最终报告文本。

    Args:
        state:    当前共享状态
        summaries: Summarize 节点产出的摘要列表

    Returns:
        str — 面向用户的最终报告
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
