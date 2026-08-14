"""
LangGraph 多Agent编排 — 单元测试。

运行: uv run pytest tests/test_graph.py -v

通过注入 mock Agent 来验证图结构、状态流转和条件路由，
不触发真实 LLM / API 调用。
"""

from unittest.mock import MagicMock

import pytest

from src.agents.graph import (
    AgentState,
    _compose_report,
    build_graph,
    route_after_supervisor,
)

# ── Fixtures：mock 4 个 Agent ────────────────────────────────────────────


@pytest.fixture
def mock_supervisor():
    sup = MagicMock()
    sup.generate_plan.return_value = [
        "Search PubMed for CAR-T solid tumor trials",
        "Review clinical outcomes",
    ]
    return sup


@pytest.fixture
def mock_search_agent():
    agent = MagicMock()
    agent.search_all.return_value = [
        {"title": "Paper A", "abstract": "abstract A " * 20, "source": "pubmed"},
        {"title": "Paper B", "abstract": "abstract B " * 20, "source": "arxiv"},
    ]
    return agent


@pytest.fixture
def mock_review_agent():
    agent = MagicMock()
    agent.review.return_value = [
        {"title": "Paper A", "score": 9, "reason": "highly relevant"},
    ]
    return agent


@pytest.fixture
def mock_summarizer():
    s = MagicMock()
    s.summarize_batch.return_value = [
        {"paper_title": "Paper A", "summary_type": "brief", "brief": {"one_liner": "x"}},
    ]
    return s


@pytest.fixture
def all_mocks(mock_supervisor, mock_search_agent, mock_review_agent, mock_summarizer):
    return {
        "supervisor": mock_supervisor,
        "search_agent": mock_search_agent,
        "review_agent": mock_review_agent,
        "summarizer": mock_summarizer,
    }


# ── State Schema 测试 ────────────────────────────────────────────────────


class TestAgentState:
    """测试 AgentState Schema 定义完整性。"""

    def test_state_has_all_fields(self):
        """AgentState 应包含 query/plan/papers/reviews/summaries/final_report。"""
        required = {"query", "plan", "papers", "reviews", "summaries", "final_report"}
        # TypedDict 的字段名在 __annotations__ 中
        assert required.issubset(set(AgentState.__annotations__))

    def test_state_partial_construction(self):
        """total=False 允许只填部分字段（图初始只传 query）。"""
        state = AgentState(query="test query")
        assert state["query"] == "test query"
        assert state.get("plan") is None


# ── 条件路由测试 ─────────────────────────────────────────────────────────


class TestRouting:
    """测试 Supervisor 条件边路由逻辑。"""

    def test_route_with_plan_goes_to_search(self):
        """plan 非空 → 路由到 search。"""
        assert route_after_supervisor({"plan": ["step1"]}) == "search"

    def test_route_without_plan_goes_to_summarize(self):
        """plan 为空 → 直接 summarize。"""
        assert route_after_supervisor({"plan": []}) == "summarize"

    def test_route_missing_plan_key(self):
        """plan 字段缺失 → 视为无需检索。"""
        assert route_after_supervisor({}) == "summarize"


# ── 图结构测试 ───────────────────────────────────────────────────────────


class TestGraphStructure:
    """测试 StateGraph 编译后的节点/边结构。"""

    def test_graph_compiles(self, all_mocks):
        """build_graph 能成功编译。"""
        app = build_graph(**all_mocks)
        assert app is not None

    def test_graph_nodes(self, all_mocks):
        """图应包含 4 个节点。"""
        app = build_graph(**all_mocks)
        nodes = set(app.get_graph().nodes)
        assert {"supervisor", "search", "review", "summarize"}.issubset(nodes)

    def test_graph_has_edges(self, all_mocks):
        """图应有 supervisor→search→review→summarize 的边连接。"""
        app = build_graph(**all_mocks)
        # edges 是 Edge 对象列表，比较其 (source, target)
        edge_pairs = {(e.source, e.target) for e in app.get_graph().edges}
        assert ("search", "review") in edge_pairs
        assert ("review", "summarize") in edge_pairs
        assert ("summarize", "__end__") in edge_pairs


# ── 全链路 invoke 测试 ───────────────────────────────────────────────────


class TestFullChain:
    """测试 invoke 走完 supervisor→search→review→summarize 全链路。"""

    def test_invoke_full_chain(self, all_mocks):
        """正常 query（有 plan）走完整链路，各节点产物写入 State。"""
        app = build_graph(**all_mocks)
        result = app.invoke({"query": "CAR-T therapy for solid tumors"})

        # Supervisor 生成 plan
        assert result["plan"] == all_mocks["supervisor"].generate_plan.return_value
        # Search 生成 papers
        assert result["papers"] == all_mocks["search_agent"].search_all.return_value
        # Review 生成 reviews
        assert result["reviews"] == all_mocks["review_agent"].review.return_value
        # Summarize 生成 final_report
        assert result["final_report"]
        assert "CAR-T" in all_mocks["supervisor"].generate_plan.call_args[0][0]

    def test_invoke_updates_state_sequentially(self, all_mocks):
        """节点按 supervisor→search→review→summarize 顺序执行。"""
        app = build_graph(**all_mocks)
        app.invoke({"query": "query"})

        assert all_mocks["supervisor"].generate_plan.called
        assert all_mocks["search_agent"].search_all.called
        assert all_mocks["review_agent"].review.called
        assert all_mocks["summarizer"].summarize_batch.called


# ── 条件路由端到端测试 ───────────────────────────────────────────────────


class TestConditionalRouting:
    """测试条件边在 invoke 时的真实路由行为。"""

    def test_empty_plan_skips_search(self, all_mocks, mock_supervisor):
        """plan 为空时跳过 search，search_agent 不应被调用。"""
        mock_supervisor.generate_plan.return_value = []
        all_mocks["supervisor"] = mock_supervisor

        app = build_graph(**all_mocks)
        result = app.invoke({"query": "summarize this text"})

        assert not all_mocks["search_agent"].search_all.called
        assert result["plan"] == []
        assert result["final_report"]  # summarize 仍应产出报告


# ── 可视化导出测试 ───────────────────────────────────────────────────────


class TestVisualization:
    """测试 Mermaid 流程图导出。"""

    def test_mermaid_export(self, all_mocks):
        """graph.get_graph().draw_mermaid() 能导出含节点的 Mermaid 文本。"""
        app = build_graph(**all_mocks)
        mermaid = app.get_graph().draw_mermaid()
        assert "supervisor" in mermaid
        assert "search" in mermaid
        assert "summarize" in mermaid


# ── 报告合成测试 ─────────────────────────────────────────────────────────


class TestComposeReport:
    """测试 _compose_report 的边界处理。"""

    def test_empty_state_produces_placeholder(self):
        """空 State 应返回占位文案而非抛异常。"""
        report = _compose_report({}, [])
        assert "未检索到论文" in report

    def test_report_includes_plan_and_counts(self):
        """报告应包含研究步骤和论文/评审计数。"""
        state = {
            "plan": ["step1", "step2"],
            "papers": [{"title": "A"}, {"title": "B"}],
            "reviews": [{"title": "A"}],
        }
        report = _compose_report(state, [{"paper_title": "A"}])
        assert "step1" in report
        assert "检索到论文数: 2" in report
        assert "生成摘要数: 1" in report
