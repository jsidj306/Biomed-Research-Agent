"""
LangGraph 多Agent编排 — 单元测试。

运行: uv run pytest tests/test_graph.py -v

通过注入 mock Agent 来验证图结构、状态流转和条件路由（含 Critic 自纠错回环），
不触发真实 LLM / API 调用。
"""

from unittest.mock import MagicMock

import pytest

from src.agents.graph import (
    AgentState,
    _compose_report,
    _merge_local_document,
    build_graph,
    route_after_critic,
    route_after_supervisor,
)

# ── Fixtures：mock 7 个 Agent ────────────────────────────────────────────


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
def mock_critic():
    c = MagicMock()
    c.verify.return_value = {"issues": [], "overall_score": 10}
    return c


@pytest.fixture
def mock_writer():
    w = MagicMock()
    w.write_report.return_value = "## Abstract\n\nsummary\n\n## References\n\n[1] Paper A"
    return w


@pytest.fixture
def mock_planner():
    p = MagicMock()
    p.generate_directions.return_value = [
        {"title": "Direction 1", "feasibility": "high", "novelty": "high"},
    ]
    return p


@pytest.fixture
def all_mocks(
    mock_supervisor,
    mock_search_agent,
    mock_review_agent,
    mock_summarizer,
    mock_critic,
    mock_writer,
    mock_planner,
):
    return {
        "supervisor": mock_supervisor,
        "search_agent": mock_search_agent,
        "review_agent": mock_review_agent,
        "summarizer": mock_summarizer,
        "critic": mock_critic,
        "writer": mock_writer,
        "planner": mock_planner,
    }


# ── State Schema 测试 ────────────────────────────────────────────────────


class TestAgentState:
    """测试 AgentState Schema 定义完整性。"""

    def test_state_has_all_fields(self):
        """AgentState 应包含 V2.0 全部字段。"""
        required = {
            "query", "plan", "papers", "reviews", "summaries", "gaps",
            "final_report", "critic_score", "critic_issues", "critic_retries",
            "report", "directions",
        }
        assert required.issubset(set(AgentState.__annotations__))

    def test_state_partial_construction(self):
        """total=False 允许只填部分字段（图初始只传 query）。"""
        state = AgentState(query="test query")
        assert state["query"] == "test query"
        assert state.get("plan") is None


# ── 条件路由测试 ─────────────────────────────────────────────────────────


class TestRouting:
    """测试 Supervisor / Critic 的条件边路由逻辑。"""

    def test_route_with_plan_goes_to_search(self):
        assert route_after_supervisor({"plan": ["step1"]}) == "search"

    def test_route_without_plan_goes_to_summarize(self):
        assert route_after_supervisor({"plan": []}) == "summarize"

    def test_route_missing_plan_key(self):
        assert route_after_supervisor({}) == "summarize"

    def test_critic_low_score_retries(self):
        """分数低于阈值且未超返工上限 → 打回 writer 重写。"""
        assert route_after_critic({"critic_score": 5, "critic_retries": 1}) == "writer"

    def test_critic_high_score_goes_to_planner(self):
        assert route_after_critic({"critic_score": 8, "critic_retries": 1}) == "planner"

    def test_critic_max_retries_bounded(self):
        """超返工上限后即使低分也强制进 planner（防死循环）。"""
        assert route_after_critic({"critic_score": 5, "critic_retries": 3}) == "planner"


# ── 图结构测试 ───────────────────────────────────────────────────────────


class TestGraphStructure:
    """测试 StateGraph 编译后的节点/边结构。"""

    def test_graph_compiles(self, all_mocks):
        app = build_graph(**all_mocks)
        assert app is not None

    def test_graph_nodes(self, all_mocks):
        """图应包含 7 个节点。"""
        app = build_graph(**all_mocks)
        nodes = set(app.get_graph().nodes)
        assert {"supervisor", "search", "review", "summarize", "critic", "writer", "planner"}.issubset(nodes)

    def test_graph_has_edges(self, all_mocks):
        """图应有 7-Agent 的边连接。"""
        app = build_graph(**all_mocks)
        edge_pairs = {(e.source, e.target) for e in app.get_graph().edges}
        assert ("search", "review") in edge_pairs
        assert ("review", "summarize") in edge_pairs
        assert ("summarize", "writer") in edge_pairs
        assert ("writer", "critic") in edge_pairs
        assert ("planner", "__end__") in edge_pairs


# ── 全链路 invoke 测试 ───────────────────────────────────────────────────


class TestFullChain:
    """测试 invoke 走完 7-Agent 全链路。"""

    def test_invoke_full_chain(self, all_mocks):
        """正常 query（有 plan）走完整链路，各节点产物写入 State。"""
        app = build_graph(**all_mocks)
        result = app.invoke({"query": "CAR-T therapy for solid tumors"})

        assert result["plan"] == all_mocks["supervisor"].generate_plan.return_value
        assert result["papers"] == all_mocks["search_agent"].search_all.return_value
        assert result["reviews"] == all_mocks["review_agent"].review.return_value
        assert result["critic_score"] == 10
        assert result["report"]  # Writer 产出六段式报告
        assert result["directions"]  # Planner 产出研究方向
        assert "CAR-T" in all_mocks["supervisor"].generate_plan.call_args[0][0]

    def test_invoke_updates_state_sequentially(self, all_mocks):
        """节点按 supervisor→…→planner 顺序执行。"""
        app = build_graph(**all_mocks)
        app.invoke({"query": "query"})

        assert all_mocks["supervisor"].generate_plan.called
        assert all_mocks["search_agent"].search_all.called
        assert all_mocks["review_agent"].review.called
        assert all_mocks["summarizer"].summarize_batch.called
        assert all_mocks["critic"].verify.called
        assert all_mocks["writer"].write_report.called
        assert all_mocks["planner"].generate_directions.called

    def test_critic_rejects_triggers_rewrite(self, all_mocks, mock_critic, mock_writer):
        """Critic 低分应触发 writer 重写（write_report 被调用 ≥2 次）。"""
        mock_critic.verify.side_effect = [
            {"issues": [{"claim": "x", "problem": "无引用", "severity": "high"}], "overall_score": 3},
            {"issues": [], "overall_score": 9},
        ]
        all_mocks["critic"] = mock_critic

        app = build_graph(**all_mocks)
        result = app.invoke({"query": "query"})

        assert result["critic_score"] == 9  # 第二次评审通过
        assert mock_writer.write_report.call_count >= 2  # 被打回重写了一次


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
        assert result["report"]  # writer 仍应产出报告


# ── 可视化导出测试 ───────────────────────────────────────────────────────


class TestVisualization:
    """测试 Mermaid 流程图导出。"""

    def test_mermaid_export(self, all_mocks):
        app = build_graph(**all_mocks)
        mermaid = app.get_graph().draw_mermaid()
        assert "supervisor" in mermaid
        assert "critic" in mermaid
        assert "writer" in mermaid
        assert "planner" in mermaid


# ── 本地文档接入测试 ─────────────────────────────────────────────────────


class TestLocalDocument:
    """测试本地 PDF 全文接入流水线（_merge_local_document）。"""

    def test_no_document_returns_papers_unchanged(self):
        papers = [{"title": "A", "source": "pubmed"}]
        assert _merge_local_document({}, papers) == papers

    def test_blank_document_ignored(self):
        papers = [{"title": "A"}]
        assert _merge_local_document({"document_text": "   "}, papers) == papers

    def test_document_prepended_as_local_pdf(self):
        fulltext = "x" * 2000
        papers = [{"title": "A", "source": "pubmed"}]
        merged = _merge_local_document(
            {"document_text": fulltext, "document_title": "BCI Paper"},
            papers,
        )
        assert merged[0]["source"] == "local_pdf"
        assert merged[0]["title"] == "BCI Paper"
        assert merged[0]["fulltext"] == fulltext
        assert merged[0]["abstract"] == fulltext  # 2000 < 3000 截断上限，等于全文
        assert len(merged) == 2

    def test_invoke_with_local_document(self, all_mocks):
        """invoke 传入 document_text 时，本地文档应成为 papers 的第 1 篇。"""
        app = build_graph(**all_mocks)
        result = app.invoke({
            "query": "query",
            "document_text": "full text " * 100,
            "document_title": "My BCI Paper",
        })
        assert result["papers"][0]["source"] == "local_pdf"
        assert result["papers"][0]["title"] == "My BCI Paper"
        assert len(result["papers"]) == 3  # 本地文档 + 2 篇 mock 检索结果


# ── 报告合成测试 ─────────────────────────────────────────────────────────


class TestComposeReport:
    """测试 _compose_report 的边界处理。"""

    def test_empty_state_produces_placeholder(self):
        report = _compose_report({}, [])
        assert "未检索到论文" in report

    def test_report_includes_plan_and_counts(self):
        state = {
            "plan": ["step1", "step2"],
            "papers": [{"title": "A"}, {"title": "B"}],
            "reviews": [{"title": "A"}],
        }
        report = _compose_report(state, [{"paper_title": "A"}])
        assert "step1" in report
        assert "检索到论文数: 2" in report
        assert "生成摘要数: 1" in report


# ── LangFuse 埋点集成测试 ────────────────────────────────────────────────


class TestTracingIntegration:
    """测试各节点在 invoke 时产生 LangFuse 调用链埋点。"""

    def test_nodes_record_traces(self, all_mocks):
        """注入 mock 客户端，7 个节点应各产生一次 span 埋点。"""
        from src.observability.tracing import LangfuseTracer

        mock_client = MagicMock()
        mock_client.start_observation.return_value = MagicMock()
        tracer = LangfuseTracer(client=mock_client)

        app = build_graph(tracer=tracer, **all_mocks)
        app.invoke({"query": "CAR-T therapy for solid tumors"})

        names = [c.kwargs["name"] for c in mock_client.start_observation.call_args_list]
        for node in ("supervisor", "search", "review", "summarize", "critic", "writer", "planner"):
            assert node in names
