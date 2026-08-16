"""
Streamlit UI Components — 可复用的论文展示组件。

提供论文卡片、摘要展示、搜索结果显示等 UI 组件，
供 app.py 调用，保持代码分离和可测试性。

使用示例:
    from src.ui.components import render_paper_card, render_summary_tabs
    render_paper_card(paper, index=1)
"""

import re
from typing import Any

import streamlit as st

# ── 论文卡片 ────────────────────────────────────────────────────────────


def render_paper_card(paper: dict[str, Any], index: int = 1) -> None:
    """渲染单篇论文的展开式卡片。

    默认折叠摘要，点击展开后可选择生成 LLM 结构化摘要。

    Args:
        paper: 论文字典 (PubMed/Arxiv 统一格式)
        index: 序号
    """
    title = paper.get("title", "Untitled")
    authors = paper.get("authors", [])
    journal = paper.get("journal", "")
    year = paper.get("year", "")
    abstract = paper.get("abstract", paper.get("summary", ""))
    source = paper.get("source", "unknown")
    source_id = paper.get("pmid", paper.get("arxiv_id", ""))
    url = paper.get("url", paper.get("pdf_url", ""))
    doi = paper.get("doi", "")

    with st.container():
        # ── 标题行 ──
        st.markdown(f"### {index}. {title[:150]}")

        # ── 元数据行 ──
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        with col1:
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += " et al."
            st.caption(f"👤 {author_str}" if author_str else "👤 Unknown")
        with col2:
            source_info = f"📰 {journal} ({year})" if journal else f"📅 {year}" if year else ""
            if source_info:
                st.caption(source_info)
        with col3:
            source_badge = {
                "pubmed": "🔵 PubMed",
                "arxiv": "🟢 Arxiv",
            }.get(source, f"📄 {source}")
            st.caption(source_badge)
        with col4:
            if url:
                st.caption(f"[🔗 Link]({url})")
            if doi:
                st.caption(f"[📎 DOI](https://doi.org/{doi})")

        # ── 摘要展开区 ──
        with st.expander(f"📄 Abstract — {title[:60]}...", expanded=False):
            if abstract:
                st.markdown(abstract[:2000])
            else:
                st.caption("No abstract available.")

        # ── ID 信息（PMID 可点击跳转 PubMed 原文）──
        if source == "pubmed" and source_id:
            st.caption(f"PMID: {build_pmid_link(source_id)}")
        elif source_id:
            st.caption(f"{source}: `{source_id}`")


def render_paper_list(papers: list[dict[str, Any]], max_display: int = 10) -> None:
    """渲染论文列表。

    Args:
        papers:    论文列表
        max_display: 最多显示的论文数
    """
    if not papers:
        st.info("No papers found.")
        return

    st.markdown(f"**Found {len(papers)} papers** — showing top {min(len(papers), max_display)}")

    for i, paper in enumerate(papers[:max_display], 1):
        render_paper_card(paper, index=i)
        st.divider()


# ── 摘要展示 ────────────────────────────────────────────────────────────


def render_brief_summary(data: dict[str, str]) -> None:
    """渲染简短摘要 (one_liner + takeaway)。"""
    if not data:
        return
    one_liner = data.get("one_liner", "")
    takeaway = data.get("takeaway", "")
    if one_liner:
        st.markdown(f"💡 **{one_liner}**")
    if takeaway:
        st.markdown(f"🔑 {takeaway}")


def render_detailed_summary(data: dict[str, str]) -> None:
    """渲染详细四段式摘要。"""
    if not data:
        return
    sections = [
        ("📋 Background", "background"),
        ("🔬 Methods", "methods"),
        ("📊 Results", "results"),
        ("💡 Conclusions", "conclusions"),
    ]
    for label, key in sections:
        content = data.get(key, "")
        if content:
            st.markdown(f"**{label}:** {content}")


def render_key_findings(data: dict[str, Any]) -> None:
    """渲染关键发现要点列表。"""
    if not data:
        return
    findings = data.get("findings", [])
    significance = data.get("significance", "")
    if findings:
        for j, finding in enumerate(findings, 1):
            st.markdown(f"{j}. {finding}")
    if significance:
        st.markdown(f"---\n📊 **Significance:** {significance}")


def render_summary_tabs(summary_result: dict[str, Any]) -> None:
    """以标签页形式展示所有摘要类型。

    Args:
        summary_result: PaperSummarizer.summarize() 的返回结果
    """
    if "error" in summary_result or summary_result.get("warning"):
        msg = summary_result.get("error", summary_result.get("warning", "Unknown error"))
        st.warning(f"⚠️ {msg}")
        return

    brief = summary_result.get("brief")
    detailed = summary_result.get("detailed")
    key_findings = summary_result.get("key_findings")
    summary_type = summary_result.get("summary_type", "all")

    has_any = brief or detailed or key_findings
    if not has_any:
        st.info("No summary generated.")
        return

    if summary_type == "all":
        tabs = st.tabs(["💡 Brief", "📋 Detailed", "🔑 Key Findings"])
        with tabs[0]:
            render_brief_summary(brief)
        with tabs[1]:
            render_detailed_summary(detailed)
        with tabs[2]:
            render_key_findings(key_findings)
    elif summary_type == "brief":
        render_brief_summary(brief)
    elif summary_type == "detailed":
        render_detailed_summary(detailed)
    elif summary_type == "key_findings":
        render_key_findings(key_findings)


# ── 引用溯源 ──────────────────────────────────────────────────────────


def build_pmid_link(pmid: str) -> str:
    """生成 PMID 的 PubMed 跳转链接（Markdown，纯函数便于测试）。

    Args:
        pmid: PubMed ID 字符串

    Returns:
        str — Markdown 链接，空 pmid 返回空串
    """
    pmid = str(pmid).strip()
    if not pmid:
        return ""
    return f"[PMID:{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)"


def _linkify_citations(report: str, papers: list[dict[str, Any]]) -> str:
    """把报告里的 [n] 引用替换为可点击的 PubMed 链接（纯函数，便于测试）。

    引用的 [n] 编号对应 papers 的 1-based 下标；对应论文有 pmid 时替换为
    链接，否则保留原样，杜绝「编造引用」。

    Args:
        report: 报告 Markdown 文本
        papers: 论文列表（用于 [n] → pmid 映射）

    Returns:
        str — 替换后的报告文本
    """
    if not papers:
        return report

    def _repl(match: re.Match) -> str:
        n = int(match.group(1))
        if 1 <= n <= len(papers):
            pmid = str(papers[n - 1].get("pmid", "")).strip()
            if pmid:
                return f"[{n}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)"
        return match.group(0)

    return re.sub(r"\[(\d+)\]", _repl, report)


def render_report(report: str, papers: list[dict[str, Any]] | None = None) -> None:
    """渲染六段式报告，并把 [n] 引用转成可点击的 PubMed 链接。

    Args:
        report: 报告 Markdown 文本（Writer 产出）
        papers: 论文列表（用于引用溯源；缺省时不替换引用）
    """
    if not report:
        st.info("No report to display.")
        return
    rendered = _linkify_citations(report, papers or [])
    st.markdown(rendered)


# ── 侧边栏配置 ──────────────────────────────────────────────────────────


def render_sidebar() -> dict[str, Any]:
    """渲染侧边栏配置面板。

    Returns:
        配置字典 {search_source, max_results, summary_type, language, model}
    """
    with st.sidebar:
        st.header("⚙️ Configuration")

        search_source = st.multiselect(
            "Search Sources",
            options=["PubMed", "Arxiv"],
            default=["PubMed", "Arxiv"],
            help="Select which databases to search",
        )

        max_results = st.slider(
            "Max Results per Source",
            min_value=1,
            max_value=20,
            value=5,
            help="Maximum number of papers per source",
        )

        summary_type = st.selectbox(
            "Summary Type",
            options=["all", "brief", "detailed", "key_findings"],
            format_func=lambda x: {
                "all": "All Types",
                "brief": "Brief (1 sentence)",
                "detailed": "Detailed (4 sections)",
                "key_findings": "Key Findings",
            }.get(x, x),
            help="Type of summary to generate",
        )

        language = st.selectbox(
            "Output Language",
            options=["english", "chinese"],
            format_func=lambda x: "English" if x == "english" else "中文",
        )

        model = st.selectbox(
            "LLM Model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
            help="GPT-4o-mini is fast and affordable (recommended)",
        )

        st.divider()
        st.caption("Biomed Research Agent v0.1.0")
        st.caption("Built with LangChain + Streamlit")

    return {
        "search_source": search_source,
        "max_results": max_results,
        "summary_type": summary_type,
        "language": language,
        "model": model,
    }
