"""
Streamlit UI Components — 可复用的论文展示组件。

提供论文卡片、摘要展示、搜索结果显示等 UI 组件，供 app.py 调用，
保持代码分离和可测试性。所有用户可见文案通过 `src/ui/i18n.t(lang, key)`
支持中英文切换。

使用示例:
    from src.ui.components import render_paper_card, render_summary_tabs
    render_paper_card(paper, index=1, lang="zh")
"""

import re
from typing import Any

import streamlit as st

from src.ui.i18n import t

# ── 论文卡片 ────────────────────────────────────────────────────────────


def render_paper_card(paper: dict[str, Any], index: int = 1, lang: str = "zh") -> None:
    """渲染单篇论文的展开式卡片。

    默认折叠摘要，点击展开后可选择生成 LLM 结构化摘要。

    Args:
        paper: 论文字典 (PubMed/Arxiv 统一格式)
        index: 序号
        lang:  界面语言（zh / en）
    """
    title = paper.get("title", t(lang, "untitled"))
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
            st.caption(f"👤 {author_str}" if author_str else f"👤 {t(lang, 'unknown_author')}")
        with col2:
            source_info = f"📰 {journal} ({year})" if journal else f"📅 {year}" if year else ""
            if source_info:
                st.caption(source_info)
        with col3:
            source_badge = {
                "pubmed": "🔵 PubMed",
                "arxiv": "🟢 Arxiv",
                "cnki": "🟠 CNKI/万方",
            }.get(source, f"📄 {source}")
            st.caption(source_badge)
        with col4:
            link_label = "链接" if lang == "zh" else "Link"
            if url:
                st.caption(f"[🔗 {link_label}]({url})")
            if doi:
                st.caption(f"[📎 DOI](https://doi.org/{doi})")

        # ── 摘要展开区 ──
        with st.expander(f"📄 Abstract — {title[:60]}...", expanded=False):
            if abstract:
                st.markdown(abstract[:2000])
            else:
                st.caption(t(lang, "no_abstract"))

        # ── ID 信息（PMID 可点击跳转 PubMed 原文）──
        if source == "pubmed" and source_id:
            st.caption(f"PMID: {build_pmid_link(source_id)}")
        elif source_id:
            st.caption(f"{source}: `{source_id}`")


def render_paper_list(
    papers: list[dict[str, Any]],
    max_display: int = 10,
    lang: str = "zh",
) -> None:
    """渲染论文列表。

    Args:
        papers:    论文列表
        max_display: 最多显示的论文数
        lang:      界面语言（zh / en）
    """
    if not papers:
        st.info(t(lang, "no_papers_found"))
        return

    st.markdown(
        t(lang, "found_n_papers", count=len(papers), shown=min(len(papers), max_display))
    )

    for i, paper in enumerate(papers[:max_display], 1):
        render_paper_card(paper, index=i, lang=lang)
        st.divider()


# ── 摘要展示 ────────────────────────────────────────────────────────────


def render_brief_summary(data: dict[str, str], lang: str = "zh") -> None:
    """渲染简短摘要 (one_liner + takeaway)。"""
    if not data:
        return
    one_liner = data.get("one_liner", "")
    takeaway = data.get("takeaway", "")
    if one_liner:
        st.markdown(f"💡 **{one_liner}**")
    if takeaway:
        st.markdown(f"🔑 {takeaway}")


def render_detailed_summary(data: dict[str, str], lang: str = "zh") -> None:
    """渲染详细四段式摘要。"""
    if not data:
        return
    sections = [
        (t(lang, "background"), "background"),
        (t(lang, "methods"), "methods"),
        (t(lang, "results"), "results"),
        (t(lang, "conclusions"), "conclusions"),
    ]
    for label, key in sections:
        content = data.get(key, "")
        if content:
            st.markdown(f"**{label}:** {content}")


def render_key_findings(data: dict[str, Any], lang: str = "zh") -> None:
    """渲染关键发现要点列表。"""
    if not data:
        return
    findings = data.get("findings", [])
    significance = data.get("significance", "")
    if findings:
        for j, finding in enumerate(findings, 1):
            st.markdown(f"{j}. {finding}")
    if significance:
        st.markdown(f"---\n{t(lang, 'significance', text=significance)}")


def render_summary_tabs(summary_result: dict[str, Any], lang: str = "zh") -> None:
    """以标签页形式展示所有摘要类型。

    Args:
        summary_result: PaperSummarizer.summarize() 的返回结果
        lang:           界面语言（zh / en）
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
        st.info(t(lang, "no_summary"))
        return

    if summary_type == "all":
        tabs = st.tabs([
            t(lang, "brief_tab"),
            t(lang, "detailed_tab"),
            t(lang, "key_findings_tab"),
        ])
        with tabs[0]:
            render_brief_summary(brief, lang)
        with tabs[1]:
            render_detailed_summary(detailed, lang)
        with tabs[2]:
            render_key_findings(key_findings, lang)
    elif summary_type == "brief":
        render_brief_summary(brief, lang)
    elif summary_type == "detailed":
        render_detailed_summary(detailed, lang)
    elif summary_type == "key_findings":
        render_key_findings(key_findings, lang)


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


def render_report(
    report: str,
    papers: list[dict[str, Any]] | None = None,
    lang: str = "zh",
) -> None:
    """渲染六段式报告，并把 [n] 引用转成可点击的 PubMed 链接。

    Args:
        report: 报告 Markdown 文本（Writer 产出）
        papers: 论文列表（用于引用溯源；缺省时不替换引用）
        lang:   界面语言（zh / en）
    """
    if not report:
        st.info(t(lang, "no_report"))
        return
    rendered = _linkify_citations(report, papers or [])
    st.markdown(rendered)


# ── 侧边栏配置 ──────────────────────────────────────────────────────────


def render_sidebar() -> dict[str, Any]:
    """渲染侧边栏配置面板。

    Returns:
        配置字典 {search_source, max_results, summary_type, language, model, lang}
    """
    with st.sidebar:
        # 界面语言（zh / en），与「输出语言」是两回事：前者控制 UI 文案，后者控制摘要语言
        lang = st.radio(
            "🌐 语言 / Language",
            options=["zh", "en"],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
        )

        st.header(t(lang, "config"))

        search_source = st.multiselect(
            t(lang, "search_sources"),
            options=["PubMed", "Arxiv", "CNKI"],
            default=["PubMed", "Arxiv"],
            help=t(lang, "search_sources_help"),
        )

        max_results = st.slider(
            t(lang, "max_results"),
            min_value=1,
            max_value=20,
            value=5,
            help=t(lang, "max_results_help"),
        )

        summary_type = st.selectbox(
            t(lang, "summary_type"),
            options=["all", "brief", "detailed", "key_findings"],
            format_func=lambda x: {
                "all": t(lang, "summary_type_all"),
                "brief": t(lang, "summary_type_brief"),
                "detailed": t(lang, "summary_type_detailed"),
                "key_findings": t(lang, "summary_type_key"),
            }.get(x, x),
            help=t(lang, "summary_type_help"),
        )

        language = st.selectbox(
            t(lang, "output_language"),
            options=["english", "chinese"],
            format_func=lambda x: "English" if x == "english" else "中文",
        )

        model = st.selectbox(
            t(lang, "llm_model"),
            options=["glm-5.2", "kimi-k2.7-code"],
            index=0,
            help=t(lang, "llm_model_help"),
        )

        st.divider()
        st.caption(t(lang, "footer_version"))
        st.caption(t(lang, "footer_built"))

    return {
        "search_source": search_source,
        "max_results": max_results,
        "summary_type": summary_type,
        "language": language,
        "model": model,
        "lang": lang,
    }
