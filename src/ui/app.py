"""
Biomedical Research Agent — Streamlit Chat UI。

对话式生物医学论文研究助手。用户输入生物医学问题 →
搜索 PubMed/Arxiv → 展示论文列表 → 一键生成 LLM 结构化摘要。

运行: uv run streamlit run src/ui/app.py

架构:
  User → Streamlit UI → PubMedSearchTool / ArxivSearchTool (搜索)
                      → RAGPipeline (语义检索, 可选)
                      → PaperSummarizer (LLM 摘要)
"""

import logging
import sys
from pathlib import Path

# 确保项目根目录在 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from src.agents.summarizer import PaperSummarizer
from src.tools.arxiv_tool import ArxivSearchTool
from src.tools.pubmed_tool import PubMedSearchTool
from src.ui.components import (
    render_paper_list,
    render_sidebar,
    render_summary_tabs,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 页面配置 ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Biomed Research Agent 🧬",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State 初始化 ────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []  # 聊天历史: [{role, content, papers?, summary?}]

if "current_papers" not in st.session_state:
    st.session_state.current_papers = []  # 最近一次搜索结果

if "search_triggered" not in st.session_state:
    st.session_state.search_triggered = False


# ── 缓存工具实例（只初始化一次）─────────────────────────────────────────


@st.cache_resource
def get_pubmed_tool():
    return PubMedSearchTool()


@st.cache_resource
def get_arxiv_tool():
    return ArxivSearchTool()


@st.cache_resource
def get_summarizer(model: str = "gpt-4o-mini"):
    return PaperSummarizer(model_name=model, temperature=0.3)


# ── 搜索逻辑 ────────────────────────────────────────────────────────────


def search_papers(query: str, sources: list, max_results: int) -> list:
    """并行搜索多个数据源，合并结果。

    Args:
        query:       搜索查询词
        sources:     数据源列表 ["PubMed", "Arxiv"]
        max_results: 每个源的最大结果数

    Returns:
        合并后的论文列表
    """
    all_papers = []
    pubmed_tool = get_pubmed_tool()
    arxiv_tool = get_arxiv_tool()

    for source in sources:
        try:
            if source == "PubMed":
                result = pubmed_tool._run(query=query, max_results=max_results)
                papers = result.get("papers", [])
                for p in papers:
                    p["source"] = "pubmed"
                all_papers.extend(papers)
                logger.info("PubMed: %d papers", len(papers))

            elif source == "Arxiv":
                result = arxiv_tool._run(query=query, max_results=max_results)
                papers = result.get("papers", [])
                for p in papers:
                    p["source"] = "arxiv"
                    # 统一 abstract 字段
                    if "summary" in p and "abstract" not in p:
                        p["abstract"] = p.pop("summary")
                all_papers.extend(papers)
                logger.info("Arxiv: %d papers", len(papers))

        except Exception as e:
            logger.error("Search failed for %s: %s", source, e)
            st.error(f"Search failed for {source}: {e}")

    return all_papers


# ── 摘要生成 ────────────────────────────────────────────────────────────


def generate_summary_for_paper(paper: dict, config: dict) -> dict:
    """为单篇论文生成 LLM 摘要。

    Args:
        paper:  论文字典
        config: 配置 {summary_type, language, model}

    Returns:
        PaperSummarizer.summarize() 的结果
    """
    summarizer = get_summarizer(model=config.get("model", "gpt-4o-mini"))
    title = paper.get("title", "")
    abstract = paper.get("abstract", paper.get("summary", ""))

    with st.spinner(f"🤖 Generating {config['summary_type']} summary..."):
        result = summarizer.summarize(
            title=title,
            abstract=abstract,
            summary_type=config["summary_type"],
            language=config["language"],
        )
    return result


# ── 主界面 ──────────────────────────────────────────────────────────────

# 标题
st.title("🧬 Biomedical Research Agent")
st.markdown(
    "AI-powered biomedical literature research assistant. "
    "Ask any biomedical question — I'll find relevant papers and summarize them."
)

# 侧边栏
config = render_sidebar()

# ── 渲染聊天历史 ──
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # 如果有搜索结果，展示论文列表
        if message.get("papers"):
            render_paper_list(message["papers"], max_display=config["max_results"])

        # 如果有摘要结果，展示摘要
        if message.get("summary"):
            render_summary_tabs(message["summary"])

# ── 聊天输入 ──
if prompt := st.chat_input("Ask a biomedical research question..."):

    # 1. 添加用户消息
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 搜索论文
    with st.chat_message("assistant"):
        sources = config["search_source"]
        if not sources:
            st.warning("Please select at least one search source in the sidebar.")
        else:
            with st.spinner(f"🔍 Searching {', '.join(sources)} for '{prompt}'..."):
                papers = search_papers(prompt, sources, config["max_results"])

            st.session_state.current_papers = papers

            if papers:
                response_text = (
                    f"Found **{len(papers)} papers** across {', '.join(sources)}. "
                    f"Click the expand buttons to read abstracts, or generate summaries below."
                )
            else:
                response_text = (
                    f"No papers found for '{prompt}'. "
                    f"Try different keywords or check your search sources."
                )

            st.markdown(response_text)
            render_paper_list(papers, max_display=config["max_results"])

            # 保存助手消息
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "papers": papers,
            })

    # 3. 自动为第一篇论文生成摘要（如果配置了）
    if st.session_state.current_papers and config["summary_type"] != "all":
        # 用户可选是否自动生成
        pass  # 让用户手动触发摘要


# ── 摘要触发区 ──
if st.session_state.current_papers:
    st.divider()
    st.subheader("📝 Generate Paper Summaries")

    col1, col2 = st.columns([2, 1])
    with col1:
        paper_options = [
            f"{i}. {p.get('title', 'Untitled')[:80]}"
            for i, p in enumerate(st.session_state.current_papers[:10], 1)
        ]
        selected_idx = st.selectbox(
            "Select a paper to summarize:",
            range(len(paper_options)),
            format_func=lambda i: paper_options[i],
            key="paper_selector",
        )
    with col2:
        st.caption("")  # spacer
        st.caption("")
        summarize_clicked = st.button(
            "🤖 Generate Summary",
            type="primary",
            use_container_width=True,
        )

    if summarize_clicked:
        paper = st.session_state.current_papers[selected_idx]
        with st.spinner("Generating summary..."):
            summary = generate_summary_for_paper(paper, config)

        st.success(f"Summary generated for: {paper.get('title', 'Untitled')[:100]}")
        render_summary_tabs(summary)

        # 添加到聊天历史
        title = paper.get("title", "Untitled")[:100]
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"📝 Summary for: **{title}**",
            "summary": summary,
        })


# ── 清除历史按钮 ──
with st.sidebar:
    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_papers = []
        st.rerun()


# ── 独立运行入口 ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Streamlit 通过 `streamlit run` 启动，此处仅作占位
    pass
