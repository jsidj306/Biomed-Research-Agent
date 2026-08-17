"""
UI 国际化（i18n）— 中英文文案字典与翻译辅助。

把所有用户可见文案抽到 `TEXT` 字典（zh / en），UI 通过 `t(lang, key)` 取值，
语言由 `st.session_state.lang` 控制。设计为纯函数，便于单元测试。

使用示例:
    from src.ui.i18n import t, SUPPORTED_LANGUAGES

    t("zh", "title")          # '🧬 生物医学研究助手'
    t("en", "title")          # '🧬 Biomedical Research Agent'
"""

from __future__ import annotations

SUPPORTED_LANGUAGES = ("zh", "en")

# 中英文文案字典。key 为语义标识，value 为对应语言文案。
TEXT: dict[str, dict[str, str]] = {
    "zh": {
        # 页面
        "title": "🧬 生物医学研究助手",
        "subtitle": (
            "AI 驱动的生物医学文献研究助手。输入任意生物医学问题——"
            "我会检索相关论文并生成结构化摘要。"
        ),
        "chat_input": "输入一个生物医学研究问题…",
        "select_source_warning": "请在侧边栏至少选择一个检索数据源。",
        "searching": "🔍 正在 {sources} 检索「{query}」…",
        "found_papers": "在 {sources} 找到 **{count} 篇论文**。点击展开阅读摘要，或在下方生成摘要。",
        "no_papers": "未找到关于「{query}」的论文。换个关键词，或检查检索数据源。",
        "search_error": "搜索 {source} 失败：{error}",
        "generate_summaries": "📝 生成论文摘要",
        "select_paper": "选择要摘要的论文：",
        "generate_summary_btn": "🤖 生成摘要",
        "generating_summary": "生成摘要中…",
        "summary_generated": "已为以下论文生成摘要：{title}",
        "summary_for": "📝 摘要：**{title}**",
        "clear_history_btn": "🗑️ 清空聊天历史",
        # 侧边栏
        "config": "⚙️ 配置",
        "ui_language": "🌐 界面语言",
        "search_sources": "检索数据源",
        "search_sources_help": "选择要检索的数据库",
        "max_results": "每源最大结果数",
        "max_results_help": "每个数据源的最大论文数",
        "summary_type": "摘要类型",
        "summary_type_help": "要生成的摘要类型",
        "summary_type_all": "全部类型",
        "summary_type_brief": "简短（一句话）",
        "summary_type_detailed": "详细（四段式）",
        "summary_type_key": "关键发现",
        "output_language": "输出语言",
        "llm_model": "LLM 模型",
        "llm_model_help": "glm-5.2（智谱）默认；kimi-k2.7-code（月之暗面）交叉验证",
        "footer_version": "Biomed Research Agent v0.1.0",
        "footer_built": "基于 LangChain + Streamlit 构建",
        # 论文卡片
        "untitled": "无标题",
        "unknown_author": "未知作者",
        "no_abstract": "无摘要。",
        "no_papers_found": "未找到论文。",
        "found_n_papers": "**找到 {count} 篇论文** — 显示前 {shown} 篇",
        # 摘要标签
        "brief_tab": "💡 简要",
        "detailed_tab": "📋 详细",
        "key_findings_tab": "🔑 关键发现",
        "background": "📋 背景",
        "methods": "🔬 方法",
        "results": "📊 结果",
        "conclusions": "💡 结论",
        "significance": "📊 **意义：** {text}",
        "no_summary": "未生成摘要。",
        # 报告
        "no_report": "暂无报告可展示。",
    },
    "en": {
        # 页面
        "title": "🧬 Biomedical Research Agent",
        "subtitle": (
            "AI-powered biomedical literature research assistant. "
            "Ask any biomedical question — I'll find relevant papers and summarize them."
        ),
        "chat_input": "Ask a biomedical research question...",
        "select_source_warning": "Please select at least one search source in the sidebar.",
        "searching": "🔍 Searching {sources} for '{query}'...",
        "found_papers": "Found **{count} papers** across {sources}. Click the expand buttons to read abstracts, or generate summaries below.",
        "no_papers": "No papers found for '{query}'. Try different keywords or check your search sources.",
        "search_error": "Search failed for {source}: {error}",
        "generate_summaries": "📝 Generate Paper Summaries",
        "select_paper": "Select a paper to summarize:",
        "generate_summary_btn": "🤖 Generate Summary",
        "generating_summary": "Generating summary...",
        "summary_generated": "Summary generated for: {title}",
        "summary_for": "📝 Summary for: **{title}**",
        "clear_history_btn": "🗑️ Clear Chat History",
        # 侧边栏
        "config": "⚙️ Configuration",
        "ui_language": "🌐 UI Language",
        "search_sources": "Search Sources",
        "search_sources_help": "Select which databases to search",
        "max_results": "Max Results per Source",
        "max_results_help": "Maximum number of papers per source",
        "summary_type": "Summary Type",
        "summary_type_help": "Type of summary to generate",
        "summary_type_all": "All Types",
        "summary_type_brief": "Brief (1 sentence)",
        "summary_type_detailed": "Detailed (4 sections)",
        "summary_type_key": "Key Findings",
        "output_language": "Output Language",
        "llm_model": "LLM Model",
        "llm_model_help": "glm-5.2 (智谱) default; kimi-k2.7-code (月之暗面) cross-validation",
        "footer_version": "Biomed Research Agent v0.1.0",
        "footer_built": "Built with LangChain + Streamlit",
        # 论文卡片
        "untitled": "Untitled",
        "unknown_author": "Unknown",
        "no_abstract": "No abstract available.",
        "no_papers_found": "No papers found.",
        "found_n_papers": "**Found {count} papers** — showing top {shown}",
        # 摘要标签
        "brief_tab": "💡 Brief",
        "detailed_tab": "📋 Detailed",
        "key_findings_tab": "🔑 Key Findings",
        "background": "📋 Background",
        "methods": "🔬 Methods",
        "results": "📊 Results",
        "conclusions": "💡 Conclusions",
        "significance": "📊 **Significance:** {text}",
        "no_summary": "No summary generated.",
        # 报告
        "no_report": "No report to display.",
    },
}


def t(lang: str, key: str, **kwargs) -> str:
    """取指定语言的文案并做占位符替换。

    Args:
        lang:  语言代码（zh / en），非法值回退到 en
        key:   文案键
        kwargs: 占位符替换参数

    Returns:
        str — 渲染后的文案；键不存在时回退 en，仍不存在则返回键名本身
    """
    table = TEXT.get(lang, TEXT["en"])
    template = table.get(key, TEXT["en"].get(key, key))
    return template.format(**kwargs) if kwargs else template
