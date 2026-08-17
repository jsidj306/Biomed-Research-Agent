"""
Writer Agent — 结构化研究报告生成 + BibTeX 导出。

把零散的摘要 / 评分 / 矛盾，整合成一篇学术级研究报告。报告采用 IMRaD 变体的
六段式结构，正文用 [n] 标记引用，References 段落由代码模板拼接（而非 LLM 生成），
保证引用 100% 真实、格式统一——这是「引用真实性」的最强保证。

六段式结构::

    Abstract / Methods / Findings / Gaps / Directions / References

关键设计:
  - 分节生成：报告很长，一次生成易超 token 或前后不一致，故逐节生成再汇总
  - References 用代码拼接：引用是最容易幻觉的地方，用论文元数据模板化拼接杜绝编造
  - 引用协议：正文 [1][2] ↔ References 编号一一对应，可被 Critic 确定性核查

使用示例:
    from src.agents.writer_agent import WriterAgent

    writer = WriterAgent()
    report = writer.write_report({"papers": papers, "summaries": summaries})
    bibtex = writer.build_bibtex(papers)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from dotenv import load_dotenv
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.llm_utils import create_chat_model

load_dotenv()

logger = logging.getLogger(__name__)

# ── 报告结构与引用协议 ───────────────────────────────────────────────────

REPORT_SECTIONS = [
    "Abstract", "Methods", "Findings", "Gaps", "Directions", "References",
]

# 引用格式：[编号] 标题. 作者. 期刊 (年份). PMID:xxx
CITATION_FMT = "[{i}] {title}. {authors}. {journal} ({year}). PMID:{pmid}"

WRITER_SYSTEM = (
    "你是生物医学报告撰写人，基于给定 evidence 撰写严谨的 {section} 章节。"
    "所有结论必须标注引用编号 [n]（n 对应 evidence 里的论文编号）。"
    "不要编造任何 evidence 里没有的数据、药物名或结论。"
)


# ── Writer 实现 ─────────────────────────────────────────────────────────


class WriterAgent:
    """生物医学研究报告撰写 Agent。

    分节生成六段式报告，References 用代码模板拼接保证引用真实。

    Attributes:
        model_name:  LLM 模型名
        temperature: 温度（0.3，报告略具可读性）
        llm:         ChatOpenAI 实例
    """

    def __init__(
        self,
        model_name: str = "glm-5.2",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ):
        """初始化 Writer。

        Args:
            model_name:  LLM 模型名
            temperature: 温度
            max_tokens:  最大输出 token 数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = create_chat_model(model_name, temperature, max_tokens)
        logger.info("WriterAgent ready: model=%s, temp=%.2f", model_name, temperature)

    # ── 公开接口 ────────────────────────────────────────────────────────

    def write_report(self, state: dict[str, Any]) -> str:
        """生成六段式研究报告。

        Args:
            state: 共享状态，需含:
                - papers: 论文列表（用于 References 拼接 + 作为 evidence）
                - summaries: 摘要结果（作为 evidence）
                可选: gaps（研究空白）、directions（研究方向）

        Returns:
            str — 完整的六段式 Markdown 报告
        """
        papers = state.get("papers", [])
        if not papers:
            logger.warning("No papers to write report from")
            return "（未检索到论文，无法生成研究报告）"

        sections: dict[str, str] = {}
        for section in REPORT_SECTIONS[:-1]:  # References 最后统一代码生成
            sections[section] = self._generate_section(section, state)
        sections["References"] = self.build_references(papers)

        return self._assemble(sections)

    @staticmethod
    def build_references(papers: list[dict[str, Any]]) -> str:
        """用代码模板拼接 References 段落（非 LLM，保证引用 100% 真实）。

        Args:
            papers: 论文列表（统一 schema）

        Returns:
            str — 编号引文列表，每行一条 [n] 格式
        """
        lines = []
        for i, paper in enumerate(papers, 1):
            authors = WriterAgent._format_authors(paper.get("authors") or [])
            title = paper.get("title", "")
            journal = paper.get("journal", "")
            year = paper.get("year", "")
            pmid = paper.get("pmid", "")
            url = paper.get("url", "")
            if pmid:
                url = url or f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            lines.append(
                CITATION_FMT.format(
                    i=i, title=title, authors=authors,
                    journal=journal or "未知期刊", year=year or "未知年份",
                    pmid=pmid or "未知",
                )
            )
            if url:
                lines.append(f"    {url}")
        return "\n".join(lines)

    @staticmethod
    def build_bibtex(papers: list[dict[str, Any]]) -> str:
        """把论文列表转成 BibTeX 条目（可被 Zotero / EndNote / LaTeX 引用）。

        Args:
            papers: 论文列表（统一 schema）

        Returns:
            str — 多个 @article 条目，空行分隔
        """
        entries = []
        for i, paper in enumerate(papers, 1):
            title = paper.get("title", "")
            author_str = " and ".join(paper.get("authors") or [])
            journal = paper.get("journal", "")
            year = paper.get("year", "")
            pmid = paper.get("pmid", "")
            doi = paper.get("doi", "")
            key = WriterAgent._bibtex_key(paper, i)

            entry = (
                f"@article{{{key},\n"
                f"  title = {{{title}}},\n"
                f"  author = {{{author_str}}},\n"
                f"  journal = {{{journal}}},\n"
                f"  year = {{{year}}},\n"
                f"  pmid = {{{pmid}}},\n"
                f"  doi = {{{doi}}}\n"
                "}"
            )
            entries.append(entry)
        return "\n\n".join(entries)

    @staticmethod
    def extract_citations(report: str) -> list[int]:
        """提取报告正文里引用的所有 [n] 编号（去重排序）。

        Args:
            report: 报告文本

        Returns:
            List[int] — 去重排序后的引用编号
        """
        return sorted({int(m) for m in re.findall(r"\[(\d+)\]", report)})

    # ── 内部实现 ────────────────────────────────────────────────────────

    def _generate_section(self, section: str, state: dict[str, Any]) -> str:
        """用 LLM 生成报告的单节内容。"""
        papers = state.get("papers", [])
        summaries = state.get("summaries", {}) or {}
        gaps = state.get("gaps", []) or []
        directions = state.get("directions", []) or []

        evidence = self._format_evidence(papers, summaries)
        prompt = (
            WRITER_SYSTEM.format(section=section)
            + f"\n\n要撰写的章节: {section}\n\n"
            + f"evidence（论文编号从 1 开始）:\n{evidence}\n\n"
            + self._section_extra(section, gaps, directions)
            + f"\n\n现在撰写 {section} 章节（直接输出 Markdown 正文，"
              "引用用 [n] 标记，不要输出 JSON）。"
        )
        content = self._call_llm(prompt).strip()
        # LLM 偶发在正文开头重复输出 "## Abstract" 之类的标题，与 _assemble 拼装时的
        # "## {section}" 撞车导致重复标题；剥掉开头的 Markdown 标题行。
        return self._strip_leading_heading(content)

    @staticmethod
    def _format_evidence(
        papers: list[dict[str, Any]],
        summaries: Any,
    ) -> str:
        """把论文 + 摘要格式化为 evidence 文本块。"""
        lines = []
        for i, paper in enumerate(papers, 1):
            title = paper.get("title") or f"Paper {i}"
            fulltext = paper.get("fulltext")
            if fulltext:
                abstract = fulltext[:6000]  # 本地文档全文，给更长上下文
            else:
                abstract = (paper.get("abstract") or "")[:500]
            lines.append(f"[{i}] {title}\n    {abstract}")
            # 若已有该论文的摘要结果，一并作为 evidence
            if isinstance(summaries, list) and i - 1 < len(summaries):
                s = summaries[i - 1]
                brief = (s or {}).get("brief", {}) or {}
                one_liner = brief.get("one_liner", "")
                if one_liner:
                    lines.append(f"    摘要: {one_liner}")
        return "\n".join(lines)

    @staticmethod
    def _section_extra(
        section: str,
        gaps: list[Any],
        directions: list[Any],
    ) -> str:
        """按章节注入额外上下文（Gaps / Directions 章节需要空白与方向）。"""
        if section == "Gaps" and gaps:
            gap_text = "\n".join(f"- {g}" for g in gaps)
            return f"\n已知研究空白:\n{gap_text}"
        if section == "Directions" and directions:
            dir_text = "\n".join(f"- {d}" for d in directions)
            return f"\n已有研究方向建议:\n{dir_text}"
        return ""

    @staticmethod
    def _strip_leading_heading(text: str) -> str:
        """剥掉正文开头的 Markdown 标题行，避免与 _assemble 拼装的标题重复。"""
        lines = text.splitlines()
        while lines and lines[0].lstrip().startswith("#"):
            lines.pop(0)
        return "\n".join(lines).strip()

    @staticmethod
    def _assemble(sections: dict[str, str]) -> str:
        """把六段内容拼装成 Markdown 报告。"""
        parts = []
        for section in REPORT_SECTIONS:
            content = sections.get(section, "")
            parts.append(f"## {section}\n\n{content}")
        return "\n\n".join(parts)

    @staticmethod
    def _format_authors(authors: list[str], max_authors: int = 3) -> str:
        """格式化作者列表（超长截断 + et al.）。"""
        authors = [a for a in authors if a]
        if not authors:
            return "未知作者"
        shown = authors[:max_authors]
        suffix = " et al." if len(authors) > max_authors else ""
        return ", ".join(shown) + suffix

    @staticmethod
    def _bibtex_key(paper: dict[str, Any], index: int) -> str:
        """生成 BibTeX 引用 key：第一作者姓氏 + 年份（无法解析时回退 refN）。"""
        authors = paper.get("authors") or []
        if authors and authors[0]:
            surname = authors[0].split()[-1]
            surname = re.sub(r"[^a-zA-Z0-9]", "", surname).lower()
        else:
            surname = f"ref{index}"
        year = str(paper.get("year") or "")
        return f"{surname}{year}" if year else surname

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _call_llm(self, prompt: str) -> str:
        """调用 LLM 并返回文本（3 次重试 + 指数退避）。"""
        response = self.llm.invoke(prompt)
        return response.content
