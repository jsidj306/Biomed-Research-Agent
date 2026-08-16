"""
Report Exporter — 研究报告导出 Markdown / PDF / Word。

Writer 产出的报告本身是 Markdown 文本，导出是「最后一公里」：
  - Markdown：零成本直接写文件，也是 PDF/Word 的中间格式
  - Word (docx)：python-docx 转写，保留标题层级与引用
  - PDF：weasyprint（首选，中文友好）或 pandoc（CLI）转 HTML→PDF

设计原则：
  - 所有导出路径自动创建父目录
  - PDF/Word 后端延迟导入，未安装时给出明确提示而非崩溃

使用示例:
    from src.report.export import export_report

    export_report(report, "out/report.md", fmt="markdown")
    export_report(report, "out/report.docx", fmt="docx")
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def markdown_to_html(md: str) -> str:
    """把 Markdown 文本转成 HTML（纯函数，覆盖标题/段落/列表/粗体）。

    Args:
        md: Markdown 文本

    Returns:
        str — HTML 字符串
    """
    lines: list[str] = []
    in_list = False
    for raw in md.splitlines():
        line = raw.rstrip()
        if line.startswith("## "):
            if in_list:
                lines.append("</ul>")
                in_list = False
            lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("# "):
            if in_list:
                lines.append("</ul>")
                in_list = False
            lines.append(f"<h1>{line[2:]}</h1>")
        elif line.strip().startswith("- "):
            if not in_list:
                lines.append("<ul>")
                in_list = True
            lines.append(f"<li>{line.strip()[2:]}</li>")
        elif line.strip() == "":
            if in_list:
                lines.append("</ul>")
                in_list = False
        else:
            if in_list:
                lines.append("</ul>")
                in_list = False
            lines.append(f"<p>{line}</p>")
    if in_list:
        lines.append("</ul>")
    return "\n".join(lines)


def export_markdown(report: str, path: str) -> str:
    """导出 Markdown 文件（自动创建父目录）。

    Args:
        report: 报告 Markdown 文本
        path:   输出文件路径

    Returns:
        str — 输出的绝对路径
    """
    _ensure_parent(path)
    Path(path).write_text(report, encoding="utf-8")
    logger.info("Exported Markdown to %s", path)
    return str(Path(path).resolve())


def export_docx(report: str, path: str) -> str:
    """导出 Word (.docx)，保留标题层级（python-docx）。

    Args:
        report: 报告 Markdown 文本
        path:   输出文件路径

    Returns:
        str — 输出的绝对路径
    """
    from docx import Document  # 延迟导入

    _ensure_parent(path)
    doc = Document()
    for raw in report.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("## "):
            doc.add_heading(line[3:], level=2)
        elif line.startswith("# "):
            doc.add_heading(line[2:], level=1)
        else:
            doc.add_paragraph(line)
    doc.save(str(path))
    logger.info("Exported DOCX to %s", path)
    return str(Path(path).resolve())


def export_pdf(report: str, path: str) -> str:
    """导出 PDF（weasyprint 首选，pandoc 兜底；中文友好）。

    Args:
        report: 报告 Markdown 文本
        path:   输出文件路径

    Returns:
        str — 输出的绝对路径

    Raises:
        RuntimeError: weasyprint 与 pandoc 均不可用时
    """
    _ensure_parent(path)
    html = markdown_to_html(report)

    # 首选 weasyprint（中文支持好，纯 Python）
    try:
        from weasyprint import HTML
        HTML(string=html).write_pdf(str(path))
        logger.info("Exported PDF (weasyprint) to %s", path)
        return str(Path(path).resolve())
    except ImportError:
        logger.info("weasyprint not installed, falling back to pandoc")

    # 兜底 pandoc（本地 CLI）
    try:
        subprocess.run(
            ["pandoc", "-f", "markdown", "-t", "pdf", "-o", str(path)],
            input=report.encode("utf-8"),
            check=True,
            capture_output=True,
        )
        logger.info("Exported PDF (pandoc) to %s", path)
        return str(Path(path).resolve())
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            "PDF 导出需要 weasyprint（`uv add weasyprint`）或 pandoc（本地安装）。"
            "两者均不可用，请安装其一后再试。"
        ) from e


def export_report(report: str, path: str, fmt: str = "markdown") -> str:
    """按格式分发导出（markdown / pdf / docx）。

    Args:
        report: 报告 Markdown 文本
        path:   输出文件路径
        fmt:    格式（markdown / pdf / docx）

    Returns:
        str — 输出的绝对路径
    """
    fmt = fmt.lower().lstrip(".")
    if fmt in ("md", "markdown"):
        return export_markdown(report, path)
    if fmt in ("pdf",):
        return export_pdf(report, path)
    if fmt in ("docx", "word"):
        return export_docx(report, path)
    raise ValueError(f"Unsupported format '{fmt}'. Use markdown/pdf/docx.")


def _ensure_parent(path: str) -> None:
    """确保输出文件父目录存在。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
