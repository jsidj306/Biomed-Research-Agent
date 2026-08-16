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
import os
import shutil
import subprocess
import tempfile
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


def _resolve_tool(name: str, *explicit: Path) -> str | None:
    """定位命令行工具：显式候选路径优先，PATH（shutil.which）兜底。

    Args:
        name:     工具名（用于 shutil.which）
        explicit: 已知安装位置的候选路径，按优先级排列

    Returns:
        str | None — 首个存在的完整路径，全无则 None
    """
    for p in explicit:
        if p and p.exists():
            return str(p)
    return shutil.which(name)


def _find_pandoc() -> str | None:
    """定位 pandoc，优先 %LOCALAPPDATA%\\Pandoc（3.x，支持 typst 输出）。

    Anaconda 自带的 pandoc 1.19 常年在 PATH 里抢先且太老（无 typst 支持），
    所以显式优先 winget/MSI 的标准安装位置，绕开 PATH 顺序陷阱。
    """
    local = os.environ.get("LOCALAPPDATA", "")
    return _resolve_tool("pandoc", Path(local) / "Pandoc" / "pandoc.exe")


def _find_typst() -> str | None:
    """定位 typst 编译器：PATH 别名优先，winget 包目录 glob 兜底。"""
    found = shutil.which("typst")
    if found:
        return found
    local = os.environ.get("LOCALAPPDATA", "")
    base = Path(local) / "Microsoft" / "WinGet" / "Packages"
    matches = sorted(base.glob("Typst.Typst_*/**/typst.exe"))
    return str(matches[-1]) if matches else None


def _export_via_typst(pandoc: str, typst: str, report: str, path: str) -> None:
    """pandoc 先把 Markdown 转 Typst 标记，再用 typst 编译成 PDF。

    两步走 + 显式完整路径：既不要求 typst 在 pandoc 的 PATH 上，也不受
    PATH 里 Anaconda 旧 pandoc 抢先的影响。
    """
    with tempfile.NamedTemporaryFile(
        "w", suffix=".typ", delete=False, encoding="utf-8"
    ) as tmp:
        typ_path = tmp.name
    try:
        subprocess.run(
            [pandoc, "-f", "markdown", "-t", "typst", "-o", typ_path],
            input=report.encode("utf-8"),
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [typst, "compile", typ_path, str(path)],
            check=True,
            capture_output=True,
        )
        logger.info("Exported PDF (pandoc + typst) to %s", path)
    finally:
        Path(typ_path).unlink(missing_ok=True)


def export_pdf(report: str, path: str) -> str:
    """导出 PDF（weasyprint 首选，pandoc+typst 兜底，再退 pandoc 默认引擎）。

    Args:
        report: 报告 Markdown 文本
        path:   输出文件路径

    Returns:
        str — 输出的绝对路径

    Raises:
        RuntimeError: 所有 PDF 后端均不可用时
    """
    _ensure_parent(path)
    html = markdown_to_html(report)

    # 首选 weasyprint（中文支持好，纯 Python）
    try:
        from weasyprint import HTML
        HTML(string=html).write_pdf(str(path))
        logger.info("Exported PDF (weasyprint) to %s", path)
        return str(Path(path).resolve())
    except Exception:
        # 未安装，或装了但缺 GTK/Pango 原生库（Windows 常见，抛 OSError 而非
        # ImportError）→ 兜底到 pandoc 系，保证 PDF 导出不因首选后端失败而中断
        logger.info("weasyprint unavailable, falling back to pandoc")

    pandoc = _find_pandoc()

    # 兜底 1：pandoc + typst（轻量，无需 TeX 发行版）
    typst = _find_typst()
    if pandoc and typst:
        try:
            _export_via_typst(pandoc, typst, report, path)
            return str(Path(path).resolve())
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logger.info("pandoc + typst failed: %s", e)

    # 兜底 2：pandoc 默认引擎（LaTeX，需 TeX 发行版）
    if pandoc:
        try:
            subprocess.run(
                [pandoc, "-f", "markdown", "-t", "pdf", "-o", str(path)],
                input=report.encode("utf-8"),
                check=True,
                capture_output=True,
            )
            logger.info("Exported PDF (pandoc default) to %s", path)
            return str(Path(path).resolve())
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logger.info("pandoc default failed: %s", e)

    raise RuntimeError(
        "PDF 导出需要以下任一后端：weasyprint（`uv add weasyprint`）、"
        "pandoc + typst（`winget install JohnMacFarlane.Pandoc Typst.Typst`）、"
        "或 pandoc + LaTeX（如 MiKTeX）。请安装其一后再试。"
    )


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
