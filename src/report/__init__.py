"""Biomedical Research Agent — Report export package.

研究报告导出：Markdown / PDF / Word。让系统「从聊天框变成生产力工具」——
用户能把结果带走，用于写综述、做汇报。

组件:
- export_markdown: 导出 .md（基础，也是 PDF/Word 的中间格式）
- export_docx:     导出 .docx（python-docx，保留标题层级）
- export_pdf:      导出 .pdf（weasyprint 或 pandoc，中文友好）
- export_report:   按格式分发的统一入口
- markdown_to_html: Markdown → HTML（纯函数，供 PDF 路径使用）
"""

from src.report.export import (
    export_docx,
    export_markdown,
    export_pdf,
    export_report,
    markdown_to_html,
)

__all__ = [
    "export_docx",
    "export_markdown",
    "export_pdf",
    "export_report",
    "markdown_to_html",
]
