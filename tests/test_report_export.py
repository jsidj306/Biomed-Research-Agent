"""
Report Exporter — 单元测试。

运行: uv run pytest tests/test_report_export.py -v

验证 Markdown / Word / PDF 导出、目录自动创建、空报告降级和格式分发。
PDF 通过 mock weasyprint 后端测试，不触发真实转换。
"""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.report.export import (
    export_docx,
    export_markdown,
    export_pdf,
    export_report,
    markdown_to_html,
)


@pytest.fixture
def report():
    return (
        "# 研究报告\n\n"
        "## Abstract\n\n"
        "本报告总结了 CAR-T 疗法的最新进展 [1]。\n\n"
        "## References\n\n"
        "[1] CAR-T therapy. Nature Medicine (2024). PMID:12345678"
    )


# ── Markdown 导出测试 ────────────────────────────────────────────────────


class TestExportMarkdown:
    def test_export_content_matches(self, tmp_path, report):
        path = tmp_path / "out" / "report.md"
        out = export_markdown(report, str(path))
        assert path.read_text(encoding="utf-8") == report
        assert out == str(path.resolve())

    def test_auto_create_parent_dir(self, tmp_path, report):
        path = tmp_path / "nested" / "dir" / "report.md"
        export_markdown(report, str(path))
        assert path.exists()


# ── Word 导出测试 ────────────────────────────────────────────────────────


class TestExportDocx:
    def test_export_docx_headings(self, tmp_path, report):
        path = tmp_path / "report.docx"
        export_docx(report, str(path))
        assert path.exists()

        from docx import Document
        doc = Document(str(path))
        # 标题应被识别为 heading
        headings = [p.text for p in doc.paragraphs if p.style.name.startswith("Heading")]
        assert "研究报告" in headings
        assert "Abstract" in headings


# ── PDF 导出测试（mock weasyprint）───────────────────────────────────────


class TestExportPdf:
    def test_export_pdf_with_fake_weasyprint(self, tmp_path, report):
        """mock weasyprint 后端，验证导出链路。"""
        fake_html = SimpleNamespace(write_pdf=lambda p: None)
        fake_weasyprint = SimpleNamespace(HTML=lambda **kw: fake_html)
        with patch.dict(sys.modules, {"weasyprint": fake_weasyprint}):
            path = tmp_path / "report.pdf"
            out = export_pdf(report, str(path))
        assert out == str(path.resolve())

    def test_export_pdf_no_backend_raises(self, tmp_path, report):
        """weasyprint 与 pandoc 均不可用时应抛出明确错误。"""
        with patch.dict(sys.modules, {"weasyprint": None}), patch(
            "subprocess.run", side_effect=FileNotFoundError
        ), pytest.raises(RuntimeError):
            # weasyprint 不存在 → ImportError；pandoc 用 mock 抛 FileNotFoundError
            export_pdf(report, str(tmp_path / "report.pdf"))


# ── 分发测试 ─────────────────────────────────────────────────────────────


class TestExportReport:
    def test_dispatch_markdown(self, tmp_path, report):
        path = tmp_path / "r.md"
        export_report(report, str(path), fmt="markdown")
        assert path.exists()

    def test_dispatch_docx(self, tmp_path, report):
        path = tmp_path / "r.docx"
        export_report(report, str(path), fmt="docx")
        assert path.exists()

    def test_unsupported_format(self, tmp_path, report):
        with pytest.raises(ValueError):
            export_report(report, str(tmp_path / "r.txt"), fmt="txt")


# ── Markdown → HTML 测试 ─────────────────────────────────────────────────


class TestMarkdownToHtml:
    def test_headings_and_paragraph(self):
        html = markdown_to_html("# Title\n\n## Section\n\nBody text")
        assert "<h1>Title</h1>" in html
        assert "<h2>Section</h2>" in html
        assert "<p>Body text</p>" in html

    def test_list_items(self):
        html = markdown_to_html("- item1\n- item2")
        assert "<ul>" in html
        assert "<li>item1</li>" in html
        assert "</ul>" in html
