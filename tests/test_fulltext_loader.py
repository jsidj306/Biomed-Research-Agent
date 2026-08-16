"""
Fulltext Loader — 单元测试。

运行: uv run pytest tests/test_fulltext_loader.py -v

通过 mock PyMuPDF (fitz) 和 Unstructured，验证全文提取、扫描版降级、
Unstructured 兜底和文件不存在处理，不触发真实 PDF 解析。
"""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.rag.fulltext_loader import FulltextLoader

# ── fake fitz ─────────────────────────────────────────────────────────────


class _FakePage:
    def __init__(self, text: str):
        self._text = text

    def get_text(self) -> str:
        return self._text


class _FakeDoc:
    def __init__(self, pages: list[str]):
        self._pages = [_FakePage(p) for p in pages]

    def __iter__(self):
        return iter(self._pages)

    def close(self):
        pass


class _FakeFitz:
    """可迭代页面的假 PyMuPDF 模块。"""

    def __init__(self, pages: list[str]):
        self.pages = pages

    def open(self, path):
        return _FakeDoc(self.pages)


def _fake_unstructured(text: str) -> dict[str, object]:
    """构造假的 unstructured 模块链，partition_pdf 返回固定元素。"""
    fake_pdf = SimpleNamespace(partition_pdf=lambda path: [text])
    fake_partition = SimpleNamespace(pdf=fake_pdf)
    fake_unstructured = SimpleNamespace(partition=fake_partition)
    return {
        "unstructured": fake_unstructured,
        "unstructured.partition": fake_partition,
        "unstructured.partition.pdf": fake_pdf,
    }


# ── 提取测试 ─────────────────────────────────────────────────────────────


class TestExtractText:
    """测试 PyMuPDF 全文提取。"""

    def test_extract_text_joins_pages(self, tmp_path):
        pdf = tmp_path / "dummy.pdf"
        pdf.write_text("placeholder", encoding="utf-8")

        fake_fitz = _FakeFitz(["Hello page one", "Page two content"])
        with patch.dict(sys.modules, {"fitz": fake_fitz}):
            text = FulltextLoader().extract_text(str(pdf))
        assert text == "Hello page one\nPage two content"

    def test_load_returns_single_document(self, tmp_path):
        pdf = tmp_path / "dummy.pdf"
        pdf.write_text("placeholder", encoding="utf-8")

        long_text = "Introduction. " * 30 + " Methods " * 30
        fake_fitz = _FakeFitz([long_text])
        with patch.dict(sys.modules, {"fitz": fake_fitz}):
            docs = FulltextLoader().load(str(pdf))
        assert len(docs) == 1
        assert docs[0].page_content == long_text

    def test_missing_file_raises(self, tmp_path):
        loader = FulltextLoader()
        with pytest.raises(FileNotFoundError):
            loader.extract_text(str(tmp_path / "nonexistent.pdf"))


# ── 扫描版 / 兜底测试 ────────────────────────────────────────────────────


class TestDegradation:
    """测试扫描版 PDF 与 Unstructured 兜底。"""

    def test_scanned_pdf_raises(self, tmp_path):
        """无文本层 PDF（提取为空）应报错提示需 OCR。"""
        pdf = tmp_path / "scanned.pdf"
        pdf.write_text("placeholder", encoding="utf-8")

        fake_fitz = _FakeFitz([""])
        with patch.dict(sys.modules, {"fitz": fake_fitz}):
            loader = FulltextLoader()
            with pytest.raises(ValueError):
                loader.load(str(pdf))

    def test_unstructured_fallback_used(self, tmp_path):
        """PyMuPDF 提取为空时，Unstructured 兜底应被使用。"""
        pdf = tmp_path / "complex.pdf"
        pdf.write_text("placeholder", encoding="utf-8")

        fake_fitz = _FakeFitz([""])
        modules = {"fitz": fake_fitz, **_fake_unstructured("Full text from complex layout")}
        with patch.dict(sys.modules, modules):
            docs = FulltextLoader().load(str(pdf))
        assert "Full text from complex layout" in docs[0].page_content
