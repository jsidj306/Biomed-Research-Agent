"""Tests for BiomedicalDocumentLoader."""

import os
import tempfile

import pytest

from src.rag.document_loader import BiomedicalDocumentLoader


class TestBiomedicalDocumentLoader:
    """文档加载器测试。"""

    def test_load_txt_file(self):
        """加载 TXT 文件 — 应返回至少一个 Document。"""
        loader = BiomedicalDocumentLoader()

        # 创建临时 TXT 文件
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("This is a biomedical test document.\nIt has multiple lines.\nCRISPR is a gene editing tool.")
            tmp_path = f.name

        try:
            docs = loader.load(tmp_path)
            assert len(docs) >= 1
            assert "CRISPR" in docs[0].page_content
            assert "source" in docs[0].metadata
        finally:
            os.unlink(tmp_path)

    def test_load_nonexistent_file_raises(self):
        """加载不存在的文件 — 应抛出 FileNotFoundError。"""
        loader = BiomedicalDocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/paper.pdf")

    def test_unsupported_extension_raises(self):
        """不支持的扩展名 — 应抛出 ValueError。"""
        loader = BiomedicalDocumentLoader()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".docx", delete=False, encoding="utf-8"
        ) as f:
            f.write("test")
            tmp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported"):
                loader.load(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_supported_extensions(self):
        """应包含 .pdf 和 .txt。"""
        assert ".pdf" in BiomedicalDocumentLoader.SUPPORTED_EXTENSIONS
        assert ".txt" in BiomedicalDocumentLoader.SUPPORTED_EXTENSIONS
