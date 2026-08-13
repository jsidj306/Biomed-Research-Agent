"""
Biomedical Document Loader — PDF / TXT 文档加载。

支持格式: PDF (PyMuPDF), TXT (TextLoader)
统一返回 LangChain Document 对象，下游分块器可直接消费。

PyMuPDF 优势（相比 PyPDF2）:
- 速度快 10-50 倍（C 底层 MuPDF 引擎）
- 生物医学双栏排版解析更好
- 保留段落结构和字体信息
"""

import logging
import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader

logger = logging.getLogger(__name__)


class BiomedicalDocumentLoader:
    """生物医学文档加载器。

    自动识别文件类型（.pdf / .txt），统一加载为 List[Document]。

    Attributes:
        SUPPORTED_EXTENSIONS: 支持的文件扩展名集合
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

    def load_pdf(self, file_path: str) -> List[Document]:
        """加载单个 PDF 文件，每页为一个 Document。

        Args:
            file_path: PDF 文件路径

        Returns:
            List[Document] — 每个 Document 对应一页，metadata 含 source/page 等
        """
        logger.info("Loading PDF: %s", file_path)
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        logger.info("Loaded %d pages from PDF", len(documents))
        return documents

    def load_text(self, file_path: str) -> List[Document]:
        """加载纯文本文件。

        Args:
            file_path: TXT 文件路径

        Returns:
            List[Document] — 整个文件作为一个 Document（或按段落拆分）
        """
        logger.info("Loading text: %s", file_path)
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()

    def load(self, file_path: str) -> List[Document]:
        """自动识别文件类型并加载。

        Args:
            file_path: 文件路径（.pdf 或 .txt）

        Returns:
            List[Document] — 加载并解析后的文档列表

        Raises:
            ValueError: 文件扩展名不支持
            FileNotFoundError: 文件不存在
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext == ".pdf":
            return self.load_pdf(str(path))
        elif ext == ".txt":
            return self.load_text(str(path))
        else:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )
