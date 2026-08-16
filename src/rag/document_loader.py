"""
Biomedical Document Loader — PDF / TXT 文档加载。

支持格式: PDF (PyMuPDF), TXT (纯文本)
统一返回 LangChain Document 对象，下游分块器可直接消费。

V2.0 起不再依赖 langchain-community（官方已 sunset）的 PyMuPDFLoader /
TextLoader，改用项目直接依赖的 PyMuPDF (fitz) 逐页提取 + 原生文本读取：
  - 消除 DeprecationWarning
  - 去掉一个 33MB 的过渡依赖
  - 行为与原实现一致（PDF 每页一个 Document、TXT 整文件一个 Document）
"""

import logging
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class BiomedicalDocumentLoader:
    """生物医学文档加载器。

    自动识别文件类型（.pdf / .txt），统一加载为 List[Document]。

    Attributes:
        SUPPORTED_EXTENSIONS: 支持的文件扩展名集合
    """

    SUPPORTED_EXTENSIONS = frozenset({".pdf", ".txt"})

    def load_pdf(self, file_path: str) -> list[Document]:
        """加载单个 PDF 文件，每页为一个 Document。

        Args:
            file_path: PDF 文件路径

        Returns:
            List[Document] — 每个 Document 对应一页，metadata 含 source/page
        """
        import fitz  # PyMuPDF，延迟导入

        logger.info("Loading PDF: %s", file_path)
        path = Path(file_path)
        doc = fitz.open(str(path))
        try:
            docs = [
                Document(
                    page_content=page.get_text(),
                    metadata={"source": str(path.resolve()), "page": index},
                )
                for index, page in enumerate(doc)
            ]
        finally:
            doc.close()
        logger.info("Loaded %d pages from PDF", len(docs))
        return docs

    def load_text(self, file_path: str) -> list[Document]:
        """加载纯文本文件。

        Args:
            file_path: TXT 文件路径

        Returns:
            List[Document] — 整个文件作为一个 Document
        """
        logger.info("Loading text: %s", file_path)
        path = Path(file_path)
        text = path.read_text(encoding="utf-8")
        return [Document(page_content=text, metadata={"source": str(path.resolve())})]

    def load(self, file_path: str) -> list[Document]:
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
