"""
Fulltext Loader — PDF 全文提取（PyMuPDF 主 + Unstructured 兜底）。

MVP 阶段的 BiomedicalDocumentLoader 用 LangChain PyMuPDFLoader 逐页加载，但
V2.0 需要「全文级」RAG——能读整篇论文的方法/结果/讨论，而非只有摘要。本模块
提供直接的全文字符串提取，并处理最常见的坑：扫描版 PDF（图片型）提取为空。

提取策略:
  1. PyMuPDF (fitz) 快速提取文字层
  2. 文字过短（疑似扫描版）→ 尝试 Unstructured.io 版面感知解析
  3. 仍然为空 → 明确报错提示「请提供文字版 PDF 或启用 OCR」

使用示例:
    from src.rag.fulltext_loader import FulltextLoader

    loader = FulltextLoader()
    docs = loader.load("data/papers/fulltext.pdf")
    print(len(docs[0].page_content), "字符")
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# 全文视为「有内容」的最小字符数（低于此值视为扫描版/空 PDF）
MIN_TEXT_LENGTH = 100


class FulltextLoader:
    """生物医学 PDF 全文提取器。

    PyMuPDF 主提取 + Unstructured 兜底 + 扫描版降级提示。

    Attributes:
        min_text_length: 判定「无有效文本」的字符阈值
    """

    def __init__(self, min_text_length: int = MIN_TEXT_LENGTH):
        """初始化全文加载器。"""
        self.min_text_length = min_text_length

    # ── 公开接口 ────────────────────────────────────────────────────────

    def extract_text(self, pdf_path: str) -> str:
        """用 PyMuPDF 提取 PDF 全文（每页文本用换行拼接）。

        Args:
            pdf_path: PDF 文件路径

        Returns:
            str — 全文文本

        Raises:
            FileNotFoundError: 文件不存在
        """
        import fitz  # PyMuPDF，延迟导入

        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")

        doc = fitz.open(str(path))
        try:
            text = "\n".join(page.get_text() for page in doc)
        finally:
            doc.close()
        logger.info("PyMuPDF extracted %d chars from %s", len(text), pdf_path)
        return text

    def load(self, pdf_path: str) -> list[Document]:
        """加载 PDF 全文为单个 Document。

        提取策略: PyMuPDF → （过短时）Unstructured 兜底 → 仍空则报错。

        Args:
            pdf_path: PDF 文件路径

        Returns:
            List[Document] — 单个 Document，page_content 为全文

        Raises:
            ValueError: 扫描版/无文本层 PDF（提示需 OCR）
        """
        text = self.extract_text(pdf_path)

        if len(text.strip()) < self.min_text_length:
            logger.info("PyMuPDF text too short, trying Unstructured fallback")
            fallback = self._try_unstructured(pdf_path)
            if fallback:
                text = fallback

        if len(text.strip()) < 10:
            raise ValueError(
                "无法从 PDF 提取文本（可能是扫描版/图片型 PDF）。"
                "请提供文字版 PDF，或启用 OCR（如 Tesseract）后再试。"
            )

        return [Document(page_content=text, metadata={"source": str(Path(pdf_path).resolve())})]

    # ── 内部实现 ────────────────────────────────────────────────────────

    def _try_unstructured(self, pdf_path: str) -> str:
        """尝试用 Unstructured.io 版面感知解析（未安装时降级为空串）。

        Unstructured 擅长识别双栏、标题、表格，适合排版复杂的论文。
        延迟导入，未安装不影响主流程。
        """
        try:
            from unstructured.partition.pdf import partition_pdf  # 延迟导入
        except ImportError:
            logger.warning("Unstructured not installed, skipping fallback")
            return ""

        try:
            elements = partition_pdf(pdf_path)
            text = "\n".join(str(el) for el in elements)
            logger.info("Unstructured extracted %d chars", len(text))
            return text
        except Exception as e:
            logger.warning("Unstructured fallback failed: %s", e)
            return ""
