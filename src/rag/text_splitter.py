"""
Biomedical Text Splitter — 学术论文优化的递归分块策略。

使用 LangChain RecursiveCharacterTextSplitter，针对生物医学论文特点调参:
- 长句子多 → overlap 略大（200 字）
- 段落是自然语义边界 → 最高优先级分隔符
- 实验方法/统计部分 → chunk_size 不宜太小（1000 字 ≈ 400 token）

分隔符优先级（从高到低）:
  1. 段落分隔 "\n\n"
  2. 行分隔 "\n"
  3. 句子分隔 ". "
  4. 单词分隔 " "
  5. 字符硬切 ""
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class BiomedicalTextSplitter:
    """生物医学文本分块器，针对学术论文优化。

    推荐配置:
        - 论文全文: chunk_size=1500, overlap=300
        - 摘要模式: chunk_size=500,  overlap=100
        - 速览模式: chunk_size=3000, overlap=400
        - MVP 默认:  chunk_size=1000, overlap=200
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        """初始化分块器。

        Args:
            chunk_size:     每个文本块的最大字符数
            chunk_overlap:  相邻块的重叠字符数
            separators:     分隔符优先级列表
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n",   # 段落（最高优先级）
            "\n",     # 行
            ". ",     # 句子
            " ",      # 单词
            "",       # 字符（最低优先级）
        ]

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
        )

    def split(self, documents: List[Document]) -> List[Document]:
        """对文档列表分块，保留原始元数据并添加块序号。

        Args:
            documents: 原始文档列表（来自 DocumentLoader）

        Returns:
            List[Document] — 分块后的文档列表，每个 chunk 的 metadata 额外包含:
                - chunk_id: 块序号
                - chunk_size: 块实际字符数
        """
        chunks = self._splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        logger.info(
            "Split %d docs → %d chunks (chunk_size=%d, overlap=%d)",
            len(documents), len(chunks),
            self.chunk_size, self.chunk_overlap,
        )

        # 质量检查
        self._validate_chunks(chunks)

        return chunks

    def _validate_chunks(self, chunks: List[Document]):
        """内部质量检查：确保没有空块，大部分块大小合理。"""
        empty = [c for c in chunks if len(c.page_content.strip()) == 0]
        if empty:
            logger.warning("Found %d empty chunks", len(empty))

        oversized = [c for c in chunks if len(c.page_content) > self.chunk_size * 1.5]
        if oversized:
            logger.warning(
                "Found %d oversized chunks (>%.0f chars)",
                len(oversized), self.chunk_size * 1.5,
            )
