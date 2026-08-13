"""共享 pytest fixtures。

FakeEmbeddings 提供确定性的假嵌入（基于 zlib.crc32 词重叠），
用于需要向量存储/检索但不想下载真实模型的测试。
"""

import zlib

import pytest


class FakeEmbeddings:
    """确定性 Fake 嵌入。

    使用 zlib.crc32（跨进程稳定）而非 Python 内置 hash()（每进程随机），
    避免相似度排序在多次运行间抖动导致 flaky 测试。
    词重叠让「查询词命中的文档」相似度更高，保留一定的语义性。
    """

    _DIM = 384

    def _embed(self, text):
        vec = [0.0] * self._DIM
        for word in text.lower().split():
            vec[zlib.crc32(word.encode()) % self._DIM] = 1.0
        return vec

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


@pytest.fixture
def fake_emb():
    return FakeEmbeddings()


@pytest.fixture
def sample_biomedical_text():
    return (
        "CRISPR-Cas9 is a revolutionary gene-editing technology "
        "that allows precise modifications to DNA sequences. The system "
        "consists of a guide RNA and the Cas9 endonuclease."
    )
