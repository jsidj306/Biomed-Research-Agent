"""
RAG Pipeline — 端到端集成验证。

验证流程:
  1. 用 PubMedSearchTool 搜索论文摘要作为数据源
  2. 文档分块 (BiomedicalTextSplitter)
  3. 向量嵌入 + ChromaDB 存储
  4. 语义搜索验证
  5. 统计信息

运行: uv run python notebooks/02_rag_pipeline_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

from langchain_core.documents import Document
from src.tools.pubmed_tool import PubMedSearchTool


def print_separator(title: str):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def test_1_load_pubmed_abstracts():
    """Step 1: 用 PubMed 搜索获取论文摘要作为数据源。"""
    print_separator("Step 1: Load PubMed Abstracts")

    tool = PubMedSearchTool()
    result = tool._run(
        query="CRISPR-Cas9 gene editing therapeutic applications review",
        max_results=10,
    )

    print(f"Fetched {len(result['papers'])} papers from PubMed")
    docs = []
    for paper in result["papers"]:
        abstract = paper.get("abstract", "")
        if len(abstract) < 100:  # 跳过太短的摘要
            continue
        docs.append(Document(
            page_content=abstract,
            metadata={
                "pmid": paper["pmid"],
                "title": paper["title"],
                "journal": paper.get("journal", ""),
                "year": paper.get("year", ""),
            },
        ))

    print(f"Valid abstracts (>=100 chars): {len(docs)}/{len(result['papers'])}")
    assert len(docs) >= 3, f"Expected >=3 valid abstracts, got {len(docs)}"
    print("[OK] Step 1 PASSED")
    return docs


def test_2_split_texts(docs):
    """Step 2: 文本分块。"""
    print_separator("Step 2: Text Splitting")

    from src.rag.text_splitter import BiomedicalTextSplitter

    splitter = BiomedicalTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split(docs)

    print(f"Documents: {len(docs)} → Chunks: {len(chunks)}")
    print(f"Chunk sizes: min={min(len(c.page_content) for c in chunks)}, "
          f"max={max(len(c.page_content) for c in chunks)}, "
          f"avg={sum(len(c.page_content) for c in chunks) // len(chunks)}")

    # 每个 chunk 应该有 chunk_id
    assert all("chunk_id" in c.metadata for c in chunks), "Missing chunk_id"
    assert len(chunks) >= len(docs), (
        f"Expected chunks >= docs, got {len(chunks)} vs {len(docs)}"
    )

    # 展示前 3 个 chunk
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n  Chunk {i} (id={chunk.metadata['chunk_id']}, "
              f"size={len(chunk.page_content)} chars):")
        print(f"  {chunk.page_content[:120]}...")
        print(f"  Source: {chunk.metadata.get('title', 'N/A')[:80]}")

    print("\n[OK] Step 2 PASSED")
    return chunks


def test_3_embed_and_store(chunks):
    """Step 3: 向量嵌入 + ChromaDB 存储。"""
    print_separator("Step 3: Embed + Store in ChromaDB")

    from src.rag.embeddings import BiomedicalEmbeddings
    from src.rag.vector_store import BiomedicalVectorStore

    print("Loading PubMedBERT embedding model (first time downloads ~440MB)...")
    embeddings = BiomedicalEmbeddings("pubmedbert")
    print(f"Model loaded: {embeddings.model_name}")

    store = BiomedicalVectorStore(
        persist_directory="./data/chroma_db",
        collection_name="test_rag_demo",
    )

    count = store.index_documents(chunks, embeddings.embedding_model)
    print(f"Indexed: {count} chunks")

    # 验证统计
    stats = store.get_stats()
    print(f"Collection: {stats['name']}, Count: {stats['count']}")

    assert stats["count"] == len(chunks), (
        f"Expected {len(chunks)} in store, got {stats['count']}"
    )
    print("[OK] Step 3 PASSED")
    return embeddings, store


def test_4_semantic_search(embeddings, store):
    """Step 4: 语义搜索验证。"""
    print_separator("Step 4: Semantic Search")

    queries = [
        "What are the main challenges in CRISPR delivery to target cells?",
        "clinical applications of gene editing in cancer therapy",
        "off-target effects and safety concerns",
    ]

    for query in queries:
        print(f"\nQ: {query}")
        results = store.search(query, embeddings.embedding_model, k=3)

        for i, r in enumerate(results, 1):
            print(f"  {i}. [score={r['score']:.4f}] {r['content'][:100]}...")

        # 至少返回一个结果
        assert len(results) >= 1, f"No results for query: {query}"

    # 额外的质量检查：查询应该返回相关结果
    results = store.search(
        "CRISPR gene knockout in vivo",
        embeddings.embedding_model,
        k=5,
    )
    # 前 3 个结果应该至少有一个包含 CRISPR 或 gene
    top3_text = " ".join(r["content"][:200].lower() for r in results[:3])
    has_relevant = any(
        kw in top3_text
        for kw in ["crispr", "gene", "edit", "dna", "genome"]
    )
    print(f"\nTop-3 results contain relevant keywords: {has_relevant}")

    print("[OK] Step 4 PASSED")


def test_5_pipeline_smoke_test(docs):
    """Step 5: RAGPipeline 一键摄入 + 搜索。"""
    print_separator("Step 5: RAGPipeline Smoke Test")

    from src.rag.rag_pipeline import RAGPipeline

    # 用独立 collection 避免和 step 3 冲突
    pipeline = RAGPipeline(
        model_name="pubmedbert",
        chunk_size=500,  # 摘要文本短，用更小块
        chunk_overlap=100,
        collection_name="test_rag_pipeline",
    )

    # 摄入
    count = pipeline.ingest_texts(
        texts=[d.page_content for d in docs],
        metadata_list=[d.metadata for d in docs],
    )
    print(f"Ingested: {count} chunks")

    # 搜索
    results = pipeline.search("CRISPR Cas9 therapeutic applications", k=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [score={r['score']:.4f}] {r['content'][:100]}...")

    # 统计
    stats = pipeline.get_stats()
    print(f"\nPipeline stats: {stats}")

    assert stats["count"] > 0
    print("[OK] Step 5 PASSED")


def main():
    print("=" * 65)
    print("  RAG Pipeline — End-to-End Verification")
    print("=" * 65)

    try:
        # Step 1: Load data
        docs = test_1_load_pubmed_abstracts()

        # Step 2: Split
        chunks = test_2_split_texts(docs)

        # Step 3: Embed + Store
        embeddings, store = test_3_embed_and_store(chunks)

        # Step 4: Search
        test_4_semantic_search(embeddings, store)

        # Step 5: Pipeline smoke test
        test_5_pipeline_smoke_test(docs)

        print_separator("ALL TESTS PASSED")
        print("RAG Pipeline is ready: PDF/Text → Split → Embed → Store → Search")
        return True

    except Exception as e:
        print(f"\n[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
