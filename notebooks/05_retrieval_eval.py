"""
第3-4周 检查项28：检索评测 — 实测 MRR / Recall@10。

用带标注的生物医学评测集（60 篇语料 + 40 条查询）实测混合检索效果，
并对「BM25-only / 语义-only / 混合(BM25+RRF)」做消融对比。

验收线（硬性）:
    MRR        > 0.70
    Recall@10  > 0.85

运行:
    uv run python notebooks/05_retrieval_eval.py

产物:
    data/biomed_eval_set.json — 语料 + 查询 + 标注 + 实测指标（可复现留档）。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from src.rag.eval_set import CORPUS, QUERIES, TfidfSemanticRanker, build_eval_set
from src.rag.hybrid_search import (
    HybridRetriever,
    mean_reciprocal_rank,
    recall_at_k,
)


def _indices(results: list[dict]) -> list[int]:
    """从 HybridRetriever 返回的结果中提取文档下标。"""
    return [r["index"] for r in results]


def main() -> None:
    corpus_texts = [f"{d['title']}. {d['abstract']}" for d in CORPUS]
    eval_set = build_eval_set()

    ranker = TfidfSemanticRanker(corpus_texts)
    retriever = HybridRetriever(corpus_texts)  # 不加载重排模型，纯 RRF 融合

    # 三种检索策略（同一评测集、同一语料）
    hybrid_fn = lambda q: _indices(retriever.search(q, ranker.rank, k=10))       # noqa: E731
    bm25_fn = lambda q: [i for i, _ in retriever.bm25.search(q, k=10)]           # noqa: E731
    semantic_fn = lambda q: ranker.rank(q, k=10)                                  # noqa: E731

    rows = {
        "BM25-only": (
            mean_reciprocal_rank(eval_set, bm25_fn),
            recall_at_k(eval_set, bm25_fn),
        ),
        "语义-only (TF-IDF)": (
            mean_reciprocal_rank(eval_set, semantic_fn),
            recall_at_k(eval_set, semantic_fn),
        ),
        "混合 (BM25+RRF)": (
            mean_reciprocal_rank(eval_set, hybrid_fn),
            recall_at_k(eval_set, hybrid_fn),
        ),
    }

    print("=" * 64)
    print("检索评测结果 — 40 条带标注查询 / 60 篇语料")
    print("=" * 64)
    print(f"{'策略':<22}{'MRR':>8}{'Recall@10':>12}")
    print("-" * 64)
    for name, (mrr, recall) in rows.items():
        flag = "  ✅ 达标" if mrr > 0.70 and recall > 0.85 else ""
        print(f"{name:<22}{mrr:>8.3f}{recall:>12.3f}{flag}")
    print("-" * 64)

    hybrid_mrr, hybrid_recall = rows["混合 (BM25+RRF)"]
    print(f"\n验收线: MRR > 0.70, Recall@10 > 0.85")
    print(
        f"结论: {'✅ 达标' if hybrid_mrr > 0.70 and hybrid_recall > 0.85 else '❌ 未达标，需调优'}"
    )

    # ── 留档：导出评测集 + 实测指标到 JSON ─────────────────────────────
    os.makedirs("data", exist_ok=True)
    payload = {
        "description": "生物医学检索评测集 — 60 篇语料 / 40 条带标注查询",
        "corpus": CORPUS,
        "queries": QUERIES,
        "metrics": {
            "mrr": hybrid_mrr,
            "recall_at_10": hybrid_recall,
            "bm25_mrr": rows["BM25-only"][0],
            "bm25_recall_at_10": rows["BM25-only"][1],
            "semantic_mrr": rows["语义-only (TF-IDF)"][0],
            "semantic_recall_at_10": rows["语义-only (TF-IDF)"][1],
        },
        "note": "语义路为 TF-IDF 余弦（保守下界），生产环境 PubMedBERT 嵌入只高不低。",
    }
    out = os.path.join("data", "biomed_eval_set.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n评测集已留档: {out}")


if __name__ == "__main__":
    main()
