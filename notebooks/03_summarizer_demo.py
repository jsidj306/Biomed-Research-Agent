"""
Paper Summarizer — 端到端集成验证 Demo。

验证流程:
  1. 用 PubMedSearchTool 搜索论文
  2. PaperSummarizer 生成三种类型的摘要
  3. 批量摘要 + 中文输出
  4. 与 Arxiv 搜索结果串联

运行: uv run python notebooks/03_summarizer_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

from src.tools.pubmed_tool import PubMedSearchTool
from src.tools.arxiv_tool import ArxivSearchTool
from src.agents.summarizer import PaperSummarizer


def print_separator(title: str):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def test_1_brief_summary():
    """Step 1: 搜索论文 → 生成简短摘要。"""
    print_separator("Step 1: Brief Summary from PubMed")

    # 搜索论文
    tool = PubMedSearchTool()
    result = tool._run(
        query="CAR-T cell therapy solid tumors clinical trial 2024",
        max_results=3,
    )
    print(f"Fetched {len(result['papers'])} papers from PubMed\n")

    # 生成摘要
    summarizer = PaperSummarizer()
    for i, paper in enumerate(result["papers"], 1):
        if len(paper.get("abstract", "")) < 100:
            print(f"  [{i}] Skipped — abstract too short")
            continue

        print(f"\n--- Paper {i}: {paper['title'][:80]}... ---")
        summary = summarizer.summarize(
            title=paper["title"],
            abstract=paper["abstract"],
            summary_type="brief",
        )
        if summary.get("brief"):
            b = summary["brief"]
            print(f"  💡 {b['one_liner']}")
            print(f"  🔑 {b['takeaway']}")

    print("\n[OK] Step 1 PASSED")
    return True


def test_2_detailed_summary():
    """Step 2: 详细结构摘要 (中文输出)。"""
    print_separator("Step 2: Detailed Summary (Chinese)")

    tool = PubMedSearchTool()
    result = tool._run(
        query="Alzheimer disease amyloid beta monoclonal antibody",
        max_results=2,
    )

    summarizer = PaperSummarizer()
    for i, paper in enumerate(result["papers"], 1):
        if len(paper.get("abstract", "")) < 100:
            continue

        print(f"\n--- Paper {i}: {paper['title'][:80]}... ---")
        summary = summarizer.summarize(
            title=paper["title"],
            abstract=paper["abstract"],
            summary_type="detailed",
            language="chinese",
        )
        if summary.get("detailed"):
            d = summary["detailed"]
            print(f"  📋 背景: {d['background']}")
            print(f"  🔬 方法: {d['methods']}")
            print(f"  📊 结果: {d['results']}")
            print(f"  💡 结论: {d['conclusions']}")

    print("\n[OK] Step 2 PASSED")
    return True


def test_3_key_findings():
    """Step 3: 关键发现要点。"""
    print_separator("Step 3: Key Findings")

    tool = PubMedSearchTool()
    result = tool._run(
        query="mRNA vaccine cancer neoantigen personalized immunotherapy",
        max_results=2,
    )

    summarizer = PaperSummarizer()
    for i, paper in enumerate(result["papers"], 1):
        if len(paper.get("abstract", "")) < 100:
            continue

        print(f"\n--- Paper {i}: {paper['title'][:80]}... ---")
        summary = summarizer.summarize(
            title=paper["title"],
            abstract=paper["abstract"],
            summary_type="key_findings",
        )
        if summary.get("key_findings"):
            kf = summary["key_findings"]
            for j, finding in enumerate(kf["findings"], 1):
                print(f"  {j}. {finding}")
            print(f"  📊 意义: {kf['significance']}")

    print("\n[OK] Step 3 PASSED")
    return True


def test_4_arxiv_integration():
    """Step 4: Arxiv 搜索结果 → LLM 摘要。"""
    print_separator("Step 4: Arxiv → Summarizer Integration")

    tool = ArxivSearchTool()
    result = tool._run(
        query="transformer attention mechanism drug discovery",
        max_results=2,
    )
    print(f"Fetched {len(result['papers'])} papers from Arxiv\n")

    summarizer = PaperSummarizer()
    for i, paper in enumerate(result["papers"], 1):
        summary_text = paper.get("summary", "")
        if len(summary_text) < 100:
            print(f"  [{i}] Skipped — abstract too short")
            continue

        print(f"\n--- Paper {i}: {paper['title'][:80]}... ---")
        print(f"  Categories: {', '.join(paper.get('categories', []))}")
        print(f"  Published: {paper.get('published', 'N/A')}")

        summary = summarizer.summarize(
            title=paper["title"],
            abstract=summary_text,
            summary_type="brief",
        )
        if summary.get("brief"):
            b = summary["brief"]
            print(f"  💡 {b['one_liner']}")
            print(f"  🔑 {b['takeaway']}")

    print("\n[OK] Step 4 PASSED")
    return True


def test_5_batch_summary():
    """Step 5: 批量摘要 — 一次处理多篇论文。"""
    print_separator("Step 5: Batch Summarization")

    tool = PubMedSearchTool()
    result = tool._run(
        query="CRISPR base editing genetic disease therapy",
        max_results=3,
    )

    # 筛选有效论文
    papers = [
        {"title": p["title"], "abstract": p["abstract"]}
        for p in result["papers"]
        if len(p.get("abstract", "")) >= 100
    ]
    print(f"Valid papers for batch: {len(papers)}/{len(result['papers'])}")

    summarizer = PaperSummarizer()
    results = summarizer.summarize_batch(
        papers,
        summary_type="brief",
        language="english",
    )

    ok = sum(1 for r in results if "error" not in r)
    print(f"Successfully summarized: {ok}/{len(results)}")

    for i, r in enumerate(results, 1):
        if r.get("brief"):
            print(f"  [{i}] {r['brief']['one_liner'][:100]}...")

    print(f"\n[OK] Step 5 PASSED — {ok}/{len(results)} successful")
    return True


def main():
    print("=" * 65)
    print("  Paper Summarizer — Integration Verification")
    print("=" * 65)
    print("  LLM Model: GPT-4o-mini via API2D")
    print("  摘要类型: brief / detailed / key_findings / batch")

    tests = [
        test_1_brief_summary,
        test_2_detailed_summary,
        test_3_key_findings,
        test_4_arxiv_integration,
        test_5_batch_summary,
    ]

    results = {}
    for test_fn in tests:
        try:
            results[test_fn.__name__] = "PASSED" if test_fn() else "FAILED"
        except Exception as e:
            print(f"\n[X] {test_fn.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[test_fn.__name__] = f"FAILED: {e}"

    # ── Summary ──
    print_separator("Summary")
    all_pass = True
    for name, status in results.items():
        icon = "[OK]" if status == "PASSED" else "[FAIL]"
        print(f"  {icon} {name}: {status}")
        if status != "PASSED":
            all_pass = False

    print(f"\n{'*** All tests passed! ***' if all_pass else 'WARNING: Some tests failed — check above.'}")
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
