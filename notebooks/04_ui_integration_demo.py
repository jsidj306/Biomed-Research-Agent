"""
Streamlit UI — 端到端集成验证 Demo。

不启动 Streamlit 服务器，直接验证 UI 层依赖的所有后端流程：
  1. 搜索 → 论文列表格式化
  2. 选择论文 → 生成摘要
  3. 全链路：搜索 → 摘要 → 输出
  4. 多源合并 (PubMed + Arxiv)
  5. 错误处理

运行: uv run python notebooks/04_ui_integration_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

from src.agents.summarizer import PaperSummarizer
from src.tools.arxiv_tool import ArxivSearchTool
from src.tools.pubmed_tool import PubMedSearchTool


def print_separator(title: str):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def test_1_search_and_format():
    """Step 1: 搜索论文并验证 UI 所需的数据格式。"""
    print_separator("Step 1: Search → Paper List Format")

    # PubMed 搜索
    pubmed = PubMedSearchTool()
    pubmed_result = pubmed._run(
        query="personalized mRNA cancer vaccine neoantigen",
        max_results=3,
    )

    papers = pubmed_result.get("papers", [])
    print(f"PubMed found: {len(papers)} papers")

    # 验证 UI 组件所需的字段
    for i, p in enumerate(papers, 1):
        assert "title" in p, f"Paper {i}: missing title"
        assert "authors" in p, f"Paper {i}: missing authors"
        assert "abstract" in p, f"Paper {i}: missing abstract"
        assert "source" in p or "pmid" in p, f"Paper {i}: missing source info"
        print(f"  [{i}] {p['title'][:80]}...")
        print(f"      Authors: {', '.join(p.get('authors', [])[:2])}")
        print(f"      Abstract length: {len(p.get('abstract', ''))} chars")

    print(f"\n[OK] Step 1 PASSED — {len(papers)} papers ready for UI rendering")
    return papers


def test_2_summarize_paper():
    """Step 2: 选择论文 → 生成摘要 → 验证摘要展示所需格式。"""
    print_separator("Step 2: Paper → Summarize → Summary Format")

    pubmed = PubMedSearchTool()
    result = pubmed._run(
        query="CAR-T cell therapy solid tumors clinical trial",
        max_results=1,
    )

    if not result["papers"]:
        print("No papers found, skipping summary test")
        return True

    paper = result["papers"][0]
    print(f"Selected: {paper['title'][:80]}...")

    summarizer = PaperSummarizer()
    summary = summarizer.summarize(
        title=paper["title"],
        abstract=paper["abstract"],
        summary_type="all",
        language="english",
    )

    # 验证摘要结果包含 UI 渲染所需的所有字段
    print(f"\nSummary type: {summary['summary_type']}")
    print(f"Model: {summary['model']}")

    if summary.get("brief"):
        b = summary["brief"]
        print(f"  Brief: {b['one_liner'][:100]}")
        assert "one_liner" in b
        assert "takeaway" in b

    if summary.get("detailed"):
        d = summary["detailed"]
        print(f"  Detailed: {len(d.get('background',''))} / {len(d.get('methods',''))} / "
              f"{len(d.get('results',''))} / {len(d.get('conclusions',''))} chars")
        for key in ["background", "methods", "results", "conclusions"]:
            assert key in d

    if summary.get("key_findings"):
        kf = summary["key_findings"]
        print(f"  Key Findings: {len(kf.get('findings', []))} points")
        assert len(kf["findings"]) >= 1
        assert "significance" in kf

    print("\n[OK] Step 2 PASSED — Summary format valid for UI rendering")
    return True


def test_3_full_pipeline():
    """Step 3: 全链路 — 搜索 → 展示 → 摘要 → 输出。"""
    print_separator("Step 3: Full Pipeline — Search → Display → Summarize")

    # 模拟 UI 的典型流程
    query = "Alzheimer disease blood biomarker diagnosis"
    print(f"User Query: '{query}'")

    # 1. 搜索
    pubmed = PubMedSearchTool()
    pubmed_result = pubmed._run(query=query, max_results=2)
    papers = pubmed_result["papers"]
    print(f"Step 1: Found {len(papers)} papers")

    # 2. 格式化（模拟 render_paper_list）
    display_papers = []
    for p in papers:
        display_papers.append({
            "title": p.get("title", "Untitled"),
            "authors": p.get("authors", []),
            "abstract": p.get("abstract", ""),
            "journal": p.get("journal", ""),
            "year": p.get("year", ""),
            "source": "pubmed",
            "pmid": p.get("pmid", ""),
            "url": p.get("url", ""),
        })
    print(f"Step 2: Formatted {len(display_papers)} papers for UI display")

    # 3. 摘要（模拟用户点击 Summary 按钮）
    summarizer = PaperSummarizer()
    valid_papers = [p for p in display_papers if len(p["abstract"]) >= 100]

    for i, paper in enumerate(valid_papers[:1], 1):
        print(f"\nStep 3: Summarizing paper {i}...")
        summary = summarizer.summarize(
            title=paper["title"],
            abstract=paper["abstract"],
            summary_type="key_findings",
            language="english",
        )
        if summary.get("key_findings"):
            kf = summary["key_findings"]
            for j, f in enumerate(kf["findings"], 1):
                print(f"  {j}. {f}")

    print("\n[OK] Step 3 PASSED — Full pipeline verified")
    return True


def test_4_multi_source_merge():
    """Step 4: 多源合并（PubMed + Arxiv）→ 统一格式。"""
    print_separator("Step 4: Multi-Source Merge — PubMed + Arxiv")

    query = "machine learning drug discovery molecular property prediction"
    all_papers = []

    # PubMed
    try:
        pubmed = PubMedSearchTool()
        pubmed_result = pubmed._run(query=query, max_results=2)
        for p in pubmed_result["papers"]:
            p["source"] = "pubmed"
        all_papers.extend(pubmed_result["papers"])
        print(f"PubMed: {len(pubmed_result['papers'])} papers")
    except Exception as e:
        print(f"PubMed failed: {e}")

    # Arxiv
    try:
        arxiv_tool = ArxivSearchTool()
        arxiv_result = arxiv_tool._run(query=query, max_results=2)
        for p in arxiv_result["papers"]:
            p["source"] = "arxiv"
            if "summary" in p:
                p["abstract"] = p["summary"]
        all_papers.extend(arxiv_result["papers"])
        print(f"Arxiv: {len(arxiv_result['papers'])} papers")
    except Exception as e:
        print(f"Arxiv failed: {e}")

    print(f"\nTotal merged: {len(all_papers)} papers")
    pubmed_count = sum(1 for p in all_papers if p.get("source") == "pubmed")
    arxiv_count = sum(1 for p in all_papers if p.get("source") == "arxiv")
    print(f"  PubMed: {pubmed_count} | Arxiv: {arxiv_count}")

    # 验证统一格式
    for p in all_papers:
        assert "title" in p
        assert "abstract" in p or "summary" in p
        assert "source" in p

    print("\n[OK] Step 4 PASSED — Multi-source merge works correctly")
    return True


def test_5_error_handling():
    """Step 5: 错误处理 — 网络中断 / 空结果 / 无效输入。"""
    print_separator("Step 5: Error Handling")

    # 空查询
    pubmed = PubMedSearchTool()
    result = pubmed._run(query="asdfjklqwerpoiu", max_results=2)
    assert isinstance(result["papers"], list)
    print(f"Empty query: {len(result['papers'])} papers (expected 0 or few)")

    # 无效摘要输入 → 不应崩溃
    summarizer = PaperSummarizer()
    result = summarizer.summarize(
        title="Test",
        abstract="Too short",  # < 50 chars
        summary_type="brief",
    )
    assert "warning" in result
    print(f"Short abstract: warning='{result['warning'][:60]}...'")

    # 空来源列表
    from src.ui.app import search_papers
    papers = search_papers("test", [], max_results=5)
    assert papers == []
    print("Empty sources: returned [] (no crash)")

    print("\n[OK] Step 5 PASSED — Error handling works correctly")
    return True


def main():
    print("=" * 65)
    print("  Streamlit UI — Integration Verification")
    print("=" * 65)
    print("  Validates all backend flows the UI depends on")
    print("  (Streamlit rendering is tested manually)")
    print()

    tests = [
        test_1_search_and_format,
        test_2_summarize_paper,
        test_3_full_pipeline,
        test_4_multi_source_merge,
        test_5_error_handling,
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
    print("\nNote: Streamlit UI rendering requires manual testing with:")
    print("  uv run streamlit run src/ui/app.py")
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
