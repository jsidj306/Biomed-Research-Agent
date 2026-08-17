"""
PubMed Search Tool — 端到端集成验证 Demo。

验证项:
1. 基础搜索 — 返回结构化论文列表
2. 多种排序 — relevance vs date
3. 错误处理 — 空查询/无效排序
4. Tool Schema — LangChain 可正确生成 JSON Schema

运行: uv run python notebooks/01_pubmed_tool_demo.py
"""

import sys
from pathlib import Path

# Fix Windows GBK encoding — force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# 确保项目根目录在 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.pubmed_tool import PubMedSearchTool


def print_separator(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def test_1_basic_search():
    """测试 1: 基础搜索 — 返回结构化论文列表。"""
    print_separator("Test 1: Basic Search")
    print('Query: "mRNA vaccine cancer immunotherapy clinical trial 2024"\n')

    tool = PubMedSearchTool()
    result = tool._run(
        query="mRNA vaccine cancer immunotherapy clinical trial 2024",
        max_results=5,
    )

    print(f"Total papers in PubMed: {result['total_count']}")
    print(f"Returned: {len(result['papers'])}")
    print(f"Source: {result['source']}")

    assert result["source"] == "pubmed", "Source should be pubmed"
    assert len(result["papers"]) <= 5, "Should return at most 5 papers"
    assert result["total_count"] >= 0, "Total count should be >= 0"

    for i, paper in enumerate(result["papers"], 1):
        print(f"\n--- Paper {i} ---")
        print(f"Title:    {paper['title'][:100]}")
        print(f"Authors:  {', '.join(paper['authors'][:3])}")
        print(f"Journal:  {paper['journal']} ({paper['year']})")
        print(f"PMID:     {paper['pmid']}")
        print(f"DOI:      {paper.get('doi', 'N/A')}")
        print(f"URL:      {paper['url']}")
        print(f"Abstract: {paper['abstract'][:120]}...")

        # 验证必需字段存在
        for field in ["pmid", "title", "authors", "year", "journal", "abstract", "url"]:
            assert field in paper, f"Missing field: {field}"

    print("\n[OK] Test 1 PASSED")
    return True


def test_2_sort_by_date():
    """测试 2: 按日期排序 — 最新的论文排前面。"""
    print_separator("Test 2: Sort by Date")
    print('Query: "Alzheimer disease biomarker blood test"\n')

    tool = PubMedSearchTool()
    result = tool._run(
        query="Alzheimer disease biomarker blood test",
        max_results=5,
        sort="date",
    )

    years = [int(p["year"]) for p in result["papers"] if p["year"].isdigit()]
    print(f"Years (should be descending): {years}")

    if len(years) >= 2:
        assert years == sorted(years, reverse=True), (
            f"Years should be in descending order, got {years}"
        )
        print("[OK] Years are correctly sorted (newest first)")

    print("[OK] Test 2 PASSED")
    return True


def test_3_empty_query():
    """测试 3: 无意义查询 — 应正常返回空或不抛异常。"""
    print_separator("Test 3: Empty/Minimal Query Handling")

    tool = PubMedSearchTool()
    result = tool._run(query="asdfjklqwerpoiu", max_results=5)

    print(f"Total count: {result['total_count']}")
    print(f"Papers returned: {len(result['papers'])}")

    # 应该不抛异常，返回空或极少结果
    assert isinstance(result["papers"], list), "Papers should be a list"
    assert result["total_count"] >= 0, "Total count should be >= 0"

    print("[OK] Test 3 PASSED")
    return True


def test_4_tool_schema():
    """测试 4: Tool 可被 LangChain 正确序列化。"""
    print_separator("Test 4: LangChain Tool Schema")

    tool = PubMedSearchTool()

    # LangChain 的 BaseTool 会自动生成 args_schema 的 JSON Schema
    schema = tool.args_schema.model_json_schema()
    print("Tool Name:", tool.name)
    print("Tool Description:", tool.description[:80], "...")
    print("\nArgs JSON Schema:")
    import json
    print(json.dumps(schema, indent=2))

    assert tool.name == "pubmed_search"
    assert "PubMed" in tool.description
    assert "query" in schema.get("properties", {})
    assert "max_results" in schema.get("properties", {})
    assert "sort" in schema.get("properties", {})

    print("\n[OK] Test 4 PASSED")
    return True


def test_5_batch_search():
    """测试 5: 大批量搜索 — 验证分批次 EFetch 正常工作。"""
    print_separator("Test 5: Batch Search (max_results=30)")

    tool = PubMedSearchTool()
    result = tool._run(query="CRISPR-Cas9 gene editing review", max_results=30)

    print(f"Requested: 30, Returned: {len(result['papers'])}")

    assert len(result["papers"]) <= 30
    assert all(
        p["pmid"] for p in result["papers"]
    ), "Every paper should have a PMID"

    print("[OK] Test 5 PASSED")
    return True


def main():
    print("=" * 70)
    print("  PubMed Search Tool — Integration Verification")
    print("=" * 70)

    tests = [
        test_1_basic_search,
        test_2_sort_by_date,
        test_3_empty_query,
        test_4_tool_schema,
        test_5_batch_search,
    ]

    results = {}
    for test_fn in tests:
        try:
            results[test_fn.__name__] = "PASSED" if test_fn() else "FAILED"
        except Exception as e:
            print(f"\n[X] {test_fn.__name__} FAILED: {e}")
            results[test_fn.__name__] = f"FAILED: {e}"

    # ── Summary ──
    print_separator("Summary")
    all_pass = True
    for name, status in results.items():
        icon = "[OK]" if status == "PASSED" else "[FAIL]"
        print(f"  {icon} {name}: {status}")
        if status != "PASSED":
            all_pass = False

    print(f"\n{'*** All tests passed! ***' if all_pass else 'WARNING: Some tests failed - check above.'}")
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
