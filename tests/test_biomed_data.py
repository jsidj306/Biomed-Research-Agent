"""
MiniMind LoRA 指令数据集 — 单元测试。

运行: uv run pytest tests/test_biomed_data.py -v

验证数据集生成（确定性）、8:1:1 划分、JSONL 读写往返与字段完整性。
纯函数逻辑，无外部依赖。
"""

import json

from src.finetune.data import (
    build_dataset,
    read_jsonl,
    split_dataset,
    write_jsonl,
)

# ── 生成 ───────────────────────────────────────────────────────────────────


def test_build_dataset_returns_requested_count():
    records = build_dataset(n_examples=800, seed=42)
    assert len(records) == 800


def test_build_dataset_records_have_required_keys():
    records = build_dataset(n_examples=50, seed=1)
    for r in records:
        assert set(r) == {"instruction", "input", "output"}
        assert r["instruction"]
        assert r["output"]


def test_build_dataset_is_deterministic():
    a = build_dataset(n_examples=200, seed=7)
    b = build_dataset(n_examples=200, seed=7)
    assert a == b


def test_build_dataset_covers_multiple_instruction_types():
    records = build_dataset(n_examples=800, seed=42)
    instructions = {r["instruction"] for r in records}
    # 至少覆盖实体抽取 / 三元组抽取 / 术语解释 三类
    assert any("Extract the drugs" in i for i in instructions)
    assert any("knowledge-graph triples" in i for i in instructions)
    assert any(i.startswith("Explain the biomedical term") for i in instructions)


# ── 划分 ───────────────────────────────────────────────────────────────────


def test_split_dataset_8_1_1():
    records = build_dataset(n_examples=800, seed=42)
    splits = split_dataset(records, seed=42)
    assert len(splits["train"]) == 640
    assert len(splits["val"]) == 80
    assert len(splits["test"]) == 80


def test_split_dataset_no_loss_no_duplication():
    records = build_dataset(n_examples=300, seed=3)
    splits = split_dataset(records, seed=3)
    merged = splits["train"] + splits["val"] + splits["test"]
    assert len(merged) == len(records)
    # 无重复（样本本身可能重复，但数量必须一致）
    assert sorted(json.dumps(r, sort_keys=True) for r in merged) == sorted(
        json.dumps(r, sort_keys=True) for r in records
    )


# ── JSONL 读写 ─────────────────────────────────────────────────────────────


def test_write_read_jsonl_roundtrip(tmp_path):
    records = build_dataset(n_examples=20, seed=9)
    path = tmp_path / "data.jsonl"
    write_jsonl(path, records)
    loaded = read_jsonl(path)
    assert loaded == records


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "with_blank.jsonl"
    path.write_text('{"instruction":"a","input":"","output":"b"}\n\n\n', encoding="utf-8")
    assert read_jsonl(path) == [{"instruction": "a", "input": "", "output": "b"}]
