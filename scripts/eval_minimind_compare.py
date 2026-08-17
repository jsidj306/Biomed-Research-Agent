"""MiniMind 预训练 vs SFT — 零训练困惑度对比（第 7 周任务 2，方案 B）。

直接加载实验课产出的 `pretrain_512.pth` 与 `full_sft_512.pth`，在生物医学
验证集（`data/val.jsonl` 的 output 字段）上各算一次困惑度，量化 SFT 相对
预训练基座在领域文本上的收益——无需任何训练。

用法::

    uv run python scripts/eval_minimind_compare.py \
        --tokenizer_dir "D:/QLDownload/Cursor/项目1" \
        --pretrain "D:/QLDownload/Cursor/项目1/pretrain_512.pth" \
        --sft "D:/QLDownload/Cursor/项目1/full_sft_512.pth" \
        --data ./data/val.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# 直接 `python scripts/...` 运行时，把项目根目录前置到 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.finetune.minimind_model import load_minimind_pth

# ── 困惑度 ──────────────────────────────────────────────────────────────


def compute_perplexity(model, tokenizer, texts: list[str], max_len: int = 512) -> float:
    """在给定文本上计算困惑度 ppl = exp(平均交叉熵)。

    逐条 tokenize（不加特殊 token），按因果语言模型目标在 shifted labels 上
    累加交叉熵，最后取指数。
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)[:max_len]
            if len(ids) < 2:
                continue
            input_ids = torch.tensor([ids], dtype=torch.long)
            logits = model(input_ids)  # (1, t, vocab)
            loss = F.cross_entropy(
                logits[0, :-1, :], input_ids[0, 1:], reduction="sum"
            )
            total_nll += float(loss)
            total_tokens += len(ids) - 1
    return math.exp(total_nll / max(1, total_tokens))


# ── 生成对比 ──────────────────────────────────────────────────────────────


def generate_answer(model, tokenizer, prompt: str, max_new: int = 64) -> str:
    """用聊天模板构造提示，生成回答（与官方 generate 同参数）。"""
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    gen_ids = model.generate(
        torch.tensor([ids], dtype=torch.long),
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new,
    )
    return tokenizer.decode(gen_ids[0].tolist())


# ── 主流程 ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MiniMind 预训练 vs SFT 困惑度对比")
    p.add_argument("--tokenizer_dir", default="D:/QLDownload/Cursor/项目1")
    p.add_argument("--pretrain", default="D:/QLDownload/Cursor/项目1/pretrain_512.pth")
    p.add_argument("--sft", default="D:/QLDownload/Cursor/项目1/full_sft_512.pth")
    p.add_argument("--data", default="./data/val.jsonl")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--demo", type=str, default="What is CAR-T cell therapy?")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    assert tokenizer.vocab_size == 6400, f"词表大小不符：{tokenizer.vocab_size} != 6400"

    with open(args.data, encoding="utf-8") as f:
        texts = [json.loads(line)["output"] for line in f]
    print(f"评测文本：{len(texts)} 条（来自 {args.data} 的 output 字段）\n")

    results: dict[str, float] = {}
    for name, path in (("pretrain(预训练)", args.pretrain), ("SFT(全量微调)", args.sft)):
        print(f"加载 {name}: {path}")
        model = load_minimind_pth(path)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        ppl = compute_perplexity(model, tokenizer, texts, max_len=args.max_len)
        results[name] = ppl
        print(f"  → 参数 {n_params:.2f}M，困惑度 ppl = {ppl:.2f}\n")

    # 对比
    pretrain_ppl, sft_ppl = results["pretrain(预训练)"], results["SFT(全量微调)"]
    delta = pretrain_ppl - sft_ppl
    print("=" * 60)
    print(f"预训练 ppl: {pretrain_ppl:.2f}")
    print(f"SFT    ppl: {sft_ppl:.2f}")
    print(f"下降幅度:   {delta:.2f} ({100 * delta / pretrain_ppl:.1f}%)")
    print("=" * 60)

    # 生成样例对比（定性）
    print(f"\n生成对比（提示：{args.demo!r}）\n")
    for name, path in (("pretrain", args.pretrain), ("SFT", args.sft)):
        model = load_minimind_pth(path)
        answer = generate_answer(model, tokenizer, args.demo)
        # 终端可能是 GBK，直接写文件最稳；这里仍尝试打印（多数情况下解码正常）
        print(f"[{name}] {answer}\n")


if __name__ == "__main__":
    main()
