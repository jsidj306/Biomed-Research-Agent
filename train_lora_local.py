"""MiniMind（本地 2 层权重）LoRA 微调脚本 — 第 7 周任务 2（方案 A）。

加载实验课产出的 `pretrain_512.pth`（本地基座，无需外部下载）与本地 tokenizer，
在生物医学指令数据上做 LoRA 微调，验证「小模型 + 参数高效微调」能否学会
「药物-靶点-疾病」领域的结构化任务。以聊天模板构造训练文本，用验证集困惑度
量化微调收益（困惑度应随训练下降）。

用法::

    uv run python train_lora_local.py \
        --base /path/to/pretrain_512.pth \
        --tokenizer_dir /path/to/tokenizer \
        --data_path ./data/biomed_instruct.jsonl \
        --lora_rank 8 --lora_alpha 16 --epochs 3 --lr 1e-4

设计要点：
  - 手动实现 LoRA（`LoRALinear` 包装），不依赖 PEFT，对本项目自定义模型零兼容性风险
  - 只对 transformer 块的 attention(wq/wk/wv/wo) 与 ffn(w1/w2/w3) 挂 LoRA，
    冻结词表与输出头（与 MiniMind 官方「tied 输出」语义保持一致）
  - 聊天模板来自本地 tokenizer（`apply_chat_template`），与 SFT 训练格式一致
  - 因果掩码 + padding 掩码正确隔离 pad token，避免批量训练时的注意力泄漏
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
from pathlib import Path

# 直接 `python train_lora_local.py` 运行时，把项目根目录前置到 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer

from src.finetune.data import read_jsonl, split_dataset
from src.finetune.minimind_model import MiniMindForCausalLM, load_minimind_pth

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_lora_local")


# ── 手动 LoRA ────────────────────────────────────────────────────────────


class LoRALinear(nn.Module):
    """给一个 Linear 层挂 LoRA：冻结原权重，加低秩增量 A·B。"""

    def __init__(self, linear: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.linear = linear
        self.r = r
        self.scale = alpha / r
        out_dim, in_dim = linear.weight.shape
        # A 用小随机值初始化，B 用 0 初始化 → 初始增量 = 0，与原模型等价
        self.lora_A = nn.Parameter(torch.randn(r, in_dim) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_dim, r))
        linear.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        delta = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scale
        return base + delta


_TARGET_SUFFIXES = (".wq", ".wk", ".wv", ".wo", ".w1", ".w2", ".w3")


def apply_lora(model: MiniMindForCausalLM, r: int, alpha: int) -> None:
    """把模型里所有 attention / ffn 的 Linear 替换为 LoRALinear，冻结其余参数。"""
    replaced = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.endswith(_TARGET_SUFFIXES):
            parent_name, child = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            setattr(parent, child, LoRALinear(module, r=r, alpha=alpha))
            replaced += 1
    # 冻结所有非 LoRA 参数（词表 + 输出头 + 原 Linear 权重已在 LoRALinear 内冻结）
    for p in model.parameters():
        p.requires_grad = False
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)
    logger.info("LoRA: 替换 %d 个 Linear（r=%d alpha=%d scale=%.2f）", replaced, r, alpha, alpha / r)


def lora_parameters(model: MiniMindForCausalLM):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module.lora_A
            yield module.lora_B


def save_lora(model: MiniMindForCausalLM, out_dir: Path, r: int, alpha: int) -> None:
    """保存 LoRA 增量（lora_A / lora_B + 缩放参数）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {
        name.replace(".linear", "") + ".lora_A": module.lora_A.detach().cpu()
        for name, module in _iter_lora(model)
    }
    state.update({
        name.replace(".linear", "") + ".lora_B": module.lora_B.detach().cpu()
        for name, module in _iter_lora(model)
    })
    torch.save({"r": r, "alpha": alpha, "state_dict": state}, out_dir / "lora_adapter.pt")
    logger.info("saved LoRA adapter -> %s", out_dir / "lora_adapter.pt")


def _iter_lora(model):
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def load_lora(model: MiniMindForCausalLM, adapter_path: str | Path) -> None:
    """把保存的 LoRA 增量载入已 apply_lora 的模型（用于推理）。"""
    ckpt = torch.load(adapter_path, map_location="cpu")
    state = ckpt["state_dict"]
    loaded = 0
    for name, module in _iter_lora(model):
        module.lora_A.data.copy_(state[name + ".lora_A"])
        module.lora_B.data.copy_(state[name + ".lora_B"])
        loaded += 1
    logger.info("loaded LoRA adapter (%d layers) from %s", loaded, adapter_path)


# ── 数据格式化 ────────────────────────────────────────────────────────────


def build_chat_texts(records: list[dict], tokenizer) -> list[str]:
    """用 tokenizer 的聊天模板把 {instruction, input, output} 拼成训练文本。"""
    texts = []
    for rec in records:
        instruction = rec["instruction"]
        inp = rec.get("input", "")
        user = instruction if not inp else f"{instruction}\n{inp}"
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": rec["output"]},
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return texts


def tokenize_batch(texts: list[str], tokenizer, max_len: int):
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
    )
    return enc["input_ids"], enc["attention_mask"]


# ── 困惑度评估 ────────────────────────────────────────────────────────────


def evaluate_ppl(model, tokenizer, texts: list[str], max_len: int, batch_size: int) -> float:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            input_ids, attention_mask = tokenize_batch(texts[i : i + batch_size], tokenizer, max_len)
            logits = model(input_ids, attention_mask=attention_mask)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_nll += float(loss)
            total_tokens += int((labels[:, 1:] != -100).sum())
    model.train()
    return math.exp(total_nll / max(1, total_tokens))


# ── 训练主流程 ─────────────────────────────────────────────────────────────


def train(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    assert tokenizer.vocab_size == 6400, f"词表大小不符：{tokenizer.vocab_size} != 6400"

    records = read_jsonl(args.data_path)
    splits = split_dataset(records, seed=args.seed)
    train_texts = build_chat_texts(splits["train"], tokenizer)
    val_texts = build_chat_texts(splits["val"], tokenizer)
    logger.info("data: %d train / %d val / %d test", len(train_texts), len(val_texts), len(splits["test"]))

    model = load_minimind_pth(args.base)
    apply_lora(model, r=args.lora_rank, alpha=args.lora_alpha)

    optimizer = torch.optim.AdamW(lora_parameters(model), lr=args.lr)

    # 基线困惑度（未微调）
    baseline = evaluate_ppl(model, tokenizer, val_texts, args.max_length, args.batch_size)
    logger.info("baseline val_ppl = %.2f", baseline)

    rng = random.Random(args.seed)
    model.train()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_texts)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(train_texts), args.batch_size):
            batch = train_texts[i : i + args.batch_size]
            input_ids, attention_mask = tokenize_batch(batch, tokenizer, args.max_length)
            logits = model(input_ids, attention_mask=attention_mask)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            n_batches += 1
            if n_batches % 20 == 0 or i + args.batch_size >= len(train_texts):
                logger.info("epoch %d/%d step %d loss=%.4f", epoch, args.epochs, n_batches, float(loss))

        avg_loss = total_loss / max(1, n_batches)
        ppl = evaluate_ppl(model, tokenizer, val_texts, args.max_length, args.batch_size)
        logger.info("epoch %d/%d avg_loss=%.4f val_ppl=%.2f", epoch, args.epochs, avg_loss, ppl)

    save_lora(model, Path(args.output_dir), args.lora_rank, args.lora_alpha)
    logger.info("done: baseline=%.2f final_val_ppl=%.2f", baseline, ppl)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MiniMind 本地 LoRA 微调")
    p.add_argument("--base", required=True, help="本地 MiniMind 基座权重 .pth 路径（实验课产出，未随仓库分发）")
    p.add_argument("--tokenizer_dir", required=True, help="本地 tokenizer 目录（含 tokenizer.json / vocab.json）")
    p.add_argument("--data_path", default="./data/biomed_instruct.jsonl")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="./out/minimind-lora-local")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
