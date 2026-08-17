"""
MiniMind 本地模型定义 — 用于加载实验课产出的 `.pth` 权重。

与 jingyaogong/MiniMind 的架构严格一致：RMSNorm + RoPE + GQA + SwiGLU FFN。
子模块命名与实验课保存的 state_dict 一一对应（`tok_embeddings` /
`layers.N.attention.wq|wk|wv|wo` / `attention_norm` / `ffn_norm` /
`feed_forward.w1|w2|w3` / `norm` / `output`），可直接 `load_state_dict`。

纯 torch 实现，不依赖 transformers / trust_remote_code，便于在无 GPU 环境
加载本地权重做困惑度评估或 LoRA 微调。

使用示例::

    from src.finetune.minimind_model import MiniMindConfig, MiniMindForCausalLM, load_minimind_pth

    model = load_minimind_pth("pretrain_512.pth")  # 自动推断 n_layers
    logits = model(input_ids)                      # (batch, seq, vocab)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class MiniMindConfig:
    """MiniMind 结构超参（与实验课权重对齐）。

    默认值来自 MiniMind 的公开默认配置；加载 `.pth` 时会按权重形状自动
    推断 `n_layers` / `hidden_dim`，其余保持默认。
    """

    dim: int = 512
    n_layers: int = 2
    n_heads: int = 8
    n_kv_heads: int = 2
    vocab_size: int = 6400
    hidden_dim: int = 1408  # = round(2 * 4 * dim / 3, multiple_of=64)
    max_seq_len: int = 512
    norm_eps: float = 1e-5
    rope_theta: float = 1e6
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


class RMSNorm(nn.Module):
    """RMSNorm（MiniMind 使用的归一化）。"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_normed = x.float()
        x_normed = x_normed * torch.rsqrt(x_normed.pow(2).mean(-1, keepdim=True) + self.eps)
        return x_normed.type_as(x) * self.weight


class RoPE(nn.Module):
    """旋转位置编码（RoPE）。freqs_cis 每次按需计算，不进 state_dict。"""

    def __init__(self, dim: int, max_seq_len: int, base: float = 1e6):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self._freqs_cis = None

    def _precompute(self, device: torch.device) -> torch.Tensor:
        if self._freqs_cis is None or self._freqs_cis.device != device:
            theta = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim)
            )
            seq = torch.arange(self.max_seq_len, device=device).float()
            freqs = torch.outer(seq, theta)  # (max_seq_len, dim//2)
            self._freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return self._freqs_cis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_heads, head_dim)
        seq_len = x.shape[1]
        freqs_cis = self._precompute(x.device)[:seq_len]  # (seq_len, dim//2)
        freqs_cis = freqs_cis.unsqueeze(1)  # (seq_len, 1, dim//2) → 广播到各注意力头
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_out = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
        return x_out.type_as(x)


class Attention(nn.Module):
    """GQA 注意力（n_kv_heads 个 KV 头，按 n_rep 重复到 n_heads）。"""

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_heads // config.n_kv_heads
        self.head_dim = config.head_dim
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        self.rope = RoPE(self.head_dim, config.max_seq_len, config.rope_theta)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.head_dim)
        k = self.wk(x).view(b, t, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(b, t, self.n_kv_heads, self.head_dim)

        q = self.rope(q)
        k = self.rope(k)

        # GQA：KV 头重复到 Q 头数量
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)

        q = q.transpose(1, 2)  # (b, h, t, d)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if mask is not None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.wo(out)


class FeedForward(nn.Module):
    """SwiGLU 前馈网络（w1 / w2 / w3）。"""

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MiniMindBlock(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), mask)
        return h + self.feed_forward(self.ffn_norm(h))


class MiniMindForCausalLM(nn.Module):
    """MiniMind 因果语言模型（结构严格对齐实验课保存的 state_dict）。"""

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(MiniMindBlock(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.dim, config.norm_eps)
        # 保存的权重里 output.weight 是独立键（与 tok_embeddings 数值相同但不共享），
        # 这里保持独立，保证 load_state_dict(strict=True) 能一一对应。
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.tok_embeddings(input_ids)
        mask = self._build_mask(input_ids, attention_mask)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.output(x)  # (batch, seq, vocab)

    @staticmethod
    def _build_mask(
        input_ids: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor | None:
        """构造因果 + padding 的注意力布尔掩码 (b, 1, t, t)。无 padding 时返回 None。"""
        if attention_mask is None:
            return None
        _, t = input_ids.shape
        device = input_ids.device
        causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=device))  # (t, t)
        key_ok = attention_mask.bool().unsqueeze(1).unsqueeze(2)  # (b, 1, 1, t)
        return causal.unsqueeze(0).unsqueeze(0) & key_ok  # (b, 1, t, t)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        eos_token_id: int = 2,
        max_new_tokens: int = 128,
        temperature: float = 0.75,
        top_p: float | None = 0.9,
    ) -> torch.Tensor:
        """自回归生成（贪心/采样）。输入 `input_ids` 形状 (1, seq)。"""
        self.eval()
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids[:, -self.config.max_seq_len :])[:, -1, :]  # (1, vocab)
            logits = logits / (temperature + 1e-9)
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                remove = cum_probs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                indices_to_remove = remove.scatter(1, sorted_idx, remove)
                logits = logits.masked_fill(indices_to_remove, -float("inf"))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if next_id.item() == eos_token_id:
                break
        return input_ids


# ── 加载辅助 ──────────────────────────────────────────────────────────────


def _infer_n_layers(state_dict: dict) -> int:
    """从 state_dict 推断层数。"""
    return sum(1 for k in state_dict if k.endswith(".attention.wq.weight"))


def load_minimind_pth(
    path: str,
    config: MiniMindConfig | None = None,
    device: str = "cpu",
) -> MiniMindForCausalLM:
    """从 `.pth` state_dict 加载 MiniMind 模型。

    Args:
        path:   `.pth` 权重路径（实验课产出）
        config: 结构超参；缺省时按权重形状自动推断 n_layers / hidden_dim
        device: 加载设备

    Returns:
        MiniMindForCausalLM — 已加载权重的模型（eval 模式）
    """
    state_dict = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(state_dict, dict):
        raise TypeError(f"{path} 不是 state_dict（得到 {type(state_dict).__name__}）")

    n_layers = _infer_n_layers(state_dict)
    # hidden_dim 从 w1 形状推断：layers.0.feed_forward.w1.weight → (hidden_dim, dim)
    w1 = state_dict["layers.0.feed_forward.w1.weight"]
    hidden_dim = w1.shape[0]
    dim = w1.shape[1]
    vocab_size = state_dict["tok_embeddings.weight"].shape[0]

    if config is None:
        config = MiniMindConfig(
            dim=dim, n_layers=n_layers, vocab_size=vocab_size, hidden_dim=hidden_dim
        )

    model = MiniMindForCausalLM(config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model
