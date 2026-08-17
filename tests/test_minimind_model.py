"""
MiniMind 本地模型 — 单元测试。

覆盖 src/finetune/minimind_model.py 的模型结构、因果掩码、权重加载。
用小型配置构造模型，避免依赖实验课产出的外部权重文件，测试可独立运行。
"""

import torch

from src.finetune.minimind_model import (
    MiniMindConfig,
    MiniMindForCausalLM,
    load_minimind_pth,
)


def _small_config() -> MiniMindConfig:
    """一个小型配置（vocab 200 / dim 64 / 1 层），便于快速测试。"""
    return MiniMindConfig(
        dim=64, n_layers=1, n_heads=4, n_kv_heads=1,
        vocab_size=200, hidden_dim=128, max_seq_len=64,
    )


def _small_model() -> MiniMindForCausalLM:
    return MiniMindForCausalLM(_small_config())


# ── 结构与前向 ───────────────────────────────────────────────────────────


def test_head_dim_property():
    cfg = MiniMindConfig(dim=512, n_heads=8)
    assert cfg.head_dim == 64


def test_forward_output_shape():
    model = _small_model()
    ids = torch.randint(0, 200, (2, 16))
    with torch.no_grad():
        logits = model(ids)
    assert logits.shape == (2, 16, 200)


def test_forward_is_causal():
    """因果掩码：前缀的 logits 不应受后续 token 影响。"""
    model = _small_model()
    full = torch.randint(0, 200, (1, 16))
    prefix = full[:, :8]
    with torch.no_grad():
        logits_full = model(full)
        logits_prefix = model(prefix)
    # 前缀 8 个位置的 logits 应与完整序列的前 8 位一致
    assert torch.allclose(logits_full[:, :8], logits_prefix, atol=1e-5)


def test_forward_with_attention_mask():
    """带 padding 掩码时输出形状正确且不报错。"""
    model = _small_model()
    ids = torch.randint(0, 200, (2, 10))
    mask = torch.tensor([[1] * 10, [1] * 6 + [0] * 4])
    with torch.no_grad():
        logits = model(ids, attention_mask=mask)
    assert logits.shape == (2, 10, 200)


# ── 权重加载 ──────────────────────────────────────────────────────────────


def test_load_minimind_pth_roundtrip(tmp_path):
    """构造模型 → 存 state_dict → 重新加载，参数应一致。"""
    cfg = _small_config()
    model = _small_model()
    path = tmp_path / "ckpt.pth"
    torch.save(model.state_dict(), path)

    loaded = load_minimind_pth(str(path), config=cfg)
    for (k1, p1), (k2, p2) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert k1 == k2
        assert torch.equal(p1, p2)


def test_load_minimind_pth_infers_config(tmp_path):
    """不传 config 时，应从 state_dict 推断 dim / n_layers / vocab / hidden_dim。"""
    cfg = _small_config()
    model = _small_model()
    path = tmp_path / "ckpt.pth"
    torch.save(model.state_dict(), path)

    loaded = load_minimind_pth(str(path))
    assert loaded.config.n_layers == cfg.n_layers
    assert loaded.config.vocab_size == cfg.vocab_size
    assert loaded.config.dim == cfg.dim
