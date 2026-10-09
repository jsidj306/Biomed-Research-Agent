# MiniMind 生物医学 LoRA 微调实验

在 **MiniMind**（极轻量 LLM）上用 **LoRA** 微调生物医学指令数据，验证
「小模型 + 参数高效微调」能否在消费级硬件上掌握「药物-靶点-疾病」领域的
结构化任务。这是第 7 周任务 2 的实验记录。

> 实际基座采用实验课产出的本地权重 `pretrain_512.pth`（2 层 MiniMind，
> 约 12M 参数），而非最初计划的 MiniMind2-Small——本地权重可完全离线运行，
> 无需外部下载，符合「本地部署」定案。

## 1. 实验目标

- 构造一份生物医学指令数据集（≥500 条，本实验 800 条）
- 用 LoRA（rank=8）微调 MiniMind，使基座模型学会领域任务
- 以验证集困惑度（perplexity）评估，量化微调收益
- 产出可复现的训练脚本 + 数据集 + 本实验记录

## 2. 数据集

### 生成方式

数据集由 `scripts/generate_biomed_instruct.py` 确定性生成，基于一份手工策展的
「药物-靶点-疾病-基因」事实库（`src/finetune/data.py` 中的 `BIOMED_FACTS`）。
每条样本的 `output` 与事实库严格一致，保证模型学到真实关联而非臆造。

数据格式为 JSONL，每行 `{instruction, input, output}`。

### 规模与划分

| 集合 | 条数 | 比例 |
|------|------|------|
| 训练集 `data/train.jsonl` | 640 | 80% |
| 验证集 `data/val.jsonl` | 80 | 10% |
| 测试集 `data/test.jsonl` | 80 | 10% |
| **合计** | **800** | 100% |

### 指令类型分布

| 类型 | 说明 | 条数 |
|------|------|------|
| `summarize` | 文本摘要（背景/机制/临床意义） | 158 |
| `extract_entities` | 实体抽取（drug/target/disease/gene） | 158 |
| `extract_triples` | 三元组抽取（(head, relation, tail)） | 158 |
| `relation_qa` | 关系问答（"X 是否治疗 Y？"） | 157 |
| `define_term` | 术语解释（PD-1 / HER2 / CAR-T 等） | 104 |
| `classify_study` | 研究类型分类（RCT/队列/病例对照等） | 65 |

## 3. 基座模型与 LoRA 配置

### 基座模型（本地权重）

| 项 | 值 |
|----|----|
| 权重文件 | 本地 `pretrain_512.pth`（实验课产出，未随仓库分发） |
| 架构 | MiniMind（RMSNorm + RoPE + GQA + SwiGLU） |
| 层数 / 维度 | 2 层 / 512 |
| 词表 | 6400（tokenizer 文件：`tokenizer.json` + `vocab.json`） |
| 注意力 | 8 头 / 2 KV 头（GQA），head_dim 64 |
| 参数量 | **12.19M**（fp32，约 35.7 MB） |
| 权重形式 | 纯 PyTorch state_dict（`.pth`，非 HuggingFace 目录） |

本地模型加载器 `src/finetune/minimind_model.py` 与实验课保存的 state_dict 键名
严格对齐（`tok_embeddings` / `layers.N.attention.wq|wk|wv|wo` / `attention_norm` /
`ffn_norm` / `feed_forward.w1|w2|w3` / `norm` / `output`），纯 torch 实现，
`load_state_dict(strict=True)` 验证通过。

### LoRA 配置（手动实现，见 `train_lora_local.py`）

| 项 | 值 |
|----|----|
| LoRA rank `r` | 8 |
| LoRA alpha | 16（scale = alpha/r = 2） |
| 目标层 | 14 个 Linear（每层 attention wq/wk/wv/wo + ffn w1/w2/w3） |
| 可训练参数 | 145,408（约占总参数 1.2%） |
| 冻结 | 词表 / 输出头 / 全部原始 Linear 权重 |
| 优化器 | AdamW，lr=1e-4 |
| batch size | 8 |
| max length | 256 tokens |
| epochs | 3 |
| 设备 | CPU（2 层 12M 模型，全程约 80 秒） |

> 训练文本由本地 tokenizer 的 `apply_chat_template` 生成
> （`<s>system\n你是 MiniMind...</s>\n<s>user\n...</s>\n<s>assistant\n...</s>\n`），
> 与实验课 SFT 的训练格式一致。

## 4. 训练与评估

```bash
# 1) LoRA 微调 + 验证集困惑度评估（无需 GPU，约 1.5 分钟）
uv run python train_lora_local.py \
    --base /path/to/pretrain_512.pth \
    --tokenizer_dir /path/to/tokenizer \
    --data_path ./data/biomed_instruct.jsonl \
    --lora_rank 8 --lora_alpha 16 --epochs 3 --lr 1e-4
```

训练完成后 LoRA 增量保存到 `out/minimind-lora-local/lora_adapter.pt`。

评估指标：验证集困惑度 `ppl = exp(平均交叉熵)`，每轮 epoch 打印一次，训练前
先算一次基线困惑度。

## 5. 结果

### 困惑度（定量，主指标）

| 阶段 | 验证集困惑度 | 相对基线 |
|------|-------------|---------|
| 基线（预训练基座，未微调） | 156.26 | — |
| epoch 1 | 34.11 | -78.2% |
| epoch 2 | 12.69 | -91.9% |
| epoch 3（最终） | **6.72** | **-95.7%** |

验证集困惑度从 156 骤降至 6.72，**下降 95.7%**，说明 LoRA 微调让这个 12M 小模型
稳定地学到了「药物-靶点-疾病」指令数据的分布——这是「小模型 + 参数高效微调」
可行的直接定量证据。

### 生成质量（定性）

用微调后的模型做零样本问答，观察到一个**诚实的负面结论**：

- 预训练基座生成近乎随机文本（无意义片段）。
- LoRA 微调后能输出领域词汇与格式（如三元组的 `head/tail` 结构、`melanoma`
  等术语），但**无法产生连贯的完整回答**——贪心解码会退化成反复输出训练指令
  片段（"Explain the text. Return JSON."）。

结论：**2 层 / 12M 的模型容量不足，LoRA 只能让它「记住分布与表面格式」，
无法泛化到连贯的问答。** 困惑度下降主要反映对训练分布的拟合（含记忆），
不等于理解。

## 6. 复现

1. 确认本地已存在 `pretrain_512.pth`、`tokenizer.json`、`vocab.json`（实验课产出的权重与词表）
2. `uv run python scripts/generate_biomed_instruct.py --n_examples 800`（数据已生成可跳过）
3. `uv run python train_lora_local.py --base /path/to/pretrain_512.pth --tokenizer_dir /path/to/tokenizer`
4. 生成对比：加载 `out/minimind-lora-local/lora_adapter.pt` 做问答，观察输出

## 7. 结论与下一步

- **数据管道已就绪**：800 条确定性、均衡、与事实库一致的指令数据，8:1:1 划分。
- **本地模型加载器已就绪**：`src/finetune/minimind_model.py` 纯 torch 实现，严格
  对齐实验课 state_dict，可离线加载预训练 / SFT 权重。
- **LoRA 微调已跑通**：手动 LoRA（`train_lora_local.py`），验证集困惑度 156 → 6.72
  （-95.7%），说明 PEFT 在小模型上机制有效。
- **关键限制**：2 层 12M 模型容量不足，困惑度下降不等于问答能力，生成仍不连贯。

**下一步建议**：
1. 换用更大基座（如 8 层 MiniMind 或 MiniMind2-Small ~26M）重跑 LoRA，预期生成质量
   显著改善。
2. 若需真正的问答能力，考虑直接用实验课的 `full_sft_512.pth`（全量 SFT）作为基座，
   在其上再做 LoRA 领域微调——SFT 已具备指令跟随，LoRA 只需补领域知识。

相关代码：`src/finetune/data.py`、`src/finetune/minimind_model.py`、
`scripts/generate_biomed_instruct.py`、`train_lora_local.py`、`tests/test_biomed_data.py`。
