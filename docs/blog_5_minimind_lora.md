# 回到起点：给 12M 的 MiniMind 做一次 LoRA 微调——困惑度降了 95.7%，然后呢？

> 这是「Biomed-Research-Agent」项目的第五篇技术复盘，也是一篇「番外」。
> 第一篇《从 MiniMind 到 LangGraph》是这个项目的起点——手搓小模型，搞懂「模型是怎么训练出来的」。
> 四篇正文讲完了多 Agent 编排、Critic 自纠错、知识图谱、报告导出，收官的时候我说：
> 「评估」——怎么证明一个 Agent 真的比另一个好——还留着一个更大的坑。
> 这一篇，我绕回起点，去做一件更「底层」的事：**给那个 12M 的 MiniMind 做一次 LoRA 微调**。
> 结果困惑度降了 95.7%，但我诚实地告诉你，这**不等于**它学会了。

---

## 一、为什么在「大模型 Agent」项目里，回去做「小模型微调」

项目做到第七周，我手上有的是一个 7-Agent 的 LangGraph 系统——搜索、评审、摘要、写报告，全靠调 GPT/GLM/Kimi 这些大模型 API。

这时候回去做 MiniMind 的 LoRA 微调，看起来是「开倒车」。但恰恰相反，它补的是技术叙事里最容易漏掉的一块：

> **「会编排大模型」≠「懂大模型」。** 一个只会 `llm.invoke()` 的人，和一个亲手看过 loss 曲线、调过 LoRA rank、明白「困惑度下降 ≠ 理解」的人，对 LLM 的理解是两个层次。

留学申请和面试里，这条故事线的起点恰恰是「从 MiniMind 到 LangGraph」——你得真的懂最底层的那层，故事才立得住。所以第七周，我把「垂直领域小模型」这条技术路线，用一次实验验证掉。

---

## 二、一个意外的加分项：用实验课自己的权重

原计划是下载 MiniMind2-Small（约 26M）来做基座。但我想起自己在实验课上产出的那一批权重还躺在硬盘里——`pretrain_512.pth`（预训练）和 `full_sft_512.pth`（SFT 之后）。

好处立刻出来了：

| 对比项 | MiniMind2-Small（外部下载） | 实验课本地权重 |
|--------|---------------------------|--------------|
| 参数量 | ~26M | **12.19M**（2 层 / dim 512） |
| 获取方式 | 从 HuggingFace 下载 | **纯本地 `.pth`，离线加载** |
| 依赖 | 官方 `model.py` | 自写加载器，零外部依赖 |
| 词表 | 6400 | 6400 |

正好项目定案是「本地 Docker 部署」，用完全离线的本地权重，和这条线严丝合缝。

但麻烦也随之而来：这些权重是**纯 PyTorch state_dict**（21 个键），不是 HuggingFace 目录，没有 `model.safetensors` + `config.json` 那套。要加载它，我得先把模型结构「照着手里的权重」重建出来。

---

## 三、把模型结构「照着手里的权重」重建出来

这是整个实验里最容易被低估的一步。MiniMind 的架构是 RMSNorm + RoPE + GQA + SwiGLU，我照着实验课保存的键名逐个对齐：

```
tok_embeddings.weight                      # (vocab 6400, dim 512)
layers.0.attention.wq / wk / wv / wo       # GQA：8 个 Q 头，2 个 KV 头
layers.0.feed_forward.w1 / w2 / w3         # SwiGLU
layers.0.attention_norm / ffn_norm         # RMSNorm
norm.weight                                # 最终 RMSNorm
output.weight                              # (vocab 6400, dim 512)
```

`load_state_dict(strict=True)` 一次通过——说明重建的结构和实验课保存的**逐键一致**。

这里藏着一个特别容易踩的坑：

```python
# 保存的权重里，output.weight 和 tok_embeddings.weight 数值相同，但分存两个键
# 如果图省事用「共享权重」（tied），load_state_dict(strict=True) 会报 missing key
self.tok_embeddings = nn.Embedding(vocab, dim)
self.output = nn.Linear(dim, vocab, bias=False)   # 必须独立，不能共享
```

> 踩坑记录：MiniMind 官方实现里输入嵌入和输出头是**绑定（tied）**的——`output.weight` 直接复用 `tok_embeddings.weight`。但实验课保存时把这两个存成了**两个独立的键**（数值相同）。如果我在加载器里写 `self.output.weight = self.tok_embeddings.weight`，strict 加载就会少一个键报错。解法是让它们数值相同但物理独立。

---

## 四、手动实现 LoRA：不依赖 PEFT

LoRA 的原理一句话：**冻结原权重，加一个低秩增量 `ΔW = B·A`**。原模型不变，只训练 A、B 两个小矩阵。

我选择手动实现，而不是用 PEFT 库。原因是本项目用的是**自定义模型**（自写结构），PEFT 需要模型是 HF 格式、能 `get_peft_model()`，对自写结构有兼容风险。手动实现反而零风险，而且代码只有几十行：

```python
class LoRALinear(nn.Module):
    def __init__(self, linear, r, alpha):
        super().__init__()
        self.linear = linear           # 原权重，冻结
        self.scale = alpha / r
        out_dim, in_dim = linear.weight.shape
        self.lora_A = nn.Parameter(torch.randn(r, in_dim) * 0.02)  # 小随机值
        self.lora_B = nn.Parameter(torch.zeros(out_dim, r))        # 0 初始化
        linear.weight.requires_grad_(False)

    def forward(self, x):
        base = self.linear(x)
        delta = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scale
        return base + delta
```

两个关键设计：

1. **B 用 0 初始化**。这样训练开始那一刻 `ΔW = 0`，LoRA 后的模型和原模型**逐位等价**——不会因为初始化把模型「带偏」。
2. **只对 attention 的 wq/wk/wv/wo 和 ffn 的 w1/w2/w3 挂 LoRA**，冻结词表和输出头。最终可训练参数 **145,408 个，只占总参数 1.2%**。

`scale = alpha / r = 16 / 8 = 2`，rank 取 8。这是 LoRA 论文里的经典配比。

---

## 五、数据：800 条「药物-靶点-疾病」指令

数据不能用「随便找的问答」，否则模型学到的和事实库对不上。我建了一份手工策展的事实库（`BIOMED_FACTS`），从里面**确定性**生成 800 条指令——每条 `output` 和事实库严格一致，模型学到的是**真实关联**而不是臆造：

| 类型 | 说明 | 条数 |
|------|------|------|
| summarize | 文本摘要（背景/机制/临床意义） | 158 |
| extract_entities | 实体抽取（drug/target/disease/gene） | 158 |
| extract_triples | 三元组抽取 | 158 |
| relation_qa | 关系问答（"X 是否治疗 Y？"） | 157 |
| define_term | 术语解释（PD-1 / HER2 / CAR-T） | 104 |
| classify_study | 研究类型分类（RCT/队列等） | 65 |

按 8:1:1 切 train/val/test。训练文本用本地 tokenizer 的 `apply_chat_template` 生成，和实验课 SFT 的训练格式一致（`<s>user\n...<s>assistant\n...`，注意**不是** `<|im_start|>` 那种格式）。

---

## 六、结果：困惑度 156 → 6.72，然后呢

CPU 上跑，3 个 epoch，全程约 80 秒。验证集困惑度：

| 阶段 | 验证集困惑度 | 相对基线 |
|------|-------------|---------|
| 基线（未微调） | 156.26 | — |
| epoch 1 | 34.11 | -78.2% |
| epoch 2 | 12.69 | -91.9% |
| **epoch 3** | **6.72** | **-95.7%** |

单看这个数字，很漂亮——LoRA 让一个 12M 的小模型「稳定地学到了」药物-靶点-疾病指令数据的分布，这是「小模型 + 参数高效微调」可行的**直接定量证据**。

但我必须诚实地说另一面：

> 用微调后的模型做零样本问答，它**输出不了连贯的回答**。预训练基座是近乎随机的文本，LoRA 之后能吐出领域词汇和格式（三元组的 `head/tail` 结构、`melanoma` 这类术语），但贪心解码会**退化成反复复读训练指令**——"Explain the text. Return JSON." 循环。

**困惑度下降 95.7%，主要反映的是「记住了训练分布的表面格式」，不等于「理解了问答」。** 2 层 / 12M 的容量摆在那里，LoRA 只能让它记住，不能让它泛化。

这个「负结果」我原样写进了实验记录。因为——上一篇博客里我反复强调的——**诚实的负结果 + 归因分析，比硬吹「微调成功」更有说服力**。困惑度是一个会骗人的指标，知道它在哪骗人，本身就是收获。

---

## 七、踩坑与心得

**1. 困惑度会骗人。** 它衡量的是「模型对训练分布的拟合程度」，和「能不能回答问题」是两回事。一个只有 12M、2 层的模型，困惑度可以降得很漂亮，但生成照样不连贯。**评估小模型时，定性生成 + 困惑度一起看，缺一不可。**

**2. 加载别人的权重，先照键名重建结构。** 不要想当然用「标准 MiniMind 结构」，要 `strict=True` 对齐到保存的 state_dict。尤其是 tied 权重分存两键这种细节，差一个键就加载失败。

**3. 手动实现 LoRA，性价比极高。** 几十行代码，换来对「低秩适配到底在做什么」的彻底理解，还避免了 PEFT 对自定义模型的兼容性风险。B 用 0 初始化这个细节，是「增量从零开始」的关键。

**4. 本地权重是真正的「资产」。** 实验课产出的 `pretrain_512.pth` / `full_sft_512.pth` 一旦离线加载成功，整个实验就完全脱离了对云端下载的依赖——这正好呼应项目「本地部署」的定案。

---

## 八、下一步

这个实验留下了两条明确的改进路径：

1. **换更大基座**：8 层 MiniMind 或 MiniMind2-Small（~26M）重跑 LoRA，生成质量预期显著改善。
2. **用 SFT 权重做基座**：实验课已经有 `full_sft_512.pth`（全量 SFT，已具备指令跟随），在它上面再做 LoRA 领域微调——SFT 管「会说话」，LoRA 只补「领域知识」，分工更合理。

不过那是下一个项目的事了。到这里，第七周的 LoRA 实验把「从 MiniMind 到 LangGraph」这条故事线的**最底层一环**补上了——现在我能说，我不仅会编排大模型，也亲手训练过小模型，还知道它的天花板在哪。

---

*项目地址：[github.com/jsidj306/Biomed-Research-Agent](https://github.com/jsidj306/Biomed-Research-Agent)*
*2026-08*
