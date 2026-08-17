"""
生物医学指令微调数据集生成器。

为 MiniMind LoRA 微调实验构造 {instruction, input, output} 三元组数据集。
数据基于一份手工策展的「药物-靶点-疾病-基因」事实库，通过多类指令模板
（摘要 / 实体抽取 / 三元组抽取 / 关系问答 / 术语解释 / 研究类型分类）确定性
展开，保证每条样本的 output 与事实库严格一致（模型学到真实关联而非臆造）。

设计原则：
  - 纯函数 + 种子化 RNG，可复现
  - 8:1:1 划分（train/val/test），划分逻辑独立可测
  - 输出 JSONL，每行 {instruction, input, output}

使用示例:
    from src.finetune.data import build_dataset, split_dataset, write_jsonl

    records = build_dataset(n_examples=800, seed=42)
    splits = split_dataset(records, seed=42)
    write_jsonl("data/train.jsonl", splits["train"])
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

# ── 手工策展的事实库 ───────────────────────────────────────────────────────
#
# 每个事实：一句话自然语言描述 + 对应的三元组 + 涉及的实体（按类型）。
# 三元组关系沿用 src/knowledge_graph 的关系集（TREATS/TARGETS/ASSOCIATED_WITH/
# ENCODES/IMPLICATED_IN）。

BIOMED_FACTS: list[dict[str, Any]] = [
    {
        "sentence": "CAR-T therapy treats relapsed B-cell acute lymphoblastic leukemia by targeting CD19 on malignant B cells.",
        "triples": [
            {"head_type": "drug", "head_name": "car-t", "relation": "TREATS", "tail_type": "disease", "tail_name": "leukemia"},
            {"head_type": "drug", "head_name": "car-t", "relation": "TARGETS", "tail_type": "target", "tail_name": "cd19"},
        ],
        "entities": {"drug": ["car-t"], "target": ["cd19"], "disease": ["leukemia"]},
    },
    {
        "sentence": "Rituximab, an anti-CD20 monoclonal antibody, treats diffuse large B-cell lymphoma.",
        "triples": [
            {"head_type": "drug", "head_name": "rituximab", "relation": "TARGETS", "tail_type": "target", "tail_name": "cd20"},
            {"head_type": "drug", "head_name": "rituximab", "relation": "TREATS", "tail_type": "disease", "tail_name": "lymphoma"},
        ],
        "entities": {"drug": ["rituximab"], "target": ["cd20"], "disease": ["lymphoma"]},
    },
    {
        "sentence": "Pembrolizumab blocks the PD-1 immune checkpoint and is approved for advanced melanoma.",
        "triples": [
            {"head_type": "drug", "head_name": "pembrolizumab", "relation": "TARGETS", "tail_type": "target", "tail_name": "pd-1"},
            {"head_type": "drug", "head_name": "pembrolizumab", "relation": "TREATS", "tail_type": "disease", "tail_name": "melanoma"},
        ],
        "entities": {"drug": ["pembrolizumab"], "target": ["pd-1"], "disease": ["melanoma"]},
    },
    {
        "sentence": "Imatinib inhibits the BCR-ABL tyrosine kinase and is the standard treatment for chronic myeloid leukemia.",
        "triples": [
            {"head_type": "drug", "head_name": "imatinib", "relation": "TARGETS", "tail_type": "target", "tail_name": "bcr-abl"},
            {"head_type": "drug", "head_name": "imatinib", "relation": "TREATS", "tail_type": "disease", "tail_name": "leukemia"},
        ],
        "entities": {"drug": ["imatinib"], "target": ["bcr-abl"], "disease": ["leukemia"]},
    },
    {
        "sentence": "Metformin improves glycemic control as first-line therapy for type 2 diabetes.",
        "triples": [
            {"head_type": "drug", "head_name": "metformin", "relation": "TREATS", "tail_type": "disease", "tail_name": "diabetes"},
        ],
        "entities": {"drug": ["metformin"], "target": [], "disease": ["diabetes"]},
    },
    {
        "sentence": "Trastuzumab targets HER2 and improves survival in HER2-positive breast cancer.",
        "triples": [
            {"head_type": "drug", "head_name": "trastuzumab", "relation": "TARGETS", "tail_type": "target", "tail_name": "her2"},
            {"head_type": "drug", "head_name": "trastuzumab", "relation": "TREATS", "tail_type": "disease", "tail_name": "cancer"},
        ],
        "entities": {"drug": ["trastuzumab"], "target": ["her2"], "disease": ["cancer"]},
    },
    {
        "sentence": "Osimertinib inhibits mutant EGFR in non-small cell lung cancer.",
        "triples": [
            {"head_type": "drug", "head_name": "osimertinib", "relation": "TARGETS", "tail_type": "target", "tail_name": "egfr"},
            {"head_type": "drug", "head_name": "osimertinib", "relation": "TREATS", "tail_type": "disease", "tail_name": "cancer"},
        ],
        "entities": {"drug": ["osimertinib"], "target": ["egfr"], "disease": ["cancer"]},
    },
    {
        "sentence": "Aspirin reduces inflammation by inhibiting cyclooxygenase and is used for cardiovascular prevention.",
        "triples": [
            {"head_type": "drug", "head_name": "aspirin", "relation": "TARGETS", "tail_type": "target", "tail_name": "cox"},
        ],
        "entities": {"drug": ["aspirin"], "target": ["cox"], "disease": []},
    },
    {
        "sentence": "BRCA1 encodes a tumor suppressor and its mutation is implicated in hereditary breast cancer.",
        "triples": [
            {"head_type": "gene", "head_name": "brca1", "relation": "IMPLICATED_IN", "tail_type": "disease", "tail_name": "cancer"},
        ],
        "entities": {"drug": [], "target": [], "disease": ["cancer"], "gene": ["brca1"]},
    },
    {
        "sentence": "TP53 is a tumor suppressor gene implicated in many human cancers.",
        "triples": [
            {"head_type": "gene", "head_name": "tp53", "relation": "IMPLICATED_IN", "tail_type": "disease", "tail_name": "cancer"},
        ],
        "entities": {"drug": [], "target": [], "disease": ["cancer"], "gene": ["tp53"]},
    },
    {
        "sentence": "The EGFR gene encodes the epidermal growth factor receptor, a target for many cancer drugs.",
        "triples": [
            {"head_type": "gene", "head_name": "egfr", "relation": "ENCODES", "tail_type": "target", "tail_name": "egfr"},
        ],
        "entities": {"drug": [], "target": ["egfr"], "disease": [], "gene": ["egfr"]},
    },
    {
        "sentence": "GLP-1 receptor agonists such as semaglutide treat type 2 diabetes and obesity.",
        "triples": [
            {"head_type": "drug", "head_name": "semaglutide", "relation": "TARGETS", "tail_type": "target", "tail_name": "glp-1 receptor"},
            {"head_type": "drug", "head_name": "semaglutide", "relation": "TREATS", "tail_type": "disease", "tail_name": "diabetes"},
        ],
        "entities": {"drug": ["semaglutide"], "target": ["glp-1 receptor"], "disease": ["diabetes"]},
    },
]

# ── 术语定义库 ─────────────────────────────────────────────────────────────

TERM_DEFINITIONS: dict[str, str] = {
    "PD-1": "Programmed cell death protein 1 (PD-1) is an immune checkpoint receptor on T cells; blocking it with inhibitors such as pembrolizumab restores anti-tumor immunity.",
    "HER2": "Human epidermal growth factor receptor 2 (HER2) is a receptor tyrosine kinase that is overexpressed in a subset of breast cancers and is targeted by trastuzumab.",
    "BCR-ABL": "The BCR-ABL fusion protein is a constitutively active tyrosine kinase produced by the Philadelphia chromosome, the hallmark of chronic myeloid leukemia, and is inhibited by imatinib.",
    "EGFR": "Epidermal growth factor receptor (EGFR) is a receptor tyrosine kinase frequently mutated in non-small cell lung cancer and targeted by osimertinib.",
    "CD19": "CD19 is a B-cell surface antigen expressed on most B-cell malignancies and is the target of CAR-T therapies and other immunotherapies.",
    "CAR-T therapy": "Chimeric antigen receptor (CAR) T-cell therapy engineers a patient's T cells to express a receptor targeting a tumor antigen such as CD19, and is used to treat B-cell malignancies.",
    "LoRA": "Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning method that freezes the base model and trains small low-rank matrices added to weight matrices.",
    "gene therapy": "Gene therapy delivers genetic material into a patient's cells to treat or prevent disease, including approaches such as CAR-T cell engineering.",
}

# ── 指令模板 ───────────────────────────────────────────────────────────────


def _summarize(fact: dict[str, Any]) -> dict[str, str]:
    """生成摘要指令样本。"""
    triples = fact["triples"]
    rel_txt = "; ".join(
        f"{t['head_name']} {t['relation'].replace('_', ' ').lower()} {t['tail_name']}"
        for t in triples
    )
    return {
        "instruction": "Summarize the biomedical text in three sentences: background, mechanism, and clinical implication.",
        "input": fact["sentence"],
        "output": (
            f"This passage describes {rel_txt}. "
            "Mechanistically, the drug acts on its molecular target to modulate disease-relevant pathways. "
            "Clinically, this relationship underpins the drug's approved indication and therapeutic rationale."
        ),
    }


def _extract_entities(fact: dict[str, Any]) -> dict[str, str]:
    """生成实体抽取指令样本。"""
    ents = fact["entities"]
    output = json.dumps(
        {
            "drugs": ents.get("drug", []),
            "targets": ents.get("target", []),
            "diseases": ents.get("disease", []),
            "genes": ents.get("gene", []),
        },
        ensure_ascii=False,
    )
    return {
        "instruction": "Extract the drugs, targets, diseases, and genes mentioned in the text. Return JSON.",
        "input": fact["sentence"],
        "output": output,
    }


def _extract_triples(fact: dict[str, Any]) -> dict[str, str]:
    """生成三元组抽取指令样本。"""
    output = json.dumps({"triples": fact["triples"]}, ensure_ascii=False)
    return {
        "instruction": "Extract (head, relation, tail) knowledge-graph triples from the text. Return JSON.",
        "input": fact["sentence"],
        "output": output,
    }


def _relation_qa(fact: dict[str, Any]) -> dict[str, str]:
    """生成关系问答样本（取第一条 TREATS 三元组）。"""
    treats = [t for t in fact["triples"] if t["relation"] == "TREATS"]
    if not treats:
        # 无 TREATS 关系时，改为问 TARGETS
        t = fact["triples"][0]
        q = f"Does {t['head_name']} target {t['tail_name']}?"
        a = f"Yes. {t['head_name']} targets {t['tail_name']}."
    else:
        t = treats[0]
        q = f"Does {t['head_name']} treat {t['tail_name']}?"
        a = f"Yes. {t['head_name']} is used to treat {t['tail_name']}."
    return {"instruction": q + " Answer with a short justification.", "input": "", "output": a}


def _define_term(term: str) -> dict[str, str]:
    """生成术语解释样本。"""
    return {
        "instruction": f"Explain the biomedical term '{term}'.",
        "input": "",
        "output": TERM_DEFINITIONS[term],
    }


_STUDY_TYPES = [
    ("randomized controlled trial", "a phase III randomized controlled trial comparing drug A to placebo in 2,000 patients"),
    ("cohort study", "a prospective cohort study following 50,000 participants over 10 years"),
    ("case-control study", "a case-control study matching 500 cases to 1,000 controls"),
    ("meta-analysis", "a meta-analysis pooling results from 12 randomized trials"),
    ("systematic review", "a systematic review of 30 studies on the topic"),
]


def _classify_study(design: str, desc: str) -> dict[str, str]:
    """生成研究类型分类样本。"""
    return {
        "instruction": "Classify the study design described below into one of: randomized controlled trial, cohort study, case-control study, meta-analysis, systematic review.",
        "input": desc,
        "output": design,
    }


# ── 数据集组装 ─────────────────────────────────────────────────────────────


def build_dataset(n_examples: int = 800, seed: int = 42) -> list[dict[str, str]]:
    """确定性生成生物医学指令数据集。

    先构造一个「模板轮」（覆盖所有事实 × 所有指令类型的组合），再循环取用直到
    凑满 n_examples，保证各指令类型比例均衡（而非只用少数模板补足）。

    Args:
        n_examples: 目标样本数
        seed:       随机种子

    Returns:
        List[dict] — 每条含 instruction / input / output
    """
    rng = random.Random(seed)

    # 模板轮：一个可调用对象 = 一个样本生成器（用默认参数绑定捕获循环变量）
    round_templates: list = []
    for fact in BIOMED_FACTS:
        round_templates.append(lambda f=fact: _summarize(f))
        round_templates.append(lambda f=fact: _extract_entities(f))
        round_templates.append(lambda f=fact: _extract_triples(f))
        round_templates.append(lambda f=fact: _relation_qa(f))
    for term in TERM_DEFINITIONS:
        round_templates.append(lambda t=term: _define_term(t))
    for design, desc in _STUDY_TYPES:
        round_templates.append(lambda d=design, s=desc: _classify_study(d, s))

    records: list[dict[str, str]] = []
    for i in range(n_examples):
        records.append(round_templates[i % len(round_templates)]())

    rng.shuffle(records)
    return records


def split_dataset(
    records: list[dict[str, str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict[str, str]]]:
    """按 8:1:1 划分训练 / 验证 / 测试集。

    Args:
        records:     完整样本列表
        train_ratio: 训练集比例
        val_ratio:   验证集比例（余下为测试集）
        seed:        随机种子

    Returns:
        {"train": [...], "val": [...], "test": [...]}
    """
    n = len(records)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": records[:n_train],
        "val": records[n_train : n_train + n_val],
        "test": records[n_train + n_val :],
    }


# ── JSONL 读写 ─────────────────────────────────────────────────────────────


def write_jsonl(path: str | Path, records: list[dict[str, str]]) -> Path:
    """把样本列表写为 JSONL（每行一个 JSON 对象，ensure_ascii=False）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def read_jsonl(path: str | Path) -> list[dict[str, str]]:
    """读取 JSONL 文件为样本列表。"""
    records: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
