"""
Knowledge Graph Extractor — 实体关系抽取（LLM 主抽取 + 词典兜底）。

从论文摘要中抽取「药物-靶点-疾病」三元组，用于构建知识图谱。生物医学是
知识图谱的黄金场景（药物-靶点-疾病天然是图结构）。

抽取策略（三结合）:
  1. LLM 主抽取：让 LLM 从文本中识别实体和关系（覆盖长尾、语义关系）
  2. 词典兜底：预置常用药物/靶点/疾病词典做规则匹配，LLM 抽不到的用词典补
  3. 去重合并：两种来源的三元组按 (head, relation, tail) 去重

使用示例:
    from src.knowledge_graph.extractor import KnowledgeGraphExtractor, to_mermaid

    extractor = KnowledgeGraphExtractor()
    triples = extractor.extract_triples("CAR-T therapy treats B-cell leukemia...")
    print(to_mermaid(triples))
"""

from __future__ import annotations

import logging
from typing import Any

from dotenv import load_dotenv
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.llm_utils import create_chat_model, parse_json_response

load_dotenv()

logger = logging.getLogger(__name__)

# ── 实体类型与关系 ───────────────────────────────────────────────────────

ENTITY_TYPES = ("drug", "target", "disease", "gene")

# 允许的关系（head → tail）
RELATIONS = {
    "TREATS": ("drug", "disease"),          # 药物 治疗 疾病
    "TARGETS": ("drug", "target"),          # 药物 靶向 靶点
    "ASSOCIATED_WITH": ("target", "disease"),  # 靶点 关联 疾病
    "ENCODES": ("gene", "target"),          # 基因 编码 靶点
    "IMPLICATED_IN": ("gene", "disease"),   # 基因 涉及 疾病
}

# 词典兜底（小写匹配）
DRUG_DICT = ("car-t", "rituximab", "pembrolizumab", "imatinib", "aspirin", "metformin")
TARGET_DICT = ("cd19", "pd-1", "pd-l1", "bcr-abl", "egfr", "her2")
DISEASE_DICT = (
    "cancer", "leukemia", "lymphoma", "melanoma", "diabetes",
    "alzheimer", "tumor", "carcinoma",
)

# ── Prompt 模板 ───────────────────────────────────────────────────────────

KG_PROMPT = """你是生物医学知识图谱抽取专家。从下面的文本中抽取实体关系三元组。

实体类型限定为: drug（药物）/ target（靶点）/ disease（疾病）/ gene（基因）。
关系限定为: TREATS / TARGETS / ASSOCIATED_WITH / ENCODES / IMPLICATED_IN。

文本:
{text}

返回 JSON 对象:
{{
  "triples": [
    {{"head": "drug", "head_name": "CAR-T", "relation": "TREATS", "tail": "disease", "tail_name": "leukemia"}}
  ]
}}

只抽取文本中明确出现的实体，不要臆造。若无实体关系，返回 {{"triples": []}}。
只返回合法 JSON，不要 markdown 围栏或多余文字。"""


# ── 三元组工具 ───────────────────────────────────────────────────────────


def _triple(head_type: str, head_name: str, relation: str, tail_type: str, tail_name: str) -> dict[str, str]:
    """构造规范化三元组。"""
    return {
        "head_type": head_type.lower(),
        "head_name": head_name.strip(),
        "relation": relation.upper(),
        "tail_type": tail_type.lower(),
        "tail_name": tail_name.strip(),
    }


def _triple_key(triple: dict[str, str]) -> tuple[str, str, str, str, str]:
    """三元组去重键。"""
    return (
        triple["head_type"].lower(),
        triple["head_name"].lower(),
        triple["relation"].upper(),
        triple["tail_type"].lower(),
        triple["tail_name"].lower(),
    )


def to_mermaid(triples: list[dict[str, str]]) -> str:
    """把三元组转成 Mermaid 图（无依赖，用于文档 / Streamlit 展示）。

    Args:
        triples: 三元组列表

    Returns:
        str — Mermaid graph 代码块
    """
    lines = ["graph TD"]
    for i, t in enumerate(triples):
        h = t.get("head_name", "?")
        r = t.get("relation", "RELATED_TO")
        tl = t.get("tail_name", "?")
        # Mermaid 节点 id 用序号，标签用实体名（含类型）
        hid = f"N{i}h"
        tid = f"N{i}t"
        lines.append(f'    {hid}["{h}"]')
        lines.append(f'    {tid}["{tl}"]')
        lines.append(f"    {hid} -->|{r}| {tid}")
    return "\n".join(lines)


# ── 抽取器实现 ───────────────────────────────────────────────────────────


class KnowledgeGraphExtractor:
    """生物医学实体关系抽取器。

    LLM 主抽取 + 词典兜底，输出规范化的三元组列表。

    Attributes:
        model_name:  LLM 模型名
        temperature: 温度（0.2，抽取稳定）
        llm:         ChatOpenAI 实例
    """

    def __init__(
        self,
        model_name: str = "glm-5.2",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        """初始化抽取器。"""
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = create_chat_model(model_name, temperature, max_tokens)
        logger.info("KnowledgeGraphExtractor ready: model=%s", model_name)

    # ── 公开接口 ────────────────────────────────────────────────────────

    def extract_triples(self, text: str) -> list[dict[str, str]]:
        """从文本抽取三元组：LLM 主抽取 + 词典兜底，合并去重。

        Args:
            text: 论文摘要或正文文本

        Returns:
            List[dict] — 规范化的三元组列表
        """
        if not text or len(text.strip()) < 10:
            logger.warning("Text too short for KG extraction, skipped")
            return []

        llm_triples = self._extract_llm(text)
        dict_triples = self._extract_dictionary(text)

        merged = self._dedupe(llm_triples + dict_triples)
        logger.info(
            "KG extraction: %d LLM + %d dict → %d unique triples",
            len(llm_triples), len(dict_triples), len(merged),
        )
        return merged

    # ── LLM 主抽取 ──────────────────────────────────────────────────────

    def _extract_llm(self, text: str) -> list[dict[str, str]]:
        """用 LLM 抽取三元组（失败降级为空列表）。"""
        try:
            prompt = KG_PROMPT.format(text=text[:4000])
            data = self._call_llm(prompt)
        except Exception as e:
            logger.warning("LLM KG extraction failed (degraded): %s", e)
            return []

        raw = data.get("triples", [])
        if not isinstance(raw, list):
            return []

        triples = [self._normalize(t) for t in raw]
        return [t for t in triples if t is not None]

    @staticmethod
    def _normalize(triple: Any) -> dict[str, str] | None:
        """把 LLM 输出的三元组归一化，非法项丢弃。"""
        if not isinstance(triple, dict):
            return None

        head_type = str(triple.get("head", triple.get("head_type", ""))).lower()
        tail_type = str(triple.get("tail", triple.get("tail_type", ""))).lower()
        relation = str(triple.get("relation", "")).upper()
        head_name = str(triple.get("head_name", "")).strip()
        tail_name = str(triple.get("tail_name", "")).strip()

        # 实体类型与关系合法性校验
        if head_type not in ENTITY_TYPES or tail_type not in ENTITY_TYPES:
            return None
        if relation not in RELATIONS:
            return None
        if not head_name or not tail_name:
            return None

        return _triple(head_type, head_name, relation, tail_type, tail_name)

    # ── 词典兜底 ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_dictionary(text: str) -> list[dict[str, str]]:
        """词典规则匹配兜底：从文本中匹配已知实体，生成共现三元组。

        纯规则实现，确定性可测。只对「同一文本中同时出现」的实体对生成三元组，
        避免臆造。
        """
        lower = text.lower()
        drugs = [d for d in DRUG_DICT if d in lower]
        targets = [t for t in TARGET_DICT if t in lower]
        diseases = [d for d in DISEASE_DICT if d in lower]

        triples: list[dict[str, str]] = []
        for drug in drugs:
            for disease in diseases:
                triples.append(_triple("drug", drug, "TREATS", "disease", disease))
        for drug in drugs:
            for target in targets:
                triples.append(_triple("drug", drug, "TARGETS", "target", target))
        for target in targets:
            for disease in diseases:
                triples.append(
                    _triple("target", target, "ASSOCIATED_WITH", "disease", disease)
                )
        return triples

    # ── 内部实现 ────────────────────────────────────────────────────────

    @staticmethod
    def _dedupe(triples: list[dict[str, str]]) -> list[dict[str, str]]:
        """按三元组键去重，保持首次出现顺序。"""
        seen: set[tuple[str, str, str, str, str]] = set()
        result: list[dict[str, str]] = []
        for t in triples:
            key = _triple_key(t)
            if key in seen:
                continue
            seen.add(key)
            result.append(t)
        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _call_llm(self, prompt: str) -> dict[str, Any]:
        """调用 LLM 并解析 JSON（3 次重试 + 指数退避）。"""
        response = self.llm.invoke(prompt)
        return parse_json_response(response.content)
