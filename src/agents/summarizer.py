"""
Paper Summarizer — LLM 驱动的论文摘要生成器。

使用 API2D (OpenAI 兼容) 的中转 API，调用 LLM 对生物医学论文
生成结构化摘要。支持三种摘要类型：
  - brief:       一句话概括 + 关键启示
  - detailed:    背景 / 方法 / 结果 / 结论 四段式
  - key_findings: 3-5 个关键发现要点

使用示例:
    summarizer = PaperSummarizer()
    result = summarizer.summarize(
        title="mRNA vaccine shows promise in pancreatic cancer",
        abstract="...",
        summary_type="detailed",
        language="chinese",
    )
    print(result["background"])
"""

import logging
import os
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

logger = logging.getLogger(__name__)

# ── 摘要 Prompt 模板 ────────────────────────────────────────────────────

BRIEF_PROMPT = """You are a senior biomedical researcher. Summarize the following paper in a concise way.

Paper Title: {title}
Abstract: {abstract}

Respond in {language}. Return a JSON object with exactly these keys:
{{
  "one_liner": "One sentence summarizing the core finding",
  "takeaway": "One sentence on the key implication for researchers in this field"
}}

Return ONLY valid JSON, no markdown fences or extra text."""

DETAILED_PROMPT = """You are a senior biomedical researcher. Write a structured summary of the following paper.

Paper Title: {title}
Abstract: {abstract}

Respond in {language}. Return a JSON object with exactly these keys:
{{
  "background": "Research background and the problem addressed (2-3 sentences)",
  "methods": "Methods and technical approach used (2-3 sentences)",
  "results": "Key results and findings (2-3 sentences)",
  "conclusions": "Conclusions and future directions (1-2 sentences)"
}}

Return ONLY valid JSON, no markdown fences or extra text."""

KEY_FINDINGS_PROMPT = """You are a senior biomedical researcher. Extract the key findings from the following paper.

Paper Title: {title}
Abstract: {abstract}

Respond in {language}. Return a JSON object with exactly these keys:
{{
  "findings": ["Finding 1", "Finding 2", "Finding 3", "Finding 4", "Finding 5"],
  "significance": "1-2 sentences on the significance of this research"
}}

Provide 3-5 key findings. Return ONLY valid JSON, no markdown fences or extra text."""


# ── Summarizer 实现 ─────────────────────────────────────────────────────


class PaperSummarizer:
    """LLM 驱动的生物医学论文摘要生成器。

    通过 API2D 中转调用 OpenAI 兼容的 LLM API，
    对论文标题+摘要生成结构化摘要。

    Attributes:
        llm:         LangChain ChatOpenAI 实例
        model_name:  LLM 模型名称
        temperature: LLM 温度参数
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ):
        """初始化摘要生成器。

        Args:
            model_name:  LLM 模型名称 (gpt-4o-mini / gpt-4o / gpt-3.5-turbo)
            temperature: LLM 温度 (0-1)，越低越确定
            max_tokens:  LLM 最大输出 token 数
        """
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
        )

        logger.info(
            "PaperSummarizer ready: model=%s, base_url=%s, temp=%.2f",
            model_name, base_url, temperature,
        )

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def summarize(
        self,
        title: str,
        abstract: str,
        summary_type: Literal["brief", "detailed", "key_findings", "all"] = "all",
        language: str = "english",
    ) -> dict[str, Any]:
        """对单篇论文生成摘要。

        Args:
            title:        论文标题
            abstract:     论文摘要文本
            summary_type: 摘要类型
            language:     输出语言 (english / chinese)

        Returns:
            {
                "paper_title": str,
                "summary_type": str,
                "brief": { "one_liner": str, "takeaway": str } | None,
                "detailed": { ... } | None,
                "key_findings": { ... } | None,
                "model": str,
                "tokens_used": int,
            }
        """
        if not abstract or len(abstract.strip()) < 50:
            logger.warning("Abstract too short (%d chars), skipping", len(abstract))
            return self._empty_result(title, summary_type)

        types_to_generate = (
            ["brief", "detailed", "key_findings"] if summary_type == "all"
            else [summary_type]
        )

        result = {
            "paper_title": title,
            "summary_type": summary_type,
            "brief": None,
            "detailed": None,
            "key_findings": None,
            "model": self.model_name,
            "tokens_used": 0,
        }

        prompt_map = {
            "brief": BRIEF_PROMPT,
            "detailed": DETAILED_PROMPT,
            "key_findings": KEY_FINDINGS_PROMPT,
        }

        for st in types_to_generate:
            prompt = prompt_map[st].format(
                title=title,
                abstract=abstract[:3000],  # 截断过长摘要以控制 token
                language=language,
            )
            try:
                parsed = self._call_llm(prompt)
                result[st] = parsed
            except Exception as e:
                logger.error("Failed to generate %s summary: %s", st, e)
                result[st] = None

        return result

    def summarize_batch(
        self,
        papers: list[dict[str, str]],
        summary_type: Literal["brief", "detailed", "key_findings", "all"] = "all",
        language: str = "english",
    ) -> list[dict[str, Any]]:
        """批量生成摘要。

        Args:
            papers:      论文列表，每个元素需含 title 和 abstract
            summary_type: 摘要类型
            language:    输出语言

        Returns:
            摘要结果列表
        """
        results = []
        for i, paper in enumerate(papers):
            logger.info(
                "Summarizing %d/%d: %s",
                i + 1, len(papers), paper.get("title", "")[:60],
            )
            try:
                result = self.summarize(
                    title=paper.get("title", ""),
                    abstract=paper.get("abstract", ""),
                    summary_type=summary_type,
                    language=language,
                )
                results.append(result)
            except Exception as e:
                logger.error("Failed on paper %d: %s", i + 1, e)
                results.append({
                    "paper_title": paper.get("title", ""),
                    "summary_type": summary_type,
                    "error": str(e),
                    "model": self.model_name,
                })

        ok = sum(1 for r in results if "error" not in r)
        logger.info("Batch complete: %d/%d successful", ok, len(results))
        return results

    # ── 内部实现 ──────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _call_llm(self, prompt: str) -> dict[str, Any]:
        """调用 LLM 并解析 JSON 响应。

        内置 3 次重试 + 指数退避 (2s → 4s → 8s)。

        Args:
            prompt: 完整的 prompt 文本

        Returns:
            解析后的 JSON 字典

        Raises:
            ValueError: JSON 解析失败
        """
        import json

        response = self.llm.invoke(prompt)
        content = response.content.strip()

        # 移除可能的 markdown 代码块标记
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:]) if len(lines) > 1 else content
            content = content.removesuffix("```")
            content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 尝试提取 JSON 子串
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
            raise ValueError(f"Failed to parse LLM response as JSON: {content[:200]}")

    def _empty_result(self, title: str, summary_type: str) -> dict[str, Any]:
        """返回空摘要结果（当输入文本过短时）。"""
        return {
            "paper_title": title,
            "summary_type": summary_type,
            "brief": None,
            "detailed": None,
            "key_findings": None,
            "model": self.model_name,
            "tokens_used": 0,
            "warning": "Abstract too short for meaningful summarization",
        }


# ── 快速测试入口 ──────────────────────────────────────────────────────

if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)

    summarizer = PaperSummarizer()

    # 测试单篇摘要
    test_title = (
        "mRNA vaccine combined with immune checkpoint inhibitor "
        "shows promising results in advanced pancreatic cancer: "
        "a phase I/II clinical trial"
    )
    test_abstract = (
        "Pancreatic ductal adenocarcinoma (PDAC) remains one of the "
        "deadliest cancers with a 5-year survival rate below 10%. "
        "Here we report the results of a phase I/II clinical trial "
        "(NCT04812345) evaluating autogene cevumeran, an individualized "
        "mRNA neoantigen vaccine, in combination with atezolizumab and "
        "mFOLFIRINOX in 34 patients with surgically resected PDAC. "
        "The combination therapy was well-tolerated with grade 3-4 "
        "adverse events in 41% of patients, primarily hematologic. "
        "At a median follow-up of 24 months, the 18-month RFS rate was "
        "72% compared to 45% in historical controls. T cell receptor "
        "repertoire analysis revealed significant clonal expansion of "
        "vaccine-induced neoantigen-specific T cells, with sustained "
        "persistence up to 18 months post-vaccination. Single-cell "
        "RNA sequencing of peripheral blood mononuclear cells showed "
        "enrichment of CD8+ effector memory T cells expressing high "
        "levels of granzyme B and perforin. These results provide the "
        "first evidence that individualized mRNA neoantigen vaccines "
        "can induce durable T cell responses and delay recurrence in "
        "PDAC patients, warranting further investigation in a "
        "randomized phase III trial."
    )

    result = summarizer.summarize(
        title=test_title,
        abstract=test_abstract,
        summary_type="detailed",
        language="chinese",
    )

    print(f"\n=== {result['paper_title'][:80]}... ===")
    if result.get("detailed"):
        d = result["detailed"]
        print(f"\n📋 背景: {d.get('background', 'N/A')}")
        print(f"\n🔬 方法: {d.get('methods', 'N/A')}")
        print(f"\n📊 结果: {d.get('results', 'N/A')}")
        print(f"\n💡 结论: {d.get('conclusions', 'N/A')}")
    print(f"\n模型: {result['model']}")
