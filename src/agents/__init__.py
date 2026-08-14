"""Biomedical Research Agent — Agents package.

多Agent 协作体系中的各专职 Agent：
- SupervisorAgent: 意图解析 + 研究步骤计划生成
- SearchAgent:     多源并行文献检索
- ReviewAgent:     相关性评分 + 去重 + 排序
- PaperSummarizer: LLM 论文摘要生成器
"""

from src.agents.review_agent import ReviewAgent
from src.agents.search_agent import SearchAgent
from src.agents.summarizer import PaperSummarizer
from src.agents.supervisor import SupervisorAgent

__all__ = ["PaperSummarizer", "ReviewAgent", "SearchAgent", "SupervisorAgent"]
