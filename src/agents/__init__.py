"""Biomedical Research Agent — Agents package.

多Agent 协作体系中的各专职 Agent（V2.0「7-Agent」）：
- SupervisorAgent: 意图解析 + 研究步骤计划生成
- SearchAgent:     多源并行文献检索
- ReviewAgent:     相关性评分 + 去重 + 排序 + 矛盾检测
- PaperSummarizer: LLM 论文摘要生成器（含 gap_analysis）
- CriticAgent:     交叉验证 + 事实核查 + 引用真实性检查
- WriterAgent:     六段式结构化报告生成 + BibTeX 导出
- PlannerAgent:    研究方向推荐 + 实验假设生成
"""

from src.agents.critic_agent import CriticAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.review_agent import ReviewAgent
from src.agents.search_agent import SearchAgent
from src.agents.summarizer import PaperSummarizer
from src.agents.supervisor import SupervisorAgent
from src.agents.writer_agent import WriterAgent

__all__ = [
    "CriticAgent",
    "PaperSummarizer",
    "PlannerAgent",
    "ReviewAgent",
    "SearchAgent",
    "SupervisorAgent",
    "WriterAgent",
]
