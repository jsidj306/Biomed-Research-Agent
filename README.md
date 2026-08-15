# Biomedical Research Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

AI-powered multi-agent system for biomedical literature research.
Built with LangChain + LangGraph + ChromaDB.

## Features (Planned)
- 🔍 **Multi-source literature search** — PubMed, arXiv, Semantic Scholar, bioRxiv
- 🧬 **Biomedical domain embeddings** — PubMedBERT for domain-specific retrieval
- 🤖 **Multi-agent orchestration** — LangGraph StateGraph with specialist agents
- 📊 **Structured research report** — Auto-generated literature review with citations

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/biomed-research-agent.git
cd biomed-research-agent
uv sync
cp .env.example .env  # edit with your API keys
docker-compose up -d
uv run streamlit run src/ui/app.py
```

## Architecture

```
User → Streamlit UI → FastAPI → LangGraph Agent
                                    ├── PubMed Search Tool
                                    ├── RAG Pipeline (ChromaDB)
                                    └── Report Generator
```

## Tech Stack
| Layer | Technology |
|-------|-----------|
| Agent Framework | LangChain + LangGraph |
| Vector Database | ChromaDB |
| Embeddings | Sentence-Transformers (PubMedBERT) |
| UI | Streamlit |
| API | FastAPI + Uvicorn |
| Data Sources | PubMed, arXiv, Semantic Scholar, bioRxiv |
| Infrastructure | Docker, docker-compose |

## 演示 / Demo

- 🎬 [V1.0 多 Agent 编排演示（5–8 分钟）](docs/demo/v1_demo.mp4) — Supervisor 派单 → 5 源并发搜索 → Review 评分 → Summarize 输出，含混合检索评测与 LangFuse 追踪

## 技术博客 / Blog

- [从 MiniMind 到 LangGraph——构建生物医学研究 Agent](https://juejin.cn/spost/7673780779483676678)（掘金）

## License
MIT
