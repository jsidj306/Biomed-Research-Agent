# Biomedical Research Agent

## Project
AI-powered multi-agent system for biomedical literature research.
Built with LangChain + LangGraph + ChromaDB + Streamlit.

## Tech Stack
- Python 3.11, uv for dependency management
- LangChain + LangGraph for Agent orchestration
- ChromaDB for vector storage
- Sentence-Transformers for biomedical embeddings (PubMedBERT)
- Streamlit for UI, FastAPI for API
- PubMed, arXiv, Semantic Scholar, bioRxiv APIs
- Docker + docker-compose for deployment

## Development Commands
- Install: `uv sync`
- Run UI: `uv run streamlit run src/ui/app.py`
- Run API: `uv run uvicorn src.api.main:app --reload`
- Test: `uv run pytest`
- Single test: `uv run pytest tests/test_xxx.py -v`
- Docker: `docker-compose up -d`

## Architecture
```
User → Streamlit UI → FastAPI → LangGraph Agent
                                    ├── PubMed Search Tool
                                    ├── RAG Pipeline (ChromaDB)
                                    └── Report Generator
```

## Conventions
- Type hints on all functions
- Google-style docstrings
- All API calls have retry logic (tenacity)
- Pydantic models for data validation
- Tests alongside source in tests/
- Environment variables via python-dotenv (.env file)

## Key Files
- `src/agents/` — LangGraph StateGraph agent definitions
- `src/tools/` — PubMed, arXiv, Semantic Scholar API wrappers
- `src/rag/` — ChromaDB vector store + retrieval pipeline
- `src/models/` — Pydantic data models
- `src/ui/` — Streamlit UI components
- `src/api/` — FastAPI routes
- `notebooks/` — Verification and demo scripts
