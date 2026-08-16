# ============================================================
# Biomed-Research-Agent — 多阶段构建
#
# 依赖层与源码层分离：源码改动不会触发依赖重装（缓存层复用），
# 镜像更小、构建更快。uv 比 poetry 快 10-100 倍，尤其适合容器构建。
# ============================================================

FROM python:3.11-slim AS base
WORKDIR /app

# 让 uv 使用系统 Python（避免容器里再套一层 venv 的路径问题）
ENV UV_SYSTEM_PYTHON=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---- 依赖层（先拷贝清单，充分利用 Docker 缓存）----
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev

# ---- 源码层 ----
COPY . .

# 中文字体（PDF 导出用，避免中文变方框）+ weasyprint 的 Pango/GObject 原生库
# weasyprint 不是纯 Python：需要 libpango/libgobject 等系统库，仅装字体不够，
# 否则容器内 PDF 导出会抛 OSError（而非 ImportError），必须显式补齐。
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-noto-cjk \
        libpango-1.0-0 \
        libpangoft2-1.0-0 \
        libharfbuzz-subset0 \
        libgobject-2.0-0 \
        libffi8 \
        libgdk-pixbuf-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8501

# Streamlit 在容器内需监听 0.0.0.0 + 无头模式
CMD ["uv", "run", "streamlit", "run", "src/ui/app.py", "--server.address=0.0.0.0", "--server.headless=true"]
