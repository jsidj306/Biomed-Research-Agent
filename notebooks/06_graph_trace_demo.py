"""
第3-4周 检查项29：LangFuse 调用链追踪 — 端到端验证。

跑通一次真实的多Agent调用链（Supervisor → Search → Review → Summarize），
每个节点都埋点，跑到 LangFuse 后台能看到节点级火焰图。

前置:
    1. 安装依赖（已加入 pyproject）: uv sync
    2. 注册 LangFuse 账号并拿到 Key，写进 .env（见 .env.example）:
       LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST

运行:
    uv run python notebooks/06_graph_trace_demo.py

说明:
    - 未配 LangFuse Key 时，埋点自动降级为 no-op，不影响本脚本跑通。
    - 本脚本会真实调用 PubMed 等数据源 + LLM（消耗 API 配额），
      查询已尽量缩小范围。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from src.agents.graph import build_graph
from src.observability.tracing import LangfuseTracer

QUERY = "CAR-T cell therapy for CD19-positive leukemia"


def main() -> None:
    tracer = LangfuseTracer()
    print(f"LangFuse 追踪状态: {'✅ 已启用（Key 已配置）' if tracer.enabled else '🔶 未启用（缺 Key，降级为 no-op）'}")

    print(f"\n运行多Agent调用链，查询: {QUERY}\n")
    app = build_graph(tracer=tracer)

    try:
        result = app.invoke({"query": QUERY})
        print("=" * 64)
        print("最终报告:")
        print(result.get("final_report", "（空）"))
        print("=" * 64)
    finally:
        # 短脚本退出前主动 flush，确保 span 发到 LangFuse
        tracer.flush()
        print("\n已 flush 埋点数据。若 Key 已配置，可到 LangFuse 后台查看火焰图。")


if __name__ == "__main__":
    main()
