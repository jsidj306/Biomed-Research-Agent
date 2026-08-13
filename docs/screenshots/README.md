# 演示截图与录屏说明

> 任务 7.3 交付物。以下截图与录屏需在**实际运行应用**时截取（Streamlit 是 WebSocket 动态渲染，
> 无法用普通 curl 截取），本目录给出清单与保存路径，供演示时逐项完成。

## 需要截取的 7 张截图

| # | 场景 | 保存路径 | 截图要点 |
|---|------|----------|----------|
| 1 | Streamlit 主界面 | `mvp_ui_main.png` | 整体布局：侧边栏 + 标题 + 聊天框 |
| 2 | 搜索结果列表 | `mvp_search_results.png` | 论文卡片 + 🔵PubMed/🟢Arxiv 来源标识 |
| 3 | 结构化摘要 | `mvp_structured_summary.png` | Brief/Detailed/Key Findings 三标签页 |
| 4 | pytest 覆盖率 | `mvp_test_coverage.png` | 终端 `86 passed, 82%` 输出 |
| 5 | GitHub Actions CI | `mvp_ci_passing.png` | Actions 绿色 ✅ 通过 |
| 6 | GitHub 仓库首页 | `mvp_github_repo.png` | 仓库首页含 README + 目录 |
| 7 | 项目目录结构 | `mvp_directory_structure.png` | VS Code 侧边栏 src/ 分层 |

## 截屏工具（Windows）

- 截图：`Win + Shift + S`（区域截图）或 Snipaste
- 录屏：`Win + G`（Xbox Game Bar）或 OBS Studio（免费开源）

## 录屏

- 录屏保存为 `../demo/mvp_demo_v1.mp4`
- 时长 5–8 分钟，按 `docs/demo_script.md` 的 6 个场景走完

## 本目录当前已自动生成的文件

- `coverage_report.txt` —— pytest 覆盖率报告（自动生成，可替代第 4 张截图用于文档）
- `directory_structure.txt` —— src/ + tests/ 目录结构（自动生成）
