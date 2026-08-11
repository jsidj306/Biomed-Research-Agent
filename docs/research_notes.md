# research_notes.md — 研究笔记

**项目**: biomed-research-agent  
**开始日期**: 2026-08-10

---

## 2026-08-11

### 网络架构备忘

- GitHub SSH 直连 22 端口被墙，需走 `ssh.github.com:443`
- `~/.ssh/config` 配置 `Host github.com → Hostname ssh.github.com → Port 443`
- API 调用走 API2D 中转 (`https://oa.api2d.net`)

### NVIDIA 参考项目分析

- 架构核心是四阶段研究循环：Plan → Search → Write → Reflect
- 但强依赖 NVIDIA GPU 生态（NIM、RAG Blueprint、NeMo）
- **决策**: 借鉴架构模式，用 ChromaDB + API2D 自建轻量版

### API Key 汇总

| 服务 | Key | 状态 |
|------|-----|------|
| NCBI PubMed | `ecb25dde...` | ✅ |
| API2D (GPT) | `fk248541-...` | ✅ |
| Semantic Scholar | 待审核 | ⏳ |

---

## 2026-08-10

### 环境搭建过程

- Python 3.11.15 + uv (比 Poetry 快，推荐)
- WSL 2 + Ubuntu 安装成功
- Docker Desktop 安装遇到 Manifest extraction failed 错误
  - 解决: 删除 temp 目录，管理员运行安装程序
  - hello-world 验证通过

### 技术选型记录

| 原方案 | 终方案 | 原因 |
|--------|--------|------|
| Poetry | uv | 更快更现代 |
| OpenAI 直充 | API2D 中转 | 国内银行卡被 Stripe 拒绝 |
| Anthropic 直连 | API2D 已支持 Claude | 无需额外配置 |

### RAG Demo 全链路验证

```
PubMed "CRISPR cancer" (3420篇)
  → ChromaDB embedding存储
    → 检索相似文献
      → GPT 生成回答 (API2D)
```

✅ 三阶段全部跑通。

### Docker Desktop 安装踩坑

1. `winget install Docker.DockerDesktop` → 下载失败
2. 官网手动下载 → Manifest extraction failed (WSL 2 选项导致)
3. 管理员运行安装程序 → 成功
4. 注销重登 → Docker 正常运行

---

## 待研究问题

- [ ] ChromaDB 大规模数据 (>10万篇) 性能表现
- [ ] 多语言文献支持 (中文 + 英文)
- [ ] Agent 自主决策循环的实现方案
- [ ] Gradio vs Streamlit 前端选型
