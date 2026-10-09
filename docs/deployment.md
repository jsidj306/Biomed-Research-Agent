# 部署指南（Deployment）

本文档涵盖三种部署方式：本地（uv）、Docker Compose、HuggingFace Spaces。

---

## 前置条件

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)（可选，本地开发用）
- Docker + docker-compose（容器部署用）
- 至少一个 LLM API Key（智谱 GLM / 月之暗面 Kimi，见 `.env.example`）

---

## 1. 本地部署（uv）

```bash
git clone git@github.com:jsidj306/Biomed-Research-Agent.git
cd Biomed-Research-Agent
uv sync                                   # 安装依赖
cp .env.example .env                      # 编辑填入 API Key
uv run streamlit run src/ui/app.py        # 打开 http://localhost:8501
```

可选组件：

```bash
uv add weasyprint          # PDF 导出（否则用本地 pandoc）
uv add neo4j               # 知识图谱写入（可选）
```

---

## 2. Docker Compose 部署

`docker-compose.yml` 编排三服务：`app`（Streamlit）+ `chroma`（向量库）+ `neo4j`（图库）。

```bash
cp .env.example .env       # 填入 API Key（app 服务的 env_file 会加载它）
docker-compose up -d       # 后台启动三服务
docker-compose logs -f app # 查看 app 日志
docker-compose down        # 停止（数据卷保留）
```

服务与端口：

| 服务 | 端口 | 说明 |
|------|------|------|
| app | 8501 | Streamlit UI |
| chroma | 8000 | 向量存储 API |
| neo4j | 7474（HTTP）/ 7687（Bolt） | 知识图谱 |

**注意**：容器内服务间通信用服务名做 host（`chroma` / `neo4j`），不是 `localhost`。compose
文件已通过 `CHROMA_HOST` / `NEO4J_URI` 环境变量注入正确地址。

### 数据持久化

- `chroma_data` 卷 → ChromaDB 向量
- `neo4j_data` 卷 → Neo4j 图谱数据

`docker-compose down` 不删除卷；`docker-compose down -v` 才会清空数据。

---

## 3. HuggingFace Spaces 部署

> ⚠️ **收费变更（2026-07 起）**：HuggingFace 已取消免费层的 `cpu-basic` 实例，
> **Docker / Gradio / Streamlit SDK 均需 PRO 订阅（$9/月）**；免费层只剩 Static 静态
> 站点和限制极严的 ZeroGPU。本项目是实时 Streamlit 应用、又依赖 Dockerfile（中文字体
> + uv），因此**免费部署到 HF Spaces 已不可行**。以下保留「付费 PRO」路径，并给出
> 免费替代方案。

### 3.1 方案一：付费 PRO + Docker SDK（$9/月）

1. 订阅 [HuggingFace PRO](https://huggingface.co/settings/billing)（$9/月）。
2. 访问 [huggingface.co/new-space](https://huggingface.co/new-space)，选择
   **Docker** SDK，空间名如 `biomed-research-agent`。
3. 克隆空仓库：

   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/biomed-research-agent
   cd biomed-research-agent
   ```

4. 拷贝本项目文件（`src/`、`pyproject.toml`、`uv.lock`、`Dockerfile`、`.dockerignore`）。
5. 在 Space 的 **Settings → Variables and secrets** 中配置所有环境变量（`ZHIPU_API_KEY`
   / `MOONSHOT_API_KEY` 等，见 `.env.example`），**不要**把 `.env` 提交进仓库。
6. 提交并推送：

   ```bash
   git add -A && git commit -m "deploy biomed-research-agent" && git push
   ```

7. Space 会自动构建 Docker 镜像并启动，构建完成后访问
   `https://YOUR_USERNAME-biomed-research-agent.hf.space`。

### 3.2 方案二：免费替代（无需 HF PRO）

本项目的 Dockerfile 直接可用，找一个「支持任意 Docker 容器」的免费/免费额度平台即可：

| 平台 | 免费额度 | 是否支持 Docker | 注意 |
|------|---------|----------------|------|
| **Google Cloud Run** | 每月 200 万请求免费 | ✅ | 冷启动较慢，需绑信用卡但额度内不扣费 |
| **Render** | 免费 Web Service | ✅ | 512MB RAM 偏小（PubMedBERT 可能吃紧）、闲置 15 分钟休眠 |
| **Streamlit Community Cloud** | 免费 | ❌（走 `requirements.txt`） | 需补 `requirements.txt` + `packages.txt`，不支持 Dockerfile |

> 若只是「能公网访问 demo」而非长期托管，**Streamlit Community Cloud** 上手最快：
> 关联 GitHub 仓库 → 选 `src/ui/app.py` 入口 → 在 Secrets 里配 Key。知识图谱/Neo4j
> 这类依赖外部服务的能力在免费托管上建议关闭或指向远程实例。

### 注意事项

- **Secret 管理**：API Key 必须用 Spaces 的 Secrets 面板配置，切勿写入公开仓库。
- **启动命令**：Dockerfile 的 `CMD` 已设为
  `streamlit run src/ui/app.py --server.address=0.0.0.0 --server.headless=true`，
  与 Spaces 的端口探测兼容。
- **Neo4j / Chroma**：免费 Spaces 无法跑完整三服务编排。若需知识图谱，建议用
  [Neo4j AuraDB Free](https://neo4j.com/cloud/platform/aura-graph-database/) 或本地
  Neo4j，通过 `NEO4J_URI` 环境变量指向远程实例；ChromaDB 可退回本地持久化模式。
- **冷启动**：PubMedBERT 模型首次加载较慢（下载权重），Space 冷启动会多等几十秒。

---

## 4. 生产环境清单

- [ ] `.env` 已配置全部必填 Key（NCBI、智谱、月之暗面）
- [ ] LangFuse Key 已配置（可观测性）
- [ ] Neo4j 已启动（如启用知识图谱）
- [ ] 中文字体已安装（PDF 导出用，Docker 镜像已内置 fonts-noto-cjk）
- [ ] 端口 8501 / 8000 / 7687 防火墙放行（如需外部访问）
