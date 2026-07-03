# TMF 多模型中文文本分类智能平台 — 完整使用文档

> **TMF (Text Multi-model Framework)** — 从传统机器学习到大语言模型的四代中文文本分类平台
>
> **版本**: v1.0 | **更新日期**: 2026-07-03 | **Python**: 3.10+ (推荐 3.12)

---

## 目录

- [第一章：项目概述](#第一章项目概述)
- [第二章：快速开始 (Quick Start)](#第二章快速开始-quick-start)
- [第三章：环境变量配置详解](#第三章环境变量配置详解)
- [第四章：API 接口文档](#第四章api-接口文档)
- [第五章：训练流水线使用指南](#第五章训练流水线使用指南)
- [第六章：部署与运维](#第六章部署与运维)
- [第七章：开发者指南](#第七章开发者指南)
- [第八章：FAQ（常见问题）](#第八章faq常见问题)

---

## 第一章：项目概述

### 1.1 项目简介

TMF 是一个覆盖"传统 ML → 深度学习 → 大语言模型"四代算法演进的中文文本分类平台。它提供四套并行的推理 API（V1 sklearn / V2 fastText / V3 MacBERT / V4 LLM），以及一条从数据清洗到模型压缩的完整离线训练流水线。项目采用 Flask 应用工厂模式与蓝图隔离设计，具备 Bearer Token 认证、IP 限流、日志滚动、优雅降级等企业级生产防护能力，支持 Docker Compose 一键部署。

### 1.2 核心能力

| 能力域 | 说明 |
|--------|------|
| **四代推理 API** | V1 sklearn TF-IDF + SVM、V2 fastText、V3 MacBERT 微调、V4 LLM（OpenAI 兼容接口），四代路由并存，前端只需改 URL 切换 |
| **离线训练流水线** | EDA → 数据清洗/分词 → MacBERT 微调 → 评估 → 动态量化/知识蒸馏/结构化剪枝，`run_pipeline.py` 一键执行 |
| **前端对比面板** | `front/index.html` 提供可视化界面，支持四代模型同时调用、结果对比展示 |
| **安全防护** | Bearer Token 认证（`secrets.compare_digest` 防时序攻击）、Flask-Limiter 限流、CORS 白名单、安全响应头 |
| **容器化部署** | Docker 多阶段构建、非 root 用户运行、Gunicorn `--preload` 内存共享、资源限制 (4C/4G) |

### 1.3 技术栈一览

| 分类 | 技术 | 版本要求 |
|------|------|----------|
| Web 框架 | Flask | >= 3.1 |
| WSGI 服务器 | Gunicorn (Linux/macOS) / Waitress (Windows) | >= 22.0 / >= 3.0 |
| 深度学习 | PyTorch + Transformers + ModelScope | PyTorch >= 2.0 |
| 传统 ML | scikit-learn + fastText + jieba | sklearn >= 1.0 |
| 大模型 | OpenAI Python SDK | >= 1.0.0 |
| 限流 | Flask-Limiter | >= 3.5.0 |
| 部署 | Docker + Docker Compose | Docker 24+ |
| 测试 | Pytest | >= 7.0 |
| 前端 | Tailwind CSS (CDN) | — |

### 1.4 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        接入层 (Access Layer)                      │
│  Docker (非 root) · Gunicorn 4 Worker (--preload) · Nginx (可选) │
│  Bearer Token 认证 · CORS 白名单 · 安全响应头 · IP 限流           │
├─────────────────────────────────────────────────────────────────┤
│                     应用层 (Application Layer)                    │
│  Application Factory (create_app) · Blueprint 隔离               │
│  全局异常拦截器 · 统一 Code-Message 响应体系 · 健康检查 /health   │
├──────────┬──────────┬──────────┬─────────────────────────────────┤
│  V1 API  │  V2 API  │  V3 API  │          V4 API                 │
│  sklearn │ fastText │ MacBERT  │   LLM (OpenAI 兼容)              │
│  /api/v1 │ /api/v2  │ /api/v3  │   /api/v4                       │
├──────────┴──────────┴──────────┴─────────────────────────────────┤
│                  模型推理层 (Inference Layer)                     │
│  全局单例懒加载 · torch.inference_mode() · 张量及时释放           │
│  MAX_TEXT_LENGTH 截断 · 模型未加载降级返回 code -50               │
├─────────────────────────────────────────────────────────────────┤
│                  离线训练层 (Offline Training)                    │
│  run_pipeline.py: EDA → 清洗 → MacBERT 微调 → 评估              │
│  → 动态量化(int8) → 知识蒸馏(rbt3) → 结构化剪枝(12→6层)         │
├─────────────────────────────────────────────────────────────────┤
│                  基础设施层 (Infrastructure)                      │
│  Docker Compose 编排 · 资源限制 (4C/4G) · Volume 模型热挂载      │
│  RotatingFileHandler (10MB×5) · .env 配置注入 · Healthcheck     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 第二章：快速开始 (Quick Start)

### 2.1 环境要求

| 条件 | 要求 | 备注 |
|------|------|------|
| **操作系统** | Windows 10+ / Linux / macOS | Docker 部署推荐 Linux |
| **Python** | 3.10+（推荐 3.12） | 本地开发需要 |
| **Docker** | 24+（含 Docker Compose v2） | Docker 部署需要 |
| **内存** | ≥ 8GB（MacBERT 推理约需 2GB） | — |
| **GPU** | 可选（CPU 推理可用，GPU 加速 V3） | 支持 CUDA / Apple MPS |
| **模型文件** | 预训练模型需放在 `models/` 目录 | 首次使用需先训练或获取模型 |

> ⚠️ **必须先有模型文件**：V1 需要 `models/text-clf-model.pkl`，V2 需要 `models/text-clf-model.ftz`，V3 需要 `models/` 下的 MacBERT 权重。如需训练，请先阅读[第五章](#第五章训练流水线使用指南)。

### 2.2 方式一：Docker Compose 一键启动（推荐）

```bash
# 1. 克隆项目
git clone <your-repo-url> tmf
cd tmf

# 2. 复制环境变量模板并配置
cp .env.example .env
# Windows: copy .env.example .env

# 3. 编辑 .env，填入必填项（至少设置 API_KEY 和 SECRET_KEY）
#    如果需要 V4 大模型接口，填入 OPENAI_API_KEY
notepad .env        # Windows
vim .env            # Linux/macOS

# 4. 构建并启动服务
docker compose up -d

# 5. 验证服务是否正常
curl http://localhost:8080/health
# 预期输出: {"code":0,"message":"OK"}

# 6. 查看日志（可选）
docker compose logs -f api
```

> ✅ 启动成功后，API 服务监听 `http://localhost:8080`，前端面板访问 `http://localhost:8080/ui`。

### 2.3 方式二：本地开发模式

```bash
# 1. 克隆项目
git clone <your-repo-url> tmf
cd tmf

# 2. 创建并激活虚拟环境
python -m venv .venv

# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# Windows: copy .env.example .env

# 编辑 .env 文件，至少设置：
#   API_KEY=your-api-key-here
#   FLASK_ENV=development

# 5. 设置环境变量（开发模式）
# Linux / macOS:
export FLASK_ENV=development
export API_KEY=dev-test-key

# Windows (PowerShell):
$env:FLASK_ENV="development"
$env:API_KEY="dev-test-key"

# Windows (CMD):
set FLASK_ENV=development
set API_KEY=dev-test-key

# 6. 启动开发服务器
python wsgi.py
# 服务启动在 http://127.0.0.1:8080
```

> 💡 生产环境裸机部署请使用 WSGI 服务器，不要用 `python wsgi.py`：
> - **Linux/macOS**: `gunicorn -w 4 -b 0.0.0.0:8080 --timeout 120 --graceful-timeout 30 --preload wsgi:app`
> - **Windows**: `waitress-serve --listen=0.0.0.0:8080 wsgi:app`

### 2.4 第一次调用 API

```bash
# 健康检查（免认证）
curl http://localhost:8080/health

# 调用 V3 MacBERT 分类接口
curl -X POST http://localhost:8080/api/v3/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev-test-key" \
  -d '{"text": "央行宣布降低存款准备金率0.5个百分点"}'

# 预期响应:
# {"code":0,"message":"OK","label":"财经"}
```

### 2.5 打开前端对比面板

浏览器访问：**http://localhost:8080/ui**

前端面板支持：
- 输入文本，一键调用 V1-V4 四个接口
- 并发请求，对比四代模型的分类结果
- 展示每个接口的响应时间和返回标签

> 💡 前端页面由 Flask 直接托管（`/ui` 路由），无需额外启动前端服务器。

---

## 第三章：环境变量配置详解

所有环境变量通过 `.env` 文件或系统环境变量注入。Docker Compose 会自动读取项目根目录的 `.env` 文件。

### 3.1 Flask 应用配置

| 变量名 | 必填 | 默认值 | 说明 | 示例值 |
|--------|------|--------|------|--------|
| `FLASK_ENV` | 否 | `development` | 运行环境，可选 `development` / `production` / `testing` | `production` |
| `SECRET_KEY` | **生产必填** | 随机生成 | Flask 密钥，生产环境未设置则拒绝启动 | `a1b2c3d4e5...` |
| `API_KEY` | **是** | — | Bearer Token 认证密钥，未设置则拒绝启动 | `my-secret-api-key` |
| `ALLOWED_ORIGINS` | 否 | `http://localhost:8080` | CORS 白名单，逗号分隔多个域名 | `https://app.example.com,http://localhost:3000` |
| `MAX_TEXT_LENGTH` | 否 | `5000` | API 输入文本最大长度（字符数），超长返回 code -11 | `10000` |

### 3.2 V4 大模型配置

| 变量名 | 必填 | 默认值 | 说明 | 示例值 |
|--------|------|--------|------|--------|
| `OPENAI_API_KEY` | 否 | — | OpenAI 兼容 API Key，不设置则 V4 接口不可用 | `sk-...` |
| `LLM_MODEL_NAME` | 否 | `gpt-3.5-turbo` | LLM 模型名称 | `gpt-4o-mini` |
| `LLM_BASE_URL` | 否 | 空 | LLM API 基础 URL，留空直连 `api.openai.com`；中国大陆可设代理地址 | `https://api.proxy.com/v1` |
| `LLM_TIMEOUT` | 否 | `30` | LLM 请求超时时间（秒），含线程级硬超时兜底 | `60` |

### 3.3 API 限流配置

| 变量名 | 必填 | 默认值 | 说明 | 示例值 |
|--------|------|--------|------|--------|
| `RATE_LIMIT_LOCAL` | 否 | `60 per minute` | V1/V2/V3 本地模型接口限流（按客户端 IP） | `120 per minute` |
| `RATE_LIMIT_LLM` | 否 | `10 per minute` | V4 大模型接口限流（LLM 调用成本高，限流更严格） | `5 per minute` |

### 3.4 训练参数配置

> 💡 以下变量仅在运行训练流水线时使用，推理服务不需要。

| 变量名 | 必填 | 默认值 | 说明 | 示例值 |
|--------|------|--------|------|--------|
| `TRAIN_BATCH_SIZE` | 否 | `32` | 训练批大小 | `64` |
| `TRAIN_LEARNING_RATE` | 否 | `2e-5` | 学习率 | `5e-5` |
| `TRAIN_EPOCHS` | 否 | `16` | 训练轮数 | `8` |
| `TRAIN_MAX_LEN` | 否 | `32` | 最大序列长度（Token 数） | `64` |
| `TRAIN_NUM_WORKERS` | 否 | `4` | DataLoader 工作进程数（Windows 如遇多进程报错设为 0） | `0` |
| `TRAIN_SAMPLE_SIZE` | 否 | `-1` | 训练采样数量，`-1` 表示全量数据 | `5000` |

### 3.5 环境变量优先级

```
系统环境变量 > .env 文件 > 代码默认值
```

> ⚠️ Docker Compose 部署时，`.env` 文件中的变量通过 `docker-compose.yml` 的 `environment` 段注入容器。修改 `.env` 后需重启容器：`docker compose restart api`。

---

## 第四章：API 接口文档

### 4.1 通用说明

| 项目 | 说明 |
|------|------|
| **Base URL** | `http://localhost:8080`（本地）或你的部署域名 |
| **认证方式** | 请求头 `Authorization: Bearer <API_KEY>` |
| **Content-Type** | `application/json` |
| **请求方法** | `POST`（预测接口）/ `GET`（健康检查） |
| **限流策略** | V1/V2/V3: 60 次/分钟；V4: 10 次/分钟（按客户端 IP） |
| **免认证路径** | `/health`、`/ui`、`/ui/*`、`OPTIONS` 预检请求 |

### 4.2 健康检查

```
GET /health
```

**请求示例**：

```bash
curl http://localhost:8080/health
```

**响应示例**：

```json
{
  "code": 0,
  "message": "OK"
}
```

> 💡 此端点免认证，供 Docker healthcheck 和负载均衡探针使用。

---

### 4.3 V1 — sklearn 文本分类

```
POST /api/v1/predict
```

**模型**：TF-IDF + SVM（scikit-learn），通过 jieba 中文分词后进行传统机器学习分类。

**请求头**：

| Header | 值 |
|--------|------|
| `Content-Type` | `application/json` |
| `Authorization` | `Bearer <API_KEY>` |

**请求体**：

```json
{
  "text": "湖北省黄冈市09届高三年级期末考试试题"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `text` | string | 是 | 待分类的中文文本，长度不超过 `MAX_TEXT_LENGTH`（默认 5000 字） |

**响应体（成功）**：

```json
{
  "code": 0,
  "message": "OK",
  "label": "教育"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `code` | int | 0 表示成功 |
| `message` | string | 状态描述 |
| `label` | string | 分类标签（来自 `data/tmf_class.txt`） |

**curl 示例**：

```bash
curl -X POST http://localhost:8080/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{"text": "湖北省黄冈市09届高三年级期末考试试题"}'
```

---

### 4.4 V2 — fastText 文本分类

```
POST /api/v2/predict
```

**模型**：fastText 量化模型（`.ftz` 格式），通过 jieba 分词后进行浅层神经网络分类。

**请求/响应格式**：与 V1 完全一致。

**curl 示例**：

```bash
curl -X POST http://localhost:8080/api/v2/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{"text": "央行宣布降低存款准备金率0.5个百分点"}'
```

> ⚠️ V2 接口依赖 `fasttext-wheel` 包。如未安装，模型加载失败但服务不崩溃，调用时返回 `{"code":-50,"message":"fastText 模型未加载（可能未安装 fasttext 包），请联系管理员"}`。

---

### 4.5 V3 — MacBERT 文本分类

```
POST /api/v3/predict
```

**模型**：MacBERT（`hfl/chinese-macbert-base`）微调模型，PyTorch 深度学习推理。

**请求/响应格式**：与 V1 完全一致。

**curl 示例**：

```bash
curl -X POST http://localhost:8080/api/v3/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{"text": "央行宣布降低存款准备金率0.5个百分点"}'
```

> 💡 V3 推理使用 `torch.inference_mode()` 优化性能，模型在进程启动时全局加载一次，单次推理延迟约 50ms（CPU）。

---

### 4.6 V4 — LLM 大模型文本分类

```
POST /api/v4/predict
```

**模型**：通过 OpenAI 兼容接口调用大语言模型（如 GPT-3.5/GPT-4），使用 Prompt Engineering 实现零样本分类。

**请求/响应格式**：与 V1 完全一致。

**curl 示例**：

```bash
curl -X POST http://localhost:8080/api/v4/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{"text": "央行宣布降低存款准备金率0.5个百分点"}'
```

**V4 特有行为**：

1. **双重超时保护**：OpenAI SDK 内置 `timeout` 参数 + 线程级硬超时兜底（`llm_timeout + 5` 秒），防止 DNS 解析阶段卡死导致请求线程无限阻塞。
2. **双重标签匹配**：LLM 返回文本先精确匹配分类标签，匹配失败再进行模糊匹配（包含关系），最大限度提高分类成功率。
3. **环境依赖**：需设置 `OPENAI_API_KEY`，否则返回 `{"code":-43,"message":"大模型 API 认证失败，请检查 API Key"}`。

---

### 4.7 错误码一览表

| 错误码 | HTTP 状态码 | 含义 | 触发场景 | 排查建议 |
|--------|------------|------|----------|----------|
| `0` | 200 | 成功 | 推理正常完成 | — |
| `-1` | 401 | 认证失败 | Bearer Token 缺失或错误 | 检查 `Authorization` 头格式是否为 `Bearer <API_KEY>` |
| `-10` | 200 | 请求格式错误 | `text` 字段为空或非字符串类型 | 确保请求体为 `{"text": "字符串内容"}` |
| `-11` | 200 | 文本超长 | 输入文本超过 `MAX_TEXT_LENGTH`（默认 5000 字） | 缩短文本或调大 `MAX_TEXT_LENGTH` 环境变量 |
| `-20` | 200 | JSON 格式错误 | 请求体非合法 JSON 对象，或缺少 `text` 字段 | 检查 `Content-Type: application/json` 和请求体格式 |
| `-30` | 200 | LLM 结果不可解析 | V4 大模型返回内容无法匹配任何分类标签 | 检查 `data/tmf_class.txt` 标签列表是否完整 |
| `-40` | 200 | LLM 请求超时 | V4 调用大模型超时（SDK 超时或线程硬超时触发） | 检查网络连通性，适当调大 `LLM_TIMEOUT` |
| `-41` | 200 | LLM 连接失败 | V4 无法连接到大模型 API 端点 | 检查 `LLM_BASE_URL` 是否正确，网络是否可达 |
| `-42` | 200 | LLM 频率超限 | OpenAI API 返回 RateLimit 错误 | 降低调用频率或调低 `RATE_LIMIT_LLM` |
| `-43` | 200 | LLM 认证失败 | OpenAI API Key 无效或未设置 | 检查 `OPENAI_API_KEY` 是否正确 |
| `-44` | 200 | LLM API 状态错误 | OpenAI API 返回非 2xx 状态码 | 查看日志中的 `status_code` 详情 |
| `-50` | 200 | 模型未加载 | 对应版本的模型文件缺失或加载失败 | 检查 `models/` 目录是否有对应模型文件 |
| `-51` | 200 | 推理内部错误 | 推理过程抛出未预期异常 | 查看日志中的完整堆栈信息 |
| `-60` | 429 | 请求频率超限 | 超过限流策略（60/min 或 10/min） | 降低调用频率，或调大 `RATE_LIMIT_*` 环境变量 |
| `500` | 500 | 服务器内部错误 | 全局未捕获异常 | 查看日志 `未捕获异常` 记录 |

> ⚠️ 除 `-1`（401）和 `-60`（429）外，其余业务错误码均返回 HTTP 200，通过响应体中的 `code` 字段区分。这是设计决策：前端只需解析 JSON body 即可处理所有业务逻辑，无需同时检查 HTTP 状态码。

---

## 第五章：训练流水线使用指南

### 5.1 流水线概述

```
原始数据 (data/raw/)
    │
    ▼
┌──────────────┐
│  Step 1: EDA │  探索性数据分析（标签分布、文本长度统计）
└──────┬───────┘
       ▼
┌──────────────┐
│ Step 2: 清洗 │  正则去噪 + jieba 分词 → data/pre/
└──────┬───────┘
       ▼
┌──────────────┐
│ Step 3: 训练 │  MacBERT 微调 (AdamW + CrossEntropyLoss)
└──────┬───────┘
       ▼
┌──────────────┐
│ Step 4: 评估 │  验证集 100 次随机采样准确率
└──────┬───────┘
       ▼
┌──────────────────────────────────────────────┐
│ Step 5: 模型压缩（三者并行）                   │
│  ├─ 动态量化 (int8, nn.Linear → qint8)        │
│  ├─ 知识蒸馏 (MacBERT → rbt3, KL 散度)        │
│  └─ 结构化剪枝 (12 层 Transformer → 6 层)     │
└──────┬───────────────────────────────────────┘
       ▼
┌──────────────┐
│ Step 6: 评估 │  量化/蒸馏/剪枝模型分别评估
└──────────────┘
```

### 5.2 数据准备

#### 数据格式

训练数据为 **TSV 格式**（Tab 分隔），每行一条样本：

```
<文本内容>\t<标签编号>
```

示例：

```
央行宣布降低存款准备金率0.5个百分点	0
湖北省黄冈市09届高三年级期末考试试题	1
```

#### 文件放置路径

| 文件 | 路径 | 说明 |
|------|------|------|
| 训练集 | `data/raw/tmf_train.txt` | 用于模型微调 |
| 测试集 | `data/raw/tmf_test.txt` | 用于测试 |
| 验证集 | `data/raw/tmf_valid.txt` | 用于评估 |
| 分类标签 | `data/tmf_class.txt` | 每行一个标签名，行号对应标签编号 |
| 停用词 | `data/stopwords.txt` | 每行一个停用词 |

> ⚠️ `data/tmf_class.txt` 的行号必须与 TSV 中的标签编号一一对应。例如标签编号 `0` 对应文件第 1 行，编号 `1` 对应第 2 行。

### 5.3 运行全流程

```bash
# 设置环境变量（可选，不设则使用默认值）
# Linux/macOS:
export TRAIN_EPOCHS=8
export TRAIN_BATCH_SIZE=64

# Windows (PowerShell):
$env:TRAIN_EPOCHS="8"
$env:TRAIN_BATCH_SIZE="64"

# 一键执行完整流水线
python run_pipeline.py
```

输出示例：

```
[2026-07-03 10:00:00] INFO __main__: === [Step 1/6] 开始执行数据清洗和准备工作 ===
[2026-07-03 10:00:01] INFO src.data_pre: 清洗完成: tmf_train.txt -> tmf_train.txt (20000 条)
...
[2026-07-03 10:05:00] INFO src.model_train: Epoch[1/8], Loss: 1.2345
...
[2026-07-03 10:15:00] INFO src.utils: 模型量化完成
[2026-07-03 10:15:01] INFO src.utils: 原始模型大小: 398.50 MB
[2026-07-03 10:15:01] INFO src.utils: 量化模型大小: 99.80 MB
...
[2026-07-03 10:20:00] INFO __main__: === [Step 6/6] 流水线执行完毕 ===
```

### 5.4 分步运行

如需单独执行某个步骤，可直接导入对应模块调用：

```python
# 单独执行数据清洗
from src.data_pre import clean_data
clean_data()

# 单独执行模型训练
from src.model_train import train_model
train_model()

# 单独执行模型评估
from src.model_eval import evaluate_model
evaluate_model()

# 单独执行量化
from src.utils import quantize_model, evaluate_quantized_model
quantize_model()
evaluate_quantized_model()

# 单独执行知识蒸馏
from src.utils import distill_model, evaluate_distilled_model
distill_model()
evaluate_distilled_model()

# 单独执行剪枝
from src.utils import prune_bert_layers, evaluate_pruned_model
prune_bert_layers()
evaluate_pruned_model()

# 单独执行 EDA
from src.data_eda import run_eda
run_eda()
```

### 5.5 训练超参数调整

所有超参数通过环境变量覆盖，无需修改代码：

```bash
# 示例：使用更大的 batch_size 和更少的 epochs
# Linux/macOS:
export TRAIN_BATCH_SIZE=64
export TRAIN_EPOCHS=8
export TRAIN_LEARNING_RATE=3e-5
export TRAIN_MAX_LEN=64
export TRAIN_SAMPLE_SIZE=10000  # 仅用 1 万条训练
python run_pipeline.py

# Windows (PowerShell):
$env:TRAIN_BATCH_SIZE="64"
$env:TRAIN_EPOCHS="8"
$env:TRAIN_LEARNING_RATE="3e-5"
$env:TRAIN_MAX_LEN="64"
$env:TRAIN_SAMPLE_SIZE="10000"
python run_pipeline.py
```

> 💡 超参数定义在 `src/config.py` 的 `@dataclass(frozen=True) class Config` 中，`frozen=True` 防止运行时意外修改，保证实验可复现。

### 5.6 训练产出物说明

| 产出物 | 路径 | 说明 |
|--------|------|------|
| **MacBERT 微调模型** | `models/` | 完整的 HuggingFace 格式模型（`model.safetensors` + `config.json` + tokenizer 文件），V3 推理直接使用 |
| **量化模型** | `models_quantized/` | int8 动态量化模型（`pytorch_model.bin`），体积约为原始的 1/4 |
| **蒸馏模型** | `models_distilled/` | rbt3 学生模型（3 层 Transformer），体积约为原始的 1/4 |
| **剪枝模型** | `models_pruned/` | 裁剪为 6 层的 MacBERT，层数减半 |
| **清洗后数据** | `data/pre/` | jieba 分词后的 fastText 格式数据（`__label__N word1 word2 ...`） |

> ⚠️ 推理服务默认使用 `models/` 目录下的原始微调模型。如需使用量化/蒸馏/剪枝模型，需修改 `app/extensions.py` 中的模型加载路径或通过 Volume 挂载替换。

---

## 第六章：部署与运维

### 6.1 Docker 部署详解

#### Dockerfile 解读

```dockerfile
FROM python:3.12-slim              # 基础镜像：精简版 Python 3.12

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libgomp1                  # curl: healthcheck; libgomp1: PyTorch 多线程
    && rm -rf /var/lib/apt/lists/* # 清理 apt 缓存减小镜像体积

# 先复制 requirements.txt 安装依赖（利用 Docker 缓存层）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制运行时必需文件
COPY data/ ./data/                 # 分类标签 + 停用词
COPY app/ ./app/                   # Flask 应用
COPY src/ ./src/                   # 配置 + 数据处理
COPY wsgi.py .                     # WSGI 入口
COPY front/index.html ./front/     # 前端页面

# 非 root 用户（安全加固）
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

# Gunicorn 启动命令
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", \
     "--timeout", "120", "--graceful-timeout", "30", \
     "--preload", "wsgi:app"]
```

**关键设计**：

| 设计点 | 原因 |
|--------|------|
| `python:3.12-slim` | 比 `python:3.12` 镜像小 ~200MB，不含编译工具链 |
| 依赖先复制安装 | `requirements.txt` 变化频率低，利用 Docker 缓存层加速构建 |
| 不复制 `models/` | 模型文件大（~400MB），通过 Volume 挂载，避免镜像膨胀 |
| 不复制 `data/raw/` `data/pre/` | 训练数据推理不需要，通过 `.dockerignore` 排除 |
| 非 root 用户 | 容器安全最佳实践，即使被攻破也无法获取 root 权限 |
| `--preload` | Master 进程加载模型，4 个 Worker 通过 CoW 共享内存（~400MB 而非 1.6GB） |

#### docker-compose.yml 解读

```yaml
services:
  api:
    ports:
      - "8080:8080"                # 端口映射
    environment:
      - FLASK_ENV=production       # 生产环境
      - SECRET_KEY=${SECRET_KEY}   # 从 .env 注入
      - API_KEY=${API_KEY}         # 从 .env 注入
      # ...其他环境变量
    volumes:
      - ./models:/app/models:ro    # 模型目录只读挂载，热替换无需重建镜像
      - ./logs:/app/logs           # 日志目录可写挂载，持久化到宿主机
    mem_limit: 4g                  # 内存上限，防止 OOM 影响宿主机
    cpus: 4.0                      # CPU 核数限制
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s                # 每 30 秒检查一次
      timeout: 10s                 # 超时 10 秒视为失败
      retries: 3                   # 连续 3 次失败标记为 unhealthy
      start_period: 60s            # 启动后 60 秒内失败不计入 retries
    restart: unless-stopped        # 容器异常退出自动重启
```

**常用操作**：

```bash
# 构建并启动
docker compose up -d

# 查看服务状态
docker compose ps

# 查看实时日志
docker compose logs -f api

# 重启服务（修改 .env 后生效）
docker compose restart api

# 停止并删除容器
docker compose down

# 重新构建镜像（修改代码后）
docker compose up -d --build
```

### 6.2 Gunicorn 生产配置

#### 当前配置

```bash
gunicorn -w 4 -b 0.0.0.0:8080 --timeout 120 --graceful-timeout 30 --preload wsgi:app
```

| 参数 | 值 | 说明 |
|------|------|------|
| `-w` | 4 | Worker 进程数，建议 `2 * CPU核数 + 1` |
| `-b` | `0.0.0.0:8080` | 绑定地址和端口 |
| `--timeout` | 120 | 请求超时（秒），模型加载和推理需要时间 |
| `--graceful-timeout` | 30 | 优雅关闭超时，Worker 有 30 秒处理完已接收请求 |
| `--preload` | 启用 | Master 进程预加载模型，Worker 通过 CoW 共享内存 |

#### `--preload` 内存优化原理

```
不使用 --preload:
  Master 进程 (无模型) → fork → Worker 1 (加载模型 ~400MB)
                         fork → Worker 2 (加载模型 ~400MB)
                         fork → Worker 3 (加载模型 ~400MB)
                         fork → Worker 4 (加载模型 ~400MB)
  总内存: ~1.6GB

使用 --preload:
  Master 进程 (加载模型 ~400MB) → fork → Worker 1 (CoW 共享)
                                  fork → Worker 2 (CoW 共享)
                                  fork → Worker 3 (CoW 共享)
                                  fork → Worker 4 (CoW 共享)
  总内存: ~400MB + 少量 Worker 独立内存
```

#### Worker 数量建议

| 场景 | Worker 数 | 原因 |
|------|-----------|------|
| CPU 推理（2 核） | 4-5 | `2 * CPU核数 + 1` |
| CPU 推理（4 核） | 4 | 当前默认配置 |
| CPU 推理（8 核） | 4-6 | 受限于内存，不宜过多 |
| GPU 推理 | 1-2 | GPU 显存共享，过多 Worker 会争抢显存 |

> ⚠️ Windows 不支持 Gunicorn（依赖 `os.fork()`），请使用 `waitress-serve --listen=0.0.0.0:8080 wsgi:app`。

### 6.3 日志管理

#### 日志位置与格式

| 项目 | 值 |
|------|------|
| **日志文件** | `logs/api_service.log` |
| **stderr 输出** | 同时输出到 stderr（Docker 可通过 `docker compose logs` 查看） |
| **日志格式** | `[%(asctime)s] %(levelname)s %(name)s: %(message)s` |
| **示例** | `[2026-07-03 10:00:00,123] INFO app.v3: 收到分类请求` |

#### 滚动策略

| 配置 | 值 |
|------|------|
| 单文件最大 | 10MB |
| 保留备份数 | 5 个 |
| 总最大占用 | ~50MB |
| 备份文件命名 | `api_service.log.1`、`api_service.log.2` ... |

#### 日志级别

| 级别 | 触发场景 |
|------|----------|
| `INFO` | 正常请求、模型加载成功、配置加载完成 |
| `WARNING` | 4xx 客户端错误、`OPENAI_API_KEY` 未设置、请求频率超限 |
| `ERROR` | 5xx 服务端错误、模型加载失败、推理异常、未捕获异常 |

#### 查看与清理日志

```bash
# 查看实时日志（Docker 部署）
docker compose logs -f api

# 查看日志文件（本地部署）
tail -f logs/api_service.log        # Linux/macOS
Get-Content logs\api_service.log -Wait  # Windows PowerShell

# 搜索错误日志
grep "ERROR" logs/api_service.log   # Linux/macOS
Select-String "ERROR" logs\api_service.log  # Windows

# 手动清理日志（通常不需要，RotatingFileHandler 会自动滚动）
rm logs/api_service.log.*           # Linux/macOS
del logs\api_service.log.*          # Windows
```

### 6.4 健康检查与监控

#### `/health` 端点

```bash
curl http://localhost:8080/health
# {"code":0,"message":"OK"}
```

此端点免认证，返回 HTTP 200 + `{"code":0}` 表示服务正常。

#### Docker 健康检查

`docker-compose.yml` 已配置 healthcheck：

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
  interval: 30s       # 每 30 秒检查一次
  timeout: 10s        # 超时 10 秒视为失败
  retries: 3          # 连续 3 次失败标记为 unhealthy
  start_period: 60s   # 启动后 60 秒内不检查（等模型加载）
```

```bash
# 查看健康状态
docker inspect --format='{{.State.Health.Status}}' tmf-api
# 输出: healthy / unhealthy / starting
```

### 6.5 常见问题排查

| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| 启动报错 `必须设置 API_KEY` | 未设置 `API_KEY` 环境变量 | 在 `.env` 中设置 `API_KEY=your-key`，重启服务 |
| 启动报错 `生产环境必须设置 SECRET_KEY` | 生产环境未设置 `SECRET_KEY` | 在 `.env` 中设置 `SECRET_KEY=<随机字符串>` |
| 调用返回 `code: -50` | 对应版本的模型文件缺失 | 检查 `models/` 目录是否有 pkl/ftz/safetensors 文件 |
| 调用返回 `code: -43` | `OPENAI_API_KEY` 未设置或无效 | 在 `.env` 中设置有效的 `OPENAI_API_KEY` |
| 调用返回 `code: -60` (429) | 请求频率超过限流阈值 | 降低调用频率，或调大 `RATE_LIMIT_LOCAL` / `RATE_LIMIT_LLM` |
| V3 推理极慢（秒级） | 模型未使用 `--preload` 共享，每次请求重新加载 | 确保使用 Gunicorn `--preload` 启动 |
| Docker 容器 `unhealthy` | 服务启动失败或健康检查超时 | `docker compose logs api` 查看错误日志 |
| Windows 上 `gunicorn` 命令找不到 | Gunicorn 不支持 Windows | 使用 `waitress-serve --listen=0.0.0.0:8080 wsgi:app` |
| 内存持续增长 | 推理未使用 `torch.inference_mode()` | 确认代码中 V3 推理使用了 `with torch.inference_mode():` |

---

## 第七章：开发者指南

### 7.1 项目文件结构

```
tmf_v1_test/
├── app/                              # Flask 应用层（在线推理服务）
│   ├── __init__.py                   #   应用工厂 create_app() + 全局钩子（认证/异常/CORS）
│   ├── config.py                     #   多环境配置类（Dev/Test/Prod）
│   ├── extensions.py                 #   模型加载单例 TextClassifierExtension + Flask-Limiter
│   ├── utils.py                      #   公共校验函数 validate_predict_request()
│   ├── logging_config.py             #   统一日志配置（RotatingFileHandler）
│   ├── v1/
│   │   ├── __init__.py               #   包初始化
│   │   └── predict.py                #   V1 sklearn 推理蓝图
│   ├── v2/
│   │   ├── __init__.py
│   │   └── predict.py                #   V2 fastText 推理蓝图
│   ├── v3/
│   │   ├── __init__.py
│   │   └── predict.py                #   V3 MacBERT 推理蓝图
│   └── v4/
│       ├── __init__.py
│       └── predict.py                #   V4 LLM 推理蓝图
├── src/                              # 离线训练层
│   ├── config.py                     #   训练配置（@dataclass frozen=True）
│   ├── data_eda.py                   #   探索性数据分析
│   ├── data_pre.py                   #   数据清洗 + jieba 分词
│   ├── model_train.py                #   MacBERT 微调训练
│   ├── model_eval.py                 #   模型评估
│   └── utils.py                      #   量化/蒸馏/剪枝
├── front/
│   └── index.html                    #   前端对比面板（Tailwind CSS CDN）
├── tests/                            # 自动化测试
│   ├── test_api_v1.py                #   V1 接口测试
│   ├── test_api_v2.py                #   V2 接口测试
│   ├── test_api_v3.py                #   V3 接口测试
│   └── test_api_v4.py                #   V4 接口测试
├── data/                             # 数据目录
│   ├── raw/                          #   原始数据（TSV 格式）
│   ├── pre/                          #   清洗后数据（fastText 格式）
│   ├── stopwords.txt                 #   停用词表
│   └── tmf_class.txt                 #   分类标签列表
├── models/                           # 模型文件（MacBERT + pkl + ftz）
├── models_quantized/                 # 量化模型产出
├── models_distilled/                 # 蒸馏模型产出
├── models_pruned/                    # 剪枝模型产出
├── logs/                             # 日志目录
├── run_pipeline.py                   # 训练流水线入口
├── wsgi.py                           # WSGI 启动入口
├── Dockerfile                        # 推理服务镜像构建
├── docker-compose.yml                # 容器编排
├── .env.example                      # 环境变量模板
├── .dockerignore                     # Docker 构建排除规则
├── requirements.txt                  # Python 依赖清单
├── README.md                         # 项目说明
├── CHANGELOG.md                      # 变更日志
└── USAGE.md                          # 本文档
```

### 7.2 如何新增一个模型版本（如 V5）

以下以新增 V5 为例，展示完整的添加流程：

#### Step 1：创建蓝图目录和文件

```bash
mkdir app/v5
touch app/v5/__init__.py
```

创建 `app/v5/predict.py`：

```python
"""
predict - API 接口实现（V5 新模型版本）

Author: Grant Johnny
Version: 0.0.1
"""
import logging
import os

from flask import Blueprint

from app.extensions import thy_extension, limiter
from app.utils import validate_predict_request

logger = logging.getLogger('app.v5')

# 创建蓝图对象
model_bp_v5 = Blueprint('models_v5', __name__, url_prefix='/api/v5')


@model_bp_v5.route('/predict', methods=['GET', 'POST'])
@limiter.limit(os.getenv('RATE_LIMIT_LOCAL', '60 per minute'))
def text_clf_predict():
    """文本分类预测"""
    json_string, error = validate_predict_request()
    if error:
        return error

    # 检查模型是否加载
    if thy_extension.your_new_model is None:
        return {'code': -50, 'message': 'V5 模型未加载，请联系管理员'}

    text = json_string.get('text', '')
    try:
        # 你的推理逻辑
        label = thy_extension.your_new_model.predict(text)
    except Exception as e:
        logger.error('V5 推理异常: %s', e, exc_info=True)
        return {'code': -51, 'message': '分类服务内部错误'}
    return {'code': 0, 'message': 'OK', 'label': label}
```

#### Step 2：在扩展中加载新模型

编辑 `app/extensions.py`，在 `TextClassifierExtension.init_app()` 中添加模型加载逻辑：

```python
# 4. V5 新模型
if not self.your_new_model:
    try:
        # 你的模型加载逻辑
        self.your_new_model = load_your_model(...)
        logger.info('V5 模型加载完成')
    except Exception as e:
        logger.error('V5 模型加载失败: %s', e, exc_info=True)
        self.your_new_model = None
```

#### Step 3：注册蓝图

编辑 `app/__init__.py`，在 `create_app()` 中注册新蓝图：

```python
from app.v5.predict import model_bp_v5

# 在 register_blueprints 区域添加
app.register_blueprint(model_bp_v5)
```

#### Step 4：添加测试

创建 `tests/test_api_v5.py`，参照 `tests/test_api_v1.py` 编写测试用例。

#### Step 5：验证

```bash
# 启动服务后测试
curl -X POST http://localhost:8080/api/v5/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{"text": "测试文本"}'
```

### 7.3 如何修改训练配置

训练配置定义在 `src/config.py` 的 `Config` 类中，使用 `@dataclass(frozen=True)` 装饰器：

```python
@dataclass(frozen=True)
class Config:
    pretrained_model: str = 'hfl/chinese-macbert-base'
    batch_size:        int = int(os.getenv('TRAIN_BATCH_SIZE', '32'))
    learning_rate:     float = float(os.getenv('TRAIN_LEARNING_RATE', '2e-5'))
    epochs:            int = int(os.getenv('TRAIN_EPOCHS', '16'))
    max_len:           int = int(os.getenv('TRAIN_MAX_LEN', '32'))
    # ...
```

**修改方式**：

1. **环境变量覆盖（推荐）**：通过环境变量临时修改，不改代码：
   ```bash
   export TRAIN_EPOCHS=8
   python run_pipeline.py
   ```

2. **修改默认值**：直接修改 `src/config.py` 中的默认值（影响所有未设置环境变量的运行）。

3. **新增配置项**：在 `Config` 类中添加字段，注意使用 `os.getenv()` 支持环境变量覆盖。

> ⚠️ `frozen=True` 意味着实例创建后不可修改属性值。如需运行时动态配置，请使用环境变量。

### 7.4 如何运行测试

#### 前置条件

测试需要服务已在 `http://127.0.0.1:8080` 运行（测试通过 HTTP 调用 API）。

```bash
# 先启动服务
python wsgi.py

# 或 Docker 部署
docker compose up -d
```

#### 运行测试

```bash
# 设置 API_KEY 环境变量（测试需要认证）
# Linux/macOS:
export API_KEY=your-api-key

# Windows (PowerShell):
$env:API_KEY="your-api-key"

# 运行全部测试
pytest tests/ -v

# 运行单个文件的测试
pytest tests/test_api_v3.py -v

# 仅运行 API 接口测试
pytest tests/ -v -m api

# 仅运行冒烟测试（健康检查）
pytest tests/ -v -m smoke

# 生成覆盖率报告（需安装 pytest-cov）
pytest tests/ --cov=app --cov-report=html
# 打开 htmlcov/index.html 查看报告
```

#### 测试用例覆盖

| 测试文件 | 用例数 | 覆盖场景 |
|----------|--------|----------|
| `test_api_v1.py` | 5 | 健康检查、正常预测、无效 JSON、文本超长、无认证 |
| `test_api_v2.py` | 4 | 健康检查、正常预测、无效 JSON、文本超长 |
| `test_api_v3.py` | 4 | 健康检查、正常预测、无效 JSON、文本超长 |
| `test_api_v4.py` | 9 | 健康检查、正常预测、空文本、无效 JSON、文本超长、无认证、错误认证、LLM 不可用、超时 |

### 7.5 代码规范与提交约定

#### 代码规范

- Python 代码遵循 PEP 8
- 每个模块开头包含 docstring（模块描述 + Author + Version）
- 使用 `logging` 模块输出日志，禁止 `print()`
- 函数和类包含类型注解（Type Hints）
- 配置项使用 `dataclass` 或环境变量管理，禁止硬编码

#### 提交约定

建议遵循 Conventional Commits 规范：

```
<type>(<scope>): <subject>

feat(v5): 新增 V5 模型推理接口
fix(auth): 修复 Token 校验时序攻击漏洞
docs(readme): 更新部署文档
refactor(config): 训练配置改为 dataclass frozen
chore(docker): 优化 .dockerignore 排除规则
```

| Type | 说明 |
|------|------|
| `feat` | 新功能 |
| `fix` | Bug 修复 |
| `docs` | 文档变更 |
| `refactor` | 重构（无功能变化） |
| `chore` | 构建/工具/依赖变更 |
| `test` | 测试相关 |

---

## 第八章：FAQ（常见问题）

### Q1：启动时报错 `必须设置 API_KEY 环境变量` 怎么办？

**原因**：`API_KEY` 是必填环境变量，用于 API 接口的 Bearer Token 认证。未设置时服务拒绝启动。

**解决步骤**：

```bash
# 1. 确认 .env 文件存在且包含 API_KEY
cat .env | grep API_KEY
# Windows: type .env | findstr API_KEY

# 2. 如果没有 .env 文件，从模板复制
cp .env.example .env
# Windows: copy .env.example .env

# 3. 编辑 .env，设置 API_KEY
API_KEY=your-secret-api-key

# 4. Docker 部署需重启容器
docker compose restart api

# 5. 本地部署需重新设置环境变量并重启
export API_KEY=your-secret-api-key
python wsgi.py
```

---

### Q2：V3 接口返回 `code: -50`（模型未加载）怎么解决？

**原因**：MacBERT 模型文件缺失或加载失败。V3 依赖 `models/` 目录下的 HuggingFace 格式模型文件（`config.json`、`model.safetensors`、tokenizer 文件等）。

**解决步骤**：

```bash
# 1. 检查 models/ 目录内容
ls models/
# 应包含: config.json, model.safetensors, vocab.txt, tokenizer_config.json 等

# 2. 如果目录为空，需要先训练模型
python run_pipeline.py

# 3. 如果已有模型文件但加载失败，查看日志
grep "MacBERT" logs/api_service.log
# Windows: Select-String "MacBERT" logs\api_service.log

# 4. Docker 部署时，确认 models 目录已挂载
docker compose exec api ls /app/models/
```

---

### Q3：V4 接口调用大模型一直超时怎么办？

**原因**：可能是网络无法访问 OpenAI API、`LLM_BASE_URL` 配置错误、或超时时间设置过短。

**解决步骤**：

```bash
# 1. 检查 OPENAI_API_KEY 是否设置
echo $OPENAI_API_KEY
# Windows PowerShell: echo $env:OPENAI_API_KEY

# 2. 检查网络连通性（如有 LLM_BASE_URL 则测试该地址）
curl -I https://api.openai.com/v1/models

# 3. 中国大陆用户需设置代理/中转地址
# 在 .env 中设置:
LLM_BASE_URL=https://your-proxy-address/v1

# 4. 适当调大超时时间
LLM_TIMEOUT=60

# 5. 重启服务使配置生效
docker compose restart api
```

---

### Q4：Docker 容器状态显示 `unhealthy` 怎么排查？

**原因**：容器健康检查失败，通常是服务启动失败、端口未监听、或 healthcheck 超时。

**解决步骤**：

```bash
# 1. 查看容器状态
docker compose ps

# 2. 查看健康检查历史
docker inspect --format='{{json .State.Health}}' tmf-api | python -m json.tool

# 3. 查看服务日志
docker compose logs --tail=100 api

# 4. 常见原因：
#    - API_KEY 或 SECRET_KEY 未设置 → 在 .env 中配置
#    - 模型文件未挂载 → 确认 ./models:/app/models:ro 挂载正常
#    - 启动时间超过 start_period(60s) → 调大 start_period 或增加机器配置
```

---

### Q5：如何在不重新构建 Docker 镜像的情况下更新模型？

**原因**：重新训练了模型，想热更新到线上服务。

**解决步骤**：

```bash
# 1. 将新模型文件复制到宿主机的 models/ 目录
cp /path/to/new-model/* models/

# 2. 重启容器（Gunicorn 会重新加载模型）
docker compose restart api

# 3. 验证模型已更新
curl -X POST http://localhost:8080/api/v3/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{"text": "测试文本"}'
```

> 💡 `docker-compose.yml` 中 `./models:/app/models:ro` 以只读方式挂载模型目录，更新宿主机文件后重启容器即可生效，无需重新构建镜像。

---

### Q6：Windows 上运行训练流水线报多进程错误怎么办？

**原因**：Windows 使用 `spawn` 方式创建子进程（而非 Linux 的 `fork`），DataLoader 的 `num_workers > 0` 可能导致 `RuntimeError`。

**解决步骤**：

```bash
# 将 num_workers 设为 0（使用主进程加载数据，不创建子进程）
# Windows (PowerShell):
$env:TRAIN_NUM_WORKERS="0"
python run_pipeline.py

# 或在 .env 文件中设置:
# TRAIN_NUM_WORKERS=0
```

---

### Q7：调用 API 返回 `code: -60`（429 限流）怎么调整限流策略？

**原因**：请求频率超过了限流阈值（V1/V2/V3 默认 60 次/分钟，V4 默认 10 次/分钟）。

**解决步骤**：

```bash
# 1. 在 .env 中调整限流策略
# V1/V2/V3 限流调大到 120 次/分钟:
RATE_LIMIT_LOCAL=120 per minute

# V4 限流调大到 20 次/分钟:
RATE_LIMIT_LLM=20 per minute

# 2. 重启服务
docker compose restart api

# 3. 或临时关闭限流（仅测试环境，不推荐生产使用）
RATE_LIMIT_LOCAL=100000 per minute
```

> ⚠️ V4 调大限流会增加 LLM API 调用成本，请根据预算谨慎调整。

---

### Q8：日志文件在哪里？如何查看实时日志？

**日志位置**：`logs/api_service.log`

```bash
# Docker 部署 — 查看容器日志（实时）
docker compose logs -f api

# 本地部署 — 查看日志文件（实时）
# Linux/macOS:
tail -f logs/api_service.log

# Windows (PowerShell):
Get-Content logs\api_service.log -Wait

# 搜索特定级别的日志
grep "ERROR" logs/api_service.log
Select-String "ERROR" logs\api_service.log   # Windows

# 日志文件会自动滚动（10MB × 5 备份），无需手动清理
```

---

### Q9：如何同时对比四代模型的分类效果？

**方式一：前端面板**

浏览器访问 `http://localhost:8080/ui`，输入文本后点击"四代对比"按钮，前端会并发调用 V1-V4 四个接口并展示结果。

**方式二：批量脚本**

```bash
#!/bin/bash
TEXT="央行宣布降低存款准备金率0.5个百分点"
API_KEY="your-api-key"

for version in v1 v2 v3 v4; do
  echo "=== $version ==="
  curl -s -X POST "http://localhost:8080/api/$version/predict" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    -d "{\"text\": \"$TEXT\"}"
  echo
done
```

---

### Q10：如何在前端项目中对接 TMF API？

**请求示例（JavaScript）**：

```javascript
const response = await fetch('http://your-server:8080/api/v3/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer your-api-key'
  },
  body: JSON.stringify({
    text: '央行宣布降低存款准备金率0.5个百分点'
  })
});

const result = await response.json();
console.log(result);
// { code: 0, message: "OK", label: "财经" }
```

**注意事项**：

1. **CORS**：如前端域名不在 `ALLOWED_ORIGINS` 白名单中，请求会被浏览器拦截。在 `.env` 中添加前端域名：`ALLOWED_ORIGINS=https://your-frontend.com,http://localhost:3000`
2. **错误处理**：所有业务错误（包括 `-10`、`-50` 等）均返回 HTTP 200，需通过 `response.json()` 中的 `code` 字段判断成功/失败
3. **限流**：前端高频调用需注意限流策略（60/min），建议加防抖或队列

---

### Q11：模型量化、蒸馏、剪枝有什么区别？应该用哪个？

| 技术 | 原理 | 体积压缩比 | 精度损失 | 适用场景 |
|------|------|------------|----------|----------|
| **动态量化** | `nn.Linear` 层权重 float32 → int8 | ~4× | 极小（<1%） | 通用场景，推理加速 |
| **知识蒸馏** | 大模型(MacBERT 12层) → 小模型(rbt3 3层) | ~4× | 较小（1-3%） | 边缘设备，极致轻量 |
| **结构化剪枝** | 切除 Transformer 层（12层 → 6层） | ~2× | 中等（2-5%） | 平衡精度与速度 |

**建议**：优先使用量化（精度损失最小），如需部署到资源极度受限的设备再考虑蒸馏或剪枝。三种压缩模型均已包含评估函数，可对比准确率后选择。

---

### Q12：项目支持 GPU 加速吗？如何启用？

**支持**。项目自动检测设备：

```python
self.device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu'
)
```

**启用方式**：

- **NVIDIA GPU**：安装 CUDA 版 PyTorch（`pip install torch --index-url https://download.pytorch.org/whl/cu121`），确保 `torch.cuda.is_available()` 返回 `True`
- **Apple Silicon (M1/M2/M3)**：安装 MPS 版 PyTorch，`torch.backends.mps.is_available()` 返回 `True`
- **CPU**：默认无需额外配置

> 💡 Docker 部署使用 GPU 需安装 NVIDIA Container Toolkit 并在 `docker-compose.yml` 中添加 `deploy.resources.reservations.devices` 配置。

---

> **文档结束** — 如有问题请查阅 [CHANGELOG.md](CHANGELOG.md) 了解版本变更历史，或参阅 [TMF_Technical_Whitepaper.md](TMF_Technical_Whitepaper.md) 了解核心技术架构细节。
