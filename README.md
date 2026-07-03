# TMF 中文文本分类智能平台

从传统 ML 到大模型的四代技术演进与实践，支持 v1（传统ML）/ v2（fastText）/ v3（BERT）/ v4（大模型）多接口并发对比。

## 项目架构

```
┌─────────────────────────────────────────────────────────────┐
│                    TMF 项目整体架构                          │
│                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │  离线训练层   │   │  模型压缩层   │   │  在线推理层   │    │
│  │              │   │              │   │              │    │
│  │ 数据预处理   │──▶│ 模型训练     │──▶│ Flask API    │    │
│  │ EDA 分析    │   │ MacBERT 微调 │   │ (v1~v4)      │    │
│  │ 清洗+分词   │   │ 评估         │   │              │    │
│  │              │   │ 量化/蒸馏/剪枝│   │ 前端 UI      │    │
│  │              │   │ 压缩模型评估  │   │ Docker 部署  │    │
│  └──────────────┘   └──────────────┘   └──────────────┘    │
│   run_pipeline.py      src/utils.py      app/ + front/      │
└─────────────────────────────────────────────────────────────┘
```

### 四代 API 对比

| 版本 | 路由 | 模型 | 推理方式 | 限流 |
|------|------|------|---------|------|
| v1 | `/api/v1/predict` | sklearn pkl | jieba 分词 + 传统 ML | 60/min |
| v2 | `/api/v2/predict` | fastText ftz | 词向量级分类 | 60/min |
| v3 | `/api/v3/predict` | MacBERT safetensors | PyTorch 推理 | 60/min |
| v4 | `/api/v4/predict` | OpenAI LLM | Prompt + 双重超时 | 10/min |

## 快速开始

### 前置条件

- Docker 24+ 和 Docker Compose v2
- 已训练好的模型文件（放在 `models/` 目录）

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 SECRET_KEY、API_KEY 等必填项
```

### 2. 启动服务

```bash
docker compose up -d
```

### 3. 验证服务

```bash
# 健康检查
curl http://localhost:8080/health

# 文本分类
curl -X POST http://localhost:8080/api/v3/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"text": "央行宣布降低存款准备金率 0.5 个百分点"}'
```

### 4. 访问前端

浏览器打开 `http://localhost:8080/ui`

## 本地开发

### 环境准备

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat

# 安装依赖
pip install -r requirements.txt
```

### 启动开发服务器

```bash
# 设置环境变量
# Linux / macOS:
export FLASK_ENV=development
export API_KEY=dev-test-key

# Windows (PowerShell):
$env:FLASK_ENV="development"
$env:API_KEY="dev-test-key"

# Windows (CMD):
set FLASK_ENV=development
set API_KEY=dev-test-key

# 启动（开发模式，仅限本地测试）
python wsgi.py
```

### 生产部署（本地裸机）

```bash
# Linux / macOS（使用 Gunicorn，支持多 Worker + --preload 共享模型内存）:
gunicorn -w 4 -b 0.0.0.0:8080 --timeout 120 --graceful-timeout 30 --preload wsgi:app

# Windows（使用 Waitress，Gunicorn 不支持 Windows）:
waitress-serve --listen=0.0.0.0:8080 wsgi:app
```

### 运行测试

```bash
# 全部测试
pytest tests/ -v

# 仅 API 接口测试
pytest tests/ -v -m api

# 仅冒烟测试
pytest tests/ -v -m smoke
```

## 训练流水线

### 一键执行

```bash
python run_pipeline.py
```

流水线包含 6 个步骤：
1. 数据清洗（raw → pre）
2. 模型训练（MacBERT 微调）
3. 模型评估
4. 模型压缩（量化 + 蒸馏 + 剪枝）
5. 压缩模型评估
6. 完成

### 可配置训练参数

通过环境变量调整训练超参数：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `TRAIN_BATCH_SIZE` | 32 | 训练批大小 |
| `TRAIN_LEARNING_RATE` | 2e-5 | 学习率 |
| `TRAIN_EPOCHS` | 16 | 训练轮数 |
| `TRAIN_MAX_LEN` | 32 | 最大序列长度 |
| `TRAIN_NUM_WORKERS` | 4 | DataLoader 工作进程数（Windows 如遇多进程报错设为 0） |
| `TRAIN_SAMPLE_SIZE` | -1 | 训练采样数量（-1 全量） |

### Docker 训练

```bash
docker build -f Dockerfile.pipeline -t tmf-pipeline .
docker run -v $(pwd)/models:/app/models \
           -v $(pwd)/models_quantized:/app/models_quantized \
           -v $(pwd)/models_distilled:/app/models_distilled \
           -v $(pwd)/models_pruned:/app/models_pruned \
           tmf-pipeline
```

## 配置说明

### 环境变量一览

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `FLASK_ENV` | 否 | development | 运行环境 |
| `SECRET_KEY` | **生产必填** | — | Flask 密钥 |
| `API_KEY` | **是** | — | 接口认证密钥 |
| `ALLOWED_ORIGINS` | 否 | localhost:8080 | CORS 白名单 |
| `OPENAI_API_KEY` | 否 | — | V4 LLM API Key |
| `LLM_MODEL_NAME` | 否 | gpt-3.5-turbo | LLM 模型名 |
| `LLM_BASE_URL` | 否 | — | LLM API 地址 |
| `LLM_TIMEOUT` | 否 | 30 | LLM 超时（秒） |
| `MAX_TEXT_LENGTH` | 否 | 5000 | 输入文本最大长度 |
| `RATE_LIMIT_LOCAL` | 否 | 60 per minute | v1/v2/v3 限流 |
| `RATE_LIMIT_LLM` | 否 | 10 per minute | v4 限流 |

完整配置模板见 `.env.example`。

### 错误码体系

| Code | 含义 |
|------|------|
| 0 | 成功 |
| -1 | 认证失败 |
| -10 | 空文本 |
| -11 | 文本超长 |
| -20 | JSON 格式错误 |
| -30 | LLM 结果无法解析 |
| -40 | LLM 请求超时 |
| -41 | LLM 连接失败 |
| -42 | LLM 频率超限 |
| -43 | LLM 认证失败 |
| -44 | LLM API 状态错误 |
| -50 | 模型未加载 |
| -51 | 内部错误 |
| -60 | 请求频率超限 |

## 项目结构

```
tmf_v1_test/
├── app/                    # Flask 应用（在线推理服务）
│   ├── __init__.py         # 应用工厂 + 钩子注册
│   ├── config.py           # 多环境配置
│   ├── extensions.py       # 模型加载 + 限流器
│   ├── logging_config.py   # 统一日志配置
│   ├── utils.py            # 公共校验函数
│   └── v1~v4/predict.py    # 四代 API 蓝图
├── src/                    # 训练代码（离线流水线）
│   ├── config.py           # 训练配置（frozen dataclass）
│   ├── data_pre.py         # 数据清洗 + 分词
│   ├── data_eda.py         # 探索性分析
│   ├── model_train.py      # MacBERT 微调训练
│   ├── model_eval.py       # 模型评估
│   └── utils.py            # 量化/蒸馏/剪枝
├── front/
│   └── index.html          # 前端对比面板
├── data/                   # 数据目录
│   ├── raw/                # 原始数据
│   ├── pre/                # 预处理后数据
│   ├── stopwords.txt       # 停用词表
│   └── tmf_class.txt       # 分类标签
├── tests/                  # 测试用例
├── Dockerfile              # 推理服务镜像
├── Dockerfile.pipeline     # 训练流水线镜像
├── docker-compose.yml      # 一键编排
├── run_pipeline.py         # 训练流水线入口
├── wsgi.py                 # WSGI 启动入口
├── requirements.txt        # Python 依赖
└── .env.example            # 环境变量模板
```

## 技术栈

| 分类 | 技术 |
|------|------|
| Web 框架 | Flask 3.1 + Gunicorn |
| 深度学习 | PyTorch + Transformers + ModelScope |
| 传统 ML | scikit-learn + fastText + jieba |
| 大模型 | OpenAI SDK |
| 前端 | Tailwind CSS (CDN) |
| 部署 | Docker + docker-compose |
| 测试 | pytest |
| 限流 | Flask-Limiter |
