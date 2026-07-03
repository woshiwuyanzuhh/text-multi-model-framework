# CHANGELOG

本文件记录 TMF 项目的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

---

## [v1.0.0] — 2026-07-03

### 第一阶段：代码修复与功能完善（16 项）

#### 新增
- **Task 1**: 实现 `src/data_pre.py` 的 `clean_data()` 函数，完成 raw → pre 的数据清洗分词流水线
- **Task 2**: 训练流水线支持全量数据集（`TRAIN_SAMPLE_SIZE=-1`），移除 200 条小样本限制
- **Task 5**: 新增 `tests/test_api_v4.py`，覆盖 V4 大模型接口的正常/空文本/无效JSON/认证/超时场景
- **Task 9**: 接入 `prune_bert_layers()` 结构化剪枝到训练流水线，新增 `evaluate_pruned_model()` 评估函数
- **Task 12**: 新增 `app/logging_config.py` 统一日志配置中心，支持 stderr + 文件双通道输出
- **Task 13**: 新增 `app/utils.py`，抽取 `validate_predict_request()` 公共校验函数和 `@require_json_input` 装饰器

#### 修复
- **Task 3**: 修复前端 `API_BASE` 硬编码为 `127.0.0.1`，改为 `window.location.origin` 动态获取
- **Task 4**: 添加 Bearer Token 认证机制，`@app.before_request` 拦截未认证请求
- **Task 6**: 收紧 CORS 策略，从 `*` 全开放改为基于 `ALLOWED_ORIGINS` 环境变量的白名单
- **Task 7**: `LLM_TIMEOUT` 从硬编码改为 `os.getenv('LLM_TIMEOUT', '30')` 环境变量配置
- **Task 8**: 封装 `src/data_eda.py` 为 `run_eda()` 函数，避免模块导入时执行副作用
- **Task 10**: 修复 Dockerfile 中 `sensitive_words.txt` 缺失导致的构建失败
- **Task 11**: 修复 V3 接口无模型守卫，添加 `macbert_model is None` 检查返回错误码 `-50`
- **Task 14**: 修正 `src/utils.py` 中蒸馏评估日志将"蒸馏"误写为"量化"的文案错误

#### 变更
- **Task 15**: 训练超参数（batch_size, learning_rate, epochs, max_len, num_workers）从硬编码移入 `Config` 类，支持环境变量覆盖；修复 `persistent_workers` 在 `num_workers=0` 时的兼容问题
- **Task 16**: 添加输入文本长度限制 `MAX_TEXT_LENGTH=5000`（环境变量可配置），v1-v4 所有接口统一校验，超长返回 `code: -11`

---

### 第二阶段：生产加固与文档完善

#### 删除 — 冗余文件清理
- 删除答辩产物：`generate_ppt.py`、`TMF_Defense_PPT.pptx`、`TMF_Defense_PPT_Outline.md`、`TMF_Defense_Speech_Script.md`、`TMF_Defense_Technical_Detail.md`
- 删除诊断报告：`TMF_Prediction_Module_Timeout_Diagnosis.md`（问题已在第一阶段修复）
- 删除架构分析：`TMF_Architecture_Analysis_Report.md`（内容已体现在 README 中）
- 删除架构图生成物：`diagrams/` 目录
- 删除 Streamlit 简易前端：`front/tmf_app.py`（功能已被 `index.html` 完全覆盖）
- 删除过时 Conda 环境定义：`environment.yml`（以 `requirements.txt` 为准）
- 删除运行时日志：`logs/api_service.log`（保留 `.gitkeep`）

#### 新增 — 安全加固
- 添加 API 限流：v1/v2/v3 限流 60/min，v4 限流 10/min（环境变量可配置）
- 添加安全响应头：`X-Content-Type-Options: nosniff`、`X-Frame-Options: DENY`、`X-XSS-Protection: 1; mode=block`
- Docker 镜像安全优化：添加非 root 用户 `appuser` 运行应用
- Gunicorn 添加 `--graceful-timeout 30` 优雅关闭超时
- 新增 `.env.example` 环境变量配置模板

#### 变更 — 依赖清理
- 从 `requirements.txt` 移除 `streamlit`（Streamlit 前端已删除）
- 从 `requirements.txt` 移除 `loguru`（仅 Streamlit 前端使用）
- `requirements.txt` 新增 `flask-limiter>=3.5.0`
- `docker-compose.yml` 新增 `RATE_LIMIT_LOCAL` 和 `RATE_LIMIT_LLM` 环境变量

#### 新增 — 部署文档
- 重写 `README.md`：包含架构图、快速开始、API 文档、训练流水线、配置说明、项目结构
- 新增 `CHANGELOG.md`：完整记录两阶段所有变更

---

### 第三阶段：资深 Tech Lead Code Review 修复（15 项 P1/P2）

#### 修复 — 安全与健壮性
- **CR-1**: `app/__init__.py` — Token 校验从 `==` 改为 `secrets.compare_digest()`，防止时序攻击（Timing Attack）
- **CR-2**: `app/__init__.py` — 新增 `RateLimitExceeded` 异常处理器，返回标准 JSON `{"code": -60}` 而非默认 HTML
- **CR-3**: `app/__init__.py` — 新增 `FLASK_ENV` 环境变量校验，非法值直接 `raise RuntimeError` 拒绝启动
- **CR-4**: `app/config.py` — 修复 `logging` 模块未导入导致 `WARNING` 日志输出崩溃的 `NameError`
- **CR-5**: `app/config.py` — `MAX_TEXT_LENGTH` 的 `int()` 转换包裹 `try/except ValueError`，非法值回退默认 5000
- **CR-6**: `app/utils.py` — `validate_predict_request()` 新增 `isinstance` 类型守卫，防止 `text` 传入 `int`/`list`/`None` 时静默通过导致下游崩溃
- **CR-7**: `app/logging_config.py` — 从普通 `FileHandler` 升级为 `RotatingFileHandler`（10MB × 5 备份），防止日志文件无限增长撑满磁盘
- **CR-8**: `app/v1/predict.py` ~ `app/v3/predict.py` — 推理逻辑包裹 `try/except`，未捕获异常返回标准错误码 `-51` 而非 500 崩溃
- **CR-9**: `app/v3/predict.py` & `src/utils.py` — 推理与评估中硬编码的 `max_len=32` 统一替换为 `Config.max_len`，消除训练/推理不一致
- **CR-10**: `src/data_pre.py` — `get_corpus()` 增加格式异常行检测（缺少 tab 分隔符 / 标签非整数），跳过并记录 WARNING 而非崩溃

#### 优化 — 性能与资源管理
- **CR-11**: `app/v4/predict.py` — `system_prompt` 提升至模块级预计算，避免每次请求重复拼接字符串；OpenAI Client 改为双重检查锁懒加载
- **CR-12**: `src/utils.py` — 量化后端选择改为平台感知（ARM/Apple Silicon → `qnnpack`，x86 → `fbgemm`），消除跨平台 Segfault
- **CR-13**: `docker-compose.yml` — 新增 `mem_limit: 4g` 和 `cpus: 4.0` 资源限制，防止容器 OOM 影响宿主机

#### 修复 — 前端与容器
- **CR-14**: `front/index.html` — 修复 429 限流和非 2xx 错误响应未解析 JSON body 的问题，前端无法读取错误码导致静默失败
- **CR-15**: `.dockerignore` — 优化排除规则，排除 `data/raw/`、`data/pre/`、`tests/`、`run_pipeline.py` 等训练产物，最小化生产镜像体积

---

### 第四阶段：Windows 10+ / Python 3.12 兼容性适配

#### 新增
- 新增 `TMF_Technical_Whitepaper.md` — 项目核心技术亮点与架构白皮书（电梯演讲、架构全景图、四大技术壁垒 STAR 法则阐述、重构 Before/After 对比表、面试 Q&A 预演）

#### 修复 — Windows 平台兼容性
- **WIN-1**: `requirements.txt` — 新增 `waitress>=3.0.0` 作为 Windows WSGI 服务器（Gunicorn 依赖 `os.fork()`，仅限 Unix）
- **WIN-2**: `wsgi.py` — 补充 Windows（Waitress）与 Linux/macOS（Gunicorn）双平台启动命令说明
- **WIN-3**: `README.md` — 本地开发命令补充 Windows PowerShell（`$env:`）和 CMD（`set`）环境变量设置方式；虚拟环境激活补充 `.venv\Scripts\Activate.ps1`；新增"生产部署（本地裸机）"章节
- **WIN-4**: `.env.example` — 文件复制命令补充 Windows `copy` 替代 Unix `cp`
- **WIN-5**: `src/utils.py` — 量化后端平台检测改为 `.lower()` 大小写不敏感匹配（Windows ARM 返回大写 `'ARM64'`），并添加 `try/except RuntimeError` 兜底防止后端不可用时崩溃
- **WIN-6**: `.env.example` & `README.md` — `TRAIN_NUM_WORKERS` 添加 Windows 多进程兼容提示（Windows 使用 `spawn` 启动方式，如遇多进程报错可设为 0）

---

## [v0.0.1] — 2026-06-28

### 初始版本
- 四代文本分类 API（v1 sklearn / v2 fastText / v3 MacBERT / v4 LLM）
- Flask 应用工厂模式 + Blueprint 多版本并行
- 训练流水线：数据预处理 → MacBERT 微调 → 量化/蒸馏
- Docker 部署 + Gunicorn --preload
- 前端多版本对比面板
