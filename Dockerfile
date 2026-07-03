# ============================================================
# Dockerfile — API 推理服务镜像
# 构建: docker build -t tmf-api:latest .
# 运行: docker compose up -d
# ============================================================

FROM python:3.12-slim

LABEL maintainer="tmf-project" \
      description="TMF API Inference Service"

# 安装运行时系统依赖（curl 用于 healthcheck，libgomp1 用于 PyTorch 多线程）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先安装依赖（利用 Docker 缓存层，依赖不变时无需重新安装）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制运行时必需的数据目录（tmf_class.txt、stopwords.txt 等）
COPY data/ ./data/

# 复制应用代码
COPY app/ ./app/
COPY src/ ./src/
COPY wsgi.py .

# 复制前端页面（仅 index.html，不包含 Streamlit 前端）
COPY front/index.html ./front/index.html

# 不复制 models/（通过 volume 挂载）
# 不复制 models_quantized/ models_distilled/ models_pruned/（训练产物）

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false

# 创建非 root 用户运行应用（安全加固：避免容器内 root 权限）
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# --preload 让 worker 共享模型内存（模型在 master 进程加载一次）
# --timeout 120 给模型加载和推理留足够时间
# --graceful-timeout 30 优雅关闭超时
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "--timeout", "120", "--graceful-timeout", "30", "--preload", "wsgi:app"]
