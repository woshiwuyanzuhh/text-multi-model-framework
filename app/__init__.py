"""
Flask 应用工厂与核心生命周期管理模块

本模块负责整个 Flask 应用程序的初始化、配置加载、第三方扩展组件绑定以及业务蓝图的动态注册。
采用工厂模式（Application Factory Pattern）设计，以支持多环境隔离部署并有效规避循环导入问题。

主要职责包括:
1. 环境配置分发: 依据环境变量 `FLASK_ENV` 动态加载 `app.config` 映射类。
2. 扩展组件装配: 初始化并注入数据库 (SQLAlchemy)、缓存 (Redis) 等基础中间件。
3. 路由蓝图注册: 统一挂载各版本 API 业务流（如预测推理模块）。
4. 全局拦截兜底: 注册全局异常捕获器 (Error Handlers)，确保所有错误输出标准化为 JSON 响应。

使用示例:
    >>> from app import create_app
    >>>
    >>> app = create_app()

Author: Grant Johnny
Version: 0.0.1
"""
import logging
import os
import secrets

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException
from flask_limiter import RateLimitExceeded

from app.config import config_map
from app.extensions import thy_extension, limiter
from app.logging_config import setup_logging
from app.v1.predict import model_bp_v1
from app.v2.predict import model_bp_v2
from app.v3.predict import model_bp_v3
from app.v4.predict import model_bp_v4

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """应用工厂函数"""
    # 统一日志配置（最先执行，确保后续日志输出到 stderr 和文件）
    setup_logging()

    app = Flask(__name__)

    # 动态加载环境配置
    env = os.getenv('FLASK_ENV', 'development')
    if env not in config_map:
        raise RuntimeError(f'不支持的 FLASK_ENV={env}，可选值: {list(config_map.keys())}')
    app.config.from_object(config_map[env]())
    logger.info('Flask 应用配置加载完成 (env=%s)', env)

    # 将组件动态绑定到当前 app
    # db.init_app(app)
    # redis_client.init_app(app)
    limiter.init_app(app)
    thy_extension.init_app(app.config['MODEL_PATH'])

    # 启动守卫：环境变量校验
    _validate_env_vars(app)

    # 注册业务蓝图
    app.register_blueprint(model_bp_v1)
    app.register_blueprint(model_bp_v2)
    app.register_blueprint(model_bp_v3)
    app.register_blueprint(model_bp_v4)

    # 注册钩子函数
    register_handlers(app)

    # 健康检查端点（免认证，供 Docker healthcheck 使用）
    @app.route('/health')
    def health_check():
        return jsonify({'code': 0, 'message': 'OK'})

    # Flask 托管前端页面（Docker 内一个容器提供 API + UI）
    @app.route('/ui')
    def serve_ui():
        return send_from_directory(os.path.join(app.root_path, '..', 'front'), 'index.html')

    @app.route('/ui/<path:filename>')
    def serve_ui_static(filename):
        return send_from_directory(os.path.join(app.root_path, '..', 'front'), filename)

    return app


def register_handlers(app: Flask):
    """注册拦截钩子函数"""

    @app.before_request
    def check_auth():
        # 白名单：健康检查、前端页面免认证；OPTIONS 预检请求免认证
        if request.path == '/health' or request.path.startswith('/ui'):
            return
        if request.method == 'OPTIONS':
            return

        auth = request.headers.get('Authorization', '')
        expected = f"Bearer {app.config['API_KEY']}"
        # 使用恒定时间比较防止时序攻击
        if not secrets.compare_digest(auth, expected):
            return jsonify({'code': -1, 'message': '认证失败'}), 401

    # CORS 白名单：从环境变量读取（逗号分隔），默认仅允许本地
    allowed_origins = set(
        origin.strip()
        for origin in os.getenv('ALLOWED_ORIGINS', 'http://localhost:8080').split(',')
        if origin.strip()
    )

    @app.after_request
    def add_security_headers(response):
        # CORS 白名单
        origin = request.headers.get('Origin', '')
        if origin in allowed_origins:
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        # 安全响应头
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        return response

    @app.errorhandler(RateLimitExceeded)
    def handle_rate_limit(e):
        logger.warning('请求频率超限: %s', e.description)
        return jsonify({'code': -60, 'message': '请求频率超限，请稍后再试'}), 429

    @app.errorhandler(Exception)
    def handle_exception(e):
        # 拦截标准 HTTP 错误
        if isinstance(e, HTTPException):
            # 4xx 客户端错误记 WARNING，5xx 服务端错误记 ERROR
            if e.code >= 500:
                logger.error('服务器错误: %s', e, exc_info=True)
            else:
                logger.warning('客户端请求错误: %s %s', e.code, e.description)
            return jsonify({'code': e.code, 'message': e.description}), e.code

        # 拦截业务层未捕获的异常
        logger.error('未捕获异常: %s', e, exc_info=True)
        return jsonify({'code': 500, 'message': '服务器维护中请稍后尝试'}), 500


def _validate_env_vars(app: Flask):
    """启动守卫：环境变量校验
    检查 API_KEY 是否设置（必须）；检查 OPENAI_API_KEY 是否设置（决定 V4 可用性）"""
    # API_KEY 认证密钥（必须设置）
    api_key = os.getenv('API_KEY')
    if not api_key:
        raise RuntimeError('必须设置 API_KEY 环境变量，用于接口认证')
    app.config['API_KEY'] = api_key

    # OPENAI_API_KEY 决定 V4 大模型接口可用性
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        logger.warning('OPENAI_API_KEY 未设置，V4 大模型分类接口将不可用')
        app.config['V4_LLM_AVAILABLE'] = False
    else:
        logger.info('OPENAI_API_KEY 已设置，V4 大模型分类接口可用')
        app.config['V4_LLM_AVAILABLE'] = True
