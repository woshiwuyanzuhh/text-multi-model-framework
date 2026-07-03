"""
config - 全局配置项

Author: Grant Johnny
Version: 0.0.1
"""
import logging
import os
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _get_secret_key():
    """安全修复：移除硬编码密钥 fallback
    开发环境：未设置 SECRET_KEY 时随机生成并打印 WARNING
    生产环境：未设置 SECRET_KEY 时直接拒绝启动"""
    key = os.getenv('SECRET_KEY')
    if key:
        return key
    if os.getenv('FLASK_ENV') == 'production':
        raise RuntimeError('生产环境必须设置 SECRET_KEY 环境变量，拒绝使用随机生成值')
    key = secrets.token_hex(32)
    logging.warning('SECRET_KEY 未设置，已随机生成临时密钥。建议设置环境变量以避免每次重启 session 失效。')
    return key


class BaseConfig:
    """基础配置类（所有环境共享的公共配置）"""

    # 核心安全密匙 - 安全修复：移除硬编码密钥 fallback
    SECRET_KEY = _get_secret_key()

    # 模型持久化文件路径
    MODEL_PATH = BASE_DIR / 'models' / 'text-clf-model.pkl'

    # API 输入文本最大长度（防止超长输入导致 OOM 或超时）
    try:
        MAX_TEXT_LENGTH = int(os.getenv('MAX_TEXT_LENGTH', '5000'))
    except ValueError:
        MAX_TEXT_LENGTH = 5000


class DevelopmentConfig(BaseConfig):
    """开发环境配置"""

    DEBUG = True

    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        f'sqlite:///{BASE_DIR / "app_dev.db"}'
    )


class TestingConfig(BaseConfig):
    """测试环境配置"""
    pass



class ProductionConfig(BaseConfig):
    """生产环境配置"""

    DEBUG = False

    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL")

    # 生产环境中间件配置
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 10,
        'pool_recycle': 3600,
        'max_overflow': 20
    }


# 不同环境对应的配置类
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}
