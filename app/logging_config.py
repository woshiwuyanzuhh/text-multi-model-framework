"""
logging_config - 统一日志配置

配置 Python logging，同时输出到 stderr 和 logs/api_service.log。
使用 RotatingFileHandler 防止日志文件无限增长。

Author: Grant Johnny
Version: 0.0.1
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = '[%(asctime)s] %(levelname)s %(name)s: %(message)s'

# 日志文件滚动配置：单文件最大 10MB，保留最近 5 个备份
LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 5


def setup_logging():
    """统一配置 Python logging：输出到 stderr 和 logs/api_service.log"""
    log_dir = Path(__file__).resolve().parent.parent / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'api_service.log'

    formatter = logging.Formatter(LOG_FORMAT)

    # stderr handler
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)

    # file handler（滚动写入，防止日志文件无限增长）
    file_handler = RotatingFileHandler(
        log_file, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(stderr_handler)
    root_logger.addHandler(file_handler)
