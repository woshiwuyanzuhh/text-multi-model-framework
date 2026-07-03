"""
predict - API 接口实现（V4 大模型版本）

通过调用大语言模型（OpenAI 兼容接口）完成中文文本分类。
沿用 v1/v2/v3 的 Blueprint 风格与统一 JSON 返回结构。

Author: Grant Johnny
Version: 0.0.1
"""
import logging
import os
import threading
import time

from flask import Blueprint
import openai
from openai import OpenAI

from app.extensions import limiter
from app.utils import validate_predict_request
from src.config import Config

# 创建日志器
logger = logging.getLogger('app.v4')

# 创建蓝图对象
model_bp_v4 = Blueprint('models_v4', __name__, url_prefix='/api/v4')


# ===== 模块级初始化（进程启动时执行一次） =====

# 读取配置
_config = Config()

# 惰性初始化 OpenAI client（线程安全，首次调用时创建并缓存）
_client_cache = None
_client_lock = threading.Lock()


def _get_client():
    """获取 OpenAI client 实例，api_key 为空时返回 None"""
    global _client_cache
    if _client_cache is not None:
        return _client_cache
    with _client_lock:
        if _client_cache is not None:
            return _client_cache
        api_key = _config.llm_api_key
        if not api_key:
            return None
        if _config.llm_base_url:
            _client_cache = OpenAI(api_key=api_key, base_url=_config.llm_base_url)
        else:
            _client_cache = OpenAI(api_key=api_key)
        return _client_cache

# 从 data/tmf_class.txt 读取分类标签列表（按行读取，strip 后非空则 split 取第一个元素）
_class_labels = []
with open(_config.class_file, encoding='utf-8') as _f:
    for _line in _f:
        _line = _line.strip()
        if _line:
            _class_labels.append(_line.split()[0])

# 预计算 system_prompt（仅依赖固定的 _class_labels，无需每次请求重复构造）
_labels_str = '、'.join(_class_labels)
_system_prompt = (
    f'你是一个专业的中文文本分类专家。请将用户提供的文本分类到以下类别之一：'
    f'{_labels_str}。你只需要回复类别名称，绝对不要回复任何其他内容、解释、标点或格式。'
)


@model_bp_v4.route('/predict', methods=['GET', 'POST'])
@limiter.limit(os.getenv('RATE_LIMIT_LLM', '10 per minute'))
def text_clf_predict():
    """文本分类预测（大模型版本）"""
    start_time = time.time()

    json_string, error = validate_predict_request()
    if error:
        logger.warning('请求校验失败: %s', error.get('message'))
        return error

    text = json_string.get('text', '')

    # 检查 OpenAI client 可用性
    if _get_client() is None:
        logger.error('OPENAI_API_KEY 未设置，V4 接口不可用')
        return {'code': -43, 'message': '大模型 API 认证失败，请检查 API Key'}

    # 截断超长文本用于日志展示
    logger.info('收到分类请求，输入文本: %s', text[:50])

    # 使用预计算的 system_prompt

    try:
        # openai SDK 的 timeout 参数在 DNS 解析阶段可能不生效（已知问题），
        # 用线程级硬超时兜底，确保不会无限期阻塞请求线程。
        result_container = {}

        def _call_api():
            try:
                result_container['response'] = _get_client().chat.completions.create(
                    model=_config.llm_model_name,
                    messages=[
                        {'role': 'system', 'content': _system_prompt},
                        {'role': 'user', 'content': text},
                    ],
                    max_tokens=_config.llm_max_tokens,
                    temperature=_config.llm_temperature,
                    timeout=_config.llm_timeout,
                )
            except Exception as e:
                result_container['error'] = e

        api_thread = threading.Thread(target=_call_api, daemon=True)
        api_thread.start()
        api_thread.join(timeout=_config.llm_timeout + 5)  # 给 SDK 内部超时留 5s 缓冲

        if api_thread.is_alive():
            # 线程仍在跑说明 SDK 超时失效，强制返回超时错误
            elapsed = (time.time() - start_time) * 1000
            logger.error('大模型请求硬超时 (耗时 %.2fms)', elapsed)
            return {'code': -40, 'message': '大模型服务请求超时'}

        if 'error' in result_container:
            raise result_container['error']

        response = result_container['response']
        raw_label = (response.choices[0].message.content or '').strip()

        # 第一层：精确匹配
        matched_label = None
        for label in _class_labels:
            if raw_label == label:
                matched_label = label
                break

        # 第二层：模糊匹配（返回文本包含标签，或标签包含在返回文本中）
        if matched_label is None:
            for label in _class_labels:
                if label in raw_label or raw_label in label:
                    matched_label = label
                    break

        elapsed = (time.time() - start_time) * 1000

        if matched_label is None:
            logger.error('模型返回结果无法解析为有效分类，原始返回: %s (耗时 %.2fms)', raw_label, elapsed)
            return {'code': -30, 'message': '模型返回结果无法解析为有效分类'}

        logger.info('分类成功，标签: %s (耗时 %.2fms)', matched_label, elapsed)
        return {'code': 0, 'message': 'OK', 'label': matched_label}

    except Exception as e:
        elapsed = (time.time() - start_time) * 1000

        if isinstance(e, openai.APITimeoutError):
            logger.error('大模型请求超时 (耗时 %.2fms)', elapsed, exc_info=True)
            return {'code': -40, 'message': '大模型服务请求超时'}
        if isinstance(e, openai.APIConnectionError):
            logger.error('大模型连接失败 (耗时 %.2fms)', elapsed, exc_info=True)
            return {'code': -41, 'message': '大模型服务连接失败'}
        if isinstance(e, openai.RateLimitError):
            logger.error('大模型 API 频率超限 (耗时 %.2fms)', elapsed, exc_info=True)
            return {'code': -42, 'message': '大模型 API 调用频率超限'}
        if isinstance(e, openai.AuthenticationError):
            logger.error('大模型 API 认证失败 (耗时 %.2fms)', elapsed, exc_info=True)
            return {'code': -43, 'message': '大模型 API 认证失败，请检查 API Key'}
        if isinstance(e, openai.APIStatusError):
            logger.error('大模型 API 状态错误 status=%s (耗时 %.2fms)', e.status_code, elapsed, exc_info=True)
            return {'code': -44, 'message': f'大模型 API 错误: {e.status_code}'}

        logger.error('分类服务内部错误 (耗时 %.2fms)', elapsed, exc_info=True)
        return {'code': -51, 'message': '分类服务内部错误'}
