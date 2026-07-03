"""
utils - API 公共工具函数

Author: Grant Johnny
Version: 0.0.1
"""
from functools import wraps

from flask import request, current_app


def validate_predict_request():
    """校验预测请求的 JSON 格式和 text 字段。

    Returns:
        (json_string, error_response) — 校验通过时 error_response 为 None；
        校验失败时 json_string 为 None，error_response 为可直接返回的 dict。
    """
    json_string = request.get_json(silent=True, force=True)
    if not isinstance(json_string, dict):
        return None, {'code': -20, 'message': '请求体不符合 JSON 格式规范'}

    text = json_string.get('text', '')
    if not isinstance(text, str):
        return None, {'code': -10, 'message': 'text 字段必须为字符串类型'}
    if not text:
        return None, {'code': -10, 'message': '请提供要分类的文本内容'}

    max_len = current_app.config.get('MAX_TEXT_LENGTH', 5000)
    if len(text) > max_len:
        return None, {'code': -11, 'message': f'输入文本过长（{len(text)} 字），最大允许 {max_len} 字'}

    return json_string, None


def require_json_input(func):
    """装饰器：自动校验 JSON 请求体，校验通过后将 text 注入为关键字参数。

    被装饰函数签名需包含 text 参数。
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        json_string, error = validate_predict_request()
        if error:
            return error
        kwargs['text'] = json_string.get('text', '')
        return func(*args, **kwargs)
    return wrapper
