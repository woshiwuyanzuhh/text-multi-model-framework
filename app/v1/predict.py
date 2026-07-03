"""
predict - API 接口实现

Author: Grant Johnny
Version: 0.0.1
"""
import logging
import os

from flask import Blueprint

from app.extensions import thy_extension, limiter
from app.utils import validate_predict_request
from src.data_pre import cut_zh_words

logger = logging.getLogger('app.v1')

# 创建蓝图对象
model_bp_v1 = Blueprint('models_v1', __name__, url_prefix='/api/v1')


@model_bp_v1.route('/predict', methods=['GET', 'POST'])
@limiter.limit(os.getenv('RATE_LIMIT_LOCAL', '60 per minute'))
def text_clf_predict():
    """文本分类预测"""
    json_string, error = validate_predict_request()
    if error:
        return error

    if thy_extension.text_clf_model is None:
        return {'code': -50, 'message': '传统 ML 模型未加载，请联系管理员'}

    text = cut_zh_words(json_string.get('text', ''))
    try:
        y_pred = thy_extension.text_clf_model.predict([text])
        label = thy_extension.class_labels[y_pred[0]]
    except Exception as e:
        logger.error('V1 推理异常: %s', e, exc_info=True)
        return {'code': -51, 'message': '分类服务内部错误'}
    return {'code': 0, 'message': 'OK', 'label': label}
