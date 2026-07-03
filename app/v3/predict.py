"""
predict - API 接口实现

Author: Grant Johnny
Version: 0.0.1
"""
import logging
import os

import torch
from flask import Blueprint

from app.extensions import thy_extension, limiter
from app.utils import validate_predict_request
from src.config import Config

logger = logging.getLogger('app.v3')

# 创建蓝图对象
model_bp_v3 = Blueprint('models_v3', __name__, url_prefix='/api/v3')


@model_bp_v3.route('/predict', methods=['GET', 'POST'])
@limiter.limit(os.getenv('RATE_LIMIT_LOCAL', '60 per minute'))
def text_clf_predict():
    """文本分类预测"""
    json_string, error = validate_predict_request()
    if error:
        return error

    text = json_string.get('text', '')

    if thy_extension.macbert_model is None or thy_extension.tokenizer is None:
        return {'code': -50, 'message': 'MacBERT 模型未加载，请联系管理员'}

    try:
        inputs = thy_extension.tokenizer(
            [text],
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=Config.max_len
        )
        input_ids = inputs.input_ids.to(thy_extension.device)
        attention_mask = inputs.attention_mask.to(thy_extension.device)
        with torch.inference_mode():
            output = thy_extension.macbert_model(input_ids=input_ids, attention_mask=attention_mask)
        y_pred = torch.argmax(output.logits, dim=-1).cpu().numpy()
        label = thy_extension.class_labels[y_pred[0]]
    except Exception as e:
        logger.error('V3 推理异常: %s', e, exc_info=True)
        return {'code': -51, 'message': '分类服务内部错误'}
    return {'code': 0, 'message': 'OK', 'label': label}
