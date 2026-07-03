"""
extensions - 扩展功能

Author: Grant Johnny
Version: 0.0.1
"""
import logging

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from src.config import Config

logger = logging.getLogger('app.extensions')

# API 限流器（按客户端 IP 限流，限流策略在 create_app 中从环境变量读取）
limiter = Limiter(key_func=get_remote_address)


class TextClassifierExtension:
    """文本分类器扩展"""

    def __init__(self):
        self.macbert_model = None
        self.tokenizer = None
        # v1 传统 ML 模型（pkl）与 v2 fastText 模型（ftz）——启动时按需加载，失败不阻塞 app
        self.text_clf_model = None
        self.ftz_clf_model = None
        with open(Config.class_file, encoding='utf-8') as f:
            self.class_labels = f.read().strip().splitlines()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else
            'cpu'
        )

    def init_app(self, model_path: str):
        """加载所有模型。单个模型加载失败不阻塞整体启动，对应接口运行时返回友好错误。"""
        logger.info('正在加载文本分类模型')

        # 1. MacBERT (v3)
        if not self.macbert_model:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(Config.model_output_dir)
                self.macbert_model = AutoModelForSequenceClassification.from_pretrained(Config.model_output_dir)
                self.macbert_model.to(self.device)
                logger.info('MacBERT 模型加载完成')
            except Exception as e:
                logger.error('MacBERT 模型加载失败: %s', e, exc_info=True)

        # 2. 传统 ML pkl (v1)
        if not self.text_clf_model:
            try:
                import joblib
                self.text_clf_model = joblib.load(Config.pkl_model_file)
                logger.info('传统 ML (pkl) 模型加载完成')
            except Exception as e:
                logger.error('pkl 模型加载失败: %s', e, exc_info=True)
                self.text_clf_model = None

        # 3. fastText ftz (v2) —— fasttext 未安装时会友好降级
        if not self.ftz_clf_model:
            try:
                import fasttext
                self.ftz_clf_model = fasttext.load_model(str(Config.ftz_model_file))
                logger.info('fastText (ftz) 模型加载完成')
            except Exception as e:
                logger.error('ftz 模型加载失败: %s', e, exc_info=True)
                self.ftz_clf_model = None

        logger.info('文本分类模型加载完成')


thy_extension = TextClassifierExtension()
