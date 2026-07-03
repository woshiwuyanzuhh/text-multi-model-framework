"""
run_pipeline - 离线训练流水线

Author: Grant Johnny
Version: 0.0.1
"""
import logging

from src.data_pre import clean_data
from src.model_eval import evaluate_model
from src.model_train import train_model
from src.utils import (
    quantize_model, evaluate_quantized_model,
    distill_model, evaluate_distilled_model,
    prune_bert_layers, evaluate_pruned_model,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    logger.info('=== [Step 1/6] 开始执行数据清洗和准备工作 ===')
    clean_data()

    logger.info('=== [Step 2/6] 开始训练和导出文本分类模型 ===')
    train_model()

    logger.info('=== [Step 3/6] 启动服务之前对模型进行评估 ===')
    evaluate_model()

    logger.info('=== [Step 4/6] 对已有模型进行量化、蒸馏或剪枝 ===')
    quantize_model()
    distill_model()
    prune_bert_layers()

    logger.info('=== [Step 5/6] 对量化、蒸馏或剪枝的模型进行评估 ===')
    evaluate_quantized_model()
    evaluate_distilled_model()
    evaluate_pruned_model()

    logger.info('=== [Step 6/6] 流水线执行完毕 ===')


if __name__ == "__main__":
    main()
