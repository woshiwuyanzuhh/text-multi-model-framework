"""
data_eda - 探索性数据分析

pip list --format=freeze > requirements.txt
pip install -r requirements.txt

Author: Grant Johnny
Version: 0.0.1
"""
import pandas as pd
import logging

from src.config import Config

logger = logging.getLogger(__name__)


def run_eda():
    """执行探索性数据分析：标签分布与文本长度统计"""
    df = pd.read_csv(Config.train_raw_file, sep='\t', names=['text', 'label'])
    logger.info('DataFrame info:')
    df.info()

    logger.info('标签分布:\n%s', df.label.value_counts(normalize=True))

    df['length'] = df.text.map(len)
    logger.info('Min text length: %d', df.length.min())
    logger.info('Max text length: %d', df.length.max())
    logger.info('Mean text length: %.2f', df.length.mean())
    logger.info('Std text length: %.4f', df.length.std())


if __name__ == '__main__':
    run_eda()
