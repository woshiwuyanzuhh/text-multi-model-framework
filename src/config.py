"""
config - 训练模型相关配置类

Author: Grant Johnny
Version: 0.0.1
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

if not os.path.exists(os.path.join(BASE_DIR, 'models')):
    os.mkdir(os.path.join(BASE_DIR, 'models'))

if not os.path.exists(os.path.join(BASE_DIR, 'models_quantized')):
    os.mkdir(os.path.join(BASE_DIR, 'models_quantized'))

if not os.path.exists(os.path.join(BASE_DIR, 'models_distilled')):
    os.mkdir(os.path.join(BASE_DIR, 'models_distilled'))

if not os.path.exists(os.path.join(BASE_DIR, 'models_pruned')):
    os.mkdir(os.path.join(BASE_DIR, 'models_pruned'))


@dataclass(frozen=True)
class Config:
    """配置类"""
    pretrained_model:    str = 'hfl/chinese-macbert-base'

    stopwords_file:      Path = BASE_DIR / 'data' / 'stopwords.txt'
    class_file:          Path = BASE_DIR / 'data' / 'tmf_class.txt'

    train_raw_file:      Path = BASE_DIR / 'data/raw' / 'tmf_train.txt'
    test_raw_file:       Path = BASE_DIR / 'data/raw' / 'tmf_test.txt'
    valid_raw_file:      Path = BASE_DIR / 'data/raw/' / 'tmf_valid.txt'

    train_pre_file:      Path = BASE_DIR / 'data/pre' / 'tmf_train.txt'
    test_pre_file:       Path = BASE_DIR / 'data/pre' / 'tmf_test.txt'
    valid_pre_file:      Path = BASE_DIR / 'data/pre' / 'tmf_valid.txt'

    pkl_model_file:      Path = BASE_DIR / 'models' / 'text-clf-model.pkl'
    onnx_model_file:     Path = BASE_DIR / 'models' / 'text-clf-model.onnx'
    ftz_model_file:      Path = BASE_DIR / 'models' / 'text-clf-model.ftz'

    model_output_dir:    Path = BASE_DIR / 'models'
    quantized_model_dir: Path = BASE_DIR / 'models_quantized'
    distilled_model_dir: Path = BASE_DIR / 'models_distilled'
    pruned_model_dir:    Path = BASE_DIR / 'models_pruned'

    train_sample_size: int = int(os.getenv('TRAIN_SAMPLE_SIZE', '-1'))   # 训练采样数量，-1 表示全量
    batch_size:        int = int(os.getenv('TRAIN_BATCH_SIZE', '32'))       # 训练批大小
    learning_rate:     float = float(os.getenv('TRAIN_LEARNING_RATE', '2e-5'))  # 学习率
    epochs:            int = int(os.getenv('TRAIN_EPOCHS', '16'))           # 训练轮数
    max_len:           int = int(os.getenv('TRAIN_MAX_LEN', '32'))          # 最大序列长度
    num_workers:       int = int(os.getenv('TRAIN_NUM_WORKERS', '4'))       # DataLoader 工作进程数

    llm_api_key: str = field(default_factory=lambda: os.getenv('OPENAI_API_KEY', ''))        # 从环境变量读取，不再硬编码
    llm_model_name: str = field(default_factory=lambda: os.getenv('LLM_MODEL_NAME', 'gpt-3.5-turbo'))  # 从环境变量读取，不再硬编码
    llm_base_url: str = field(default_factory=lambda: os.getenv('LLM_BASE_URL', ''))          # 从环境变量读取，不再硬编码
    llm_max_tokens: int = 512
    llm_temperature: float = 0.1
    llm_timeout: int = int(os.getenv('LLM_TIMEOUT', '30'))                       # 从环境变量读取，默认 30 秒
