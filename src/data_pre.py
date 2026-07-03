"""
data_pre - 数据预处理

Author: Grant Johnny
Version: 0.0.1
"""
import re
import logging

from src.config import Config

logger = logging.getLogger(__name__)

# 清洗文本的正则表达式
SPACE_PATTERN = re.compile(r'[\s\u3000]+')
BRACE_PATTERN = re.compile(r'\[.*?\]|\(.*?\)|【.*?】|（.*?）|「.*?」')
ATUSR_PATTERN = re.compile(r'@\w+')
TOPIC_PATTERN = re.compile(r'#.*?#')
MYURL_PATTERN = re.compile(r'https?://[^\s\u4e00-\u9fa5]+')
CHENN_PATTERN = re.compile(r'[^a-zA-Z0-9\s\u4e00-\u9fa5:,.?!;：，。？！；]+')
NOENN_PATTERN = re.compile(r'[^\s\u4e00-\u9fa5：，。？！；]+')

# 中文分词器：优先用 jieba（与训练时保持一致），未安装则退化为按字切分
try:
    import jieba
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False


def cut_zh_words(text: str) -> str:
    """中文分词（供 v1 传统 ML 与 v2 fastText 文本预处理使用）

    优先使用 jieba 分词以匹配模型训练时的分词方式；
    若环境未安装 jieba，则退化为按字切分（空格分隔），保证接口可用但精度会下降。
    """
    if _HAS_JIEBA:
        return ' '.join(jieba.cut(text))
    # 退化方案：按字符切分，保证接口不报错
    return ' '.join(text)


# 加载停用词表
with open(Config.stopwords_file, encoding='utf-8') as file_object:
    STOP_WORDS = set(file_object.read().splitlines())


def clean_raw_text(text: str, allow_eng_num: bool=True) -> str:
    """清理原始文本内容"""
    text = re.sub(SPACE_PATTERN, ' ', text)
    text = re.sub(BRACE_PATTERN, '', text)
    text = re.sub(ATUSR_PATTERN, '', text)
    text = re.sub(TOPIC_PATTERN, '', text)
    text = re.sub(MYURL_PATTERN, '', text)

    if allow_eng_num:
        text = re.sub(CHENN_PATTERN, '', text)
    else:
        text = re.sub(NOENN_PATTERN, '', text)

    text = re.sub(SPACE_PATTERN, ' ', text)
    return text.strip()


def get_corpus(corpus_file):
    """获取指定文件中的语料"""
    corpus = []
    with open(corpus_file, encoding='utf-8') as file_obj:
        content = file_obj.read()
    content = re.sub(r'\t+', '\t', content)
    for line in content.splitlines():
        if not line.strip():
            continue
        parts = line.split('\t', maxsplit=1)
        if len(parts) != 2:
            logger.warning('跳过格式异常的行（缺少 tab 分隔符）: %s', line[:50])
            continue
        doc, label = parts
        try:
            corpus.append((doc, int(label)))
        except ValueError:
            logger.warning('跳过标签非整数的行: %s', line[:50])
            continue
    return corpus


def clean_data():
    """清洗数据：将 data/raw/ 下的原始语料经清洗和分词后写入 data/pre/

    处理流程：
    1. 读取 data/raw/ 下的 TSV 格式语料（text\\tlabel）
    2. 对文本部分调用 clean_raw_text() 清洗
    3. 调用 cut_zh_words() 进行中文分词
    4. 以 fastText 格式（__label__N word1 word2 ...）写入 data/pre/
    """
    raw_pre_pairs = [
        (Config.train_raw_file, Config.train_pre_file),
        (Config.test_raw_file, Config.test_pre_file),
        (Config.valid_raw_file, Config.valid_pre_file),
    ]
    for raw_file, pre_file in raw_pre_pairs:
        corpus = get_corpus(raw_file)
        pre_file.parent.mkdir(parents=True, exist_ok=True)
        with open(pre_file, 'w', encoding='utf-8') as f:
            for doc, label in corpus:
                cleaned = clean_raw_text(doc)
                tokenized = cut_zh_words(cleaned)
                f.write(f'__label__{label} {tokenized}\n')
        logger.info('清洗完成: %s -> %s (%d 条)', raw_file.name, pre_file.name, len(corpus))
