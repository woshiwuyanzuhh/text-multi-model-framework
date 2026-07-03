"""
model_eval - 模型评估

Author: Grant Johnny
Version: 0.0.1
"""
import random
import time
import logging

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import Config
from src.data_pre import get_corpus

logger = logging.getLogger(__name__)


def evaluate_model():
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )

    tokenizer = AutoTokenizer.from_pretrained(Config.model_output_dir)
    model = AutoModelForSequenceClassification.from_pretrained(Config.model_output_dir)
    model.to(device)

    model.eval()

    valid_corpus = get_corpus(Config.valid_raw_file)
    total_accuracy, total_duration = 0.0, 0.0
    for _ in range(100):
        samples = random.sample(valid_corpus, k=100)
        texts, labels = zip(*samples)
        y_test = np.array(labels)

        inputs = tokenizer(
            texts,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=Config.max_len
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        start = time.perf_counter()
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        end = time.perf_counter()

        y_pred = torch.argmax(output.logits, dim=-1).cpu().numpy()
        total_accuracy += accuracy_score(y_test, y_pred)
        total_duration += end - start

    logger.info('Accuracy: %.2f%%', total_accuracy / 100 * 100)
    logger.info('Duration: %.3f', total_duration / 100)
