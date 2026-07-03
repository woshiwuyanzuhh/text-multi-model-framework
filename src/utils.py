"""
utils - 工具类和工具函数

Author: Grant Johnny
Version: 0.0.1
"""
import os
import random
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from modelscope import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader

from src.config import Config
from src.data_pre import get_corpus
from src.model_train import TMFDataset

logger = logging.getLogger(__name__)

MEGA_BASE = 1024 * 1024

# 根据平台选择量化后端（Intel/AMD 用 fbgemm，Apple Silicon / ARM 用 qnnpack）
import platform
_machine = platform.machine().lower()
_is_arm = _machine in ('arm64', 'aarch64') or platform.system() == 'Darwin'
try:
    torch.backends.quantized.engine = 'qnnpack' if _is_arm else 'fbgemm'
except RuntimeError:
    # 某些 PyTorch 构建可能不支持指定后端，回退到默认引擎
    logger.warning('指定的量化后端不可用，回退到默认引擎')


def quantize_model():
    """模型量化"""
    tokenizer = AutoTokenizer.from_pretrained(Config.model_output_dir)
    model = AutoModelForSequenceClassification.from_pretrained(Config.model_output_dir)

    model.eval()

    # 调用 quantize_dynamic 实现动态量化
    # qconfig_spec 参数指定只量化 nn.Linear 层（核心计算层）
    # dtype 参数指定量化的数据类型（float32 ---> int8）
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear},
        dtype=torch.qint8
    )

    # 保存量化后的模型（量化后的模型不能使用 save_pretrained 保存）
    torch.save(
        quantized_model.state_dict(),
        os.path.join(Config.quantized_model_dir, 'pytorch_model.bin')
    )

    # 保存对应的 config 和 tokenizer 文件（推理使用）
    model.config.save_pretrained(Config.quantized_model_dir)
    tokenizer.save_pretrained(Config.quantized_model_dir)

    logger.info('模型量化完成')

    original_size = os.path.getsize(
        os.path.join(Config.model_output_dir, 'model.safetensors')
    ) / MEGA_BASE
    quantized_size = os.path.getsize(
        os.path.join(Config.quantized_model_dir, 'pytorch_model.bin')
    ) / MEGA_BASE

    logger.info('原始模型大小: %.2f MB', original_size)
    logger.info('量化模型大小: %.2f MB', quantized_size)


def evaluate_quantized_model():
    tokenizer = AutoTokenizer.from_pretrained(Config.quantized_model_dir)
    base_model = AutoModelForSequenceClassification.from_pretrained(Config.quantized_model_dir)

    quantized_model = torch.quantization.quantize_dynamic(
        base_model,
        qconfig_spec={nn.Linear},
        dtype=torch.qint8
    )

    quantized_weight_path = os.path.join(Config.quantized_model_dir, 'pytorch_model.bin')
    quantized_model.load_state_dict(torch.load(quantized_weight_path))
    quantized_model.eval()

    valid_corpus = get_corpus(Config.valid_raw_file)
    samples = random.sample(valid_corpus, k=1000)
    valid_dataset = TMFDataset(samples, tokenizer, max_len=Config.max_len)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=4)

    logger.info('开始执行量化模型批量推理...')
    all_preds, all_trues = [], []
    with torch.inference_mode():
        for batch in valid_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label'].numpy()

            outputs = quantized_model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).numpy()

            all_preds.extend(preds)
            all_trues.extend(labels)

    y_true = np.array(all_trues)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    logger.info('量化后整体准确率 (Accuracy): %.2f%%', acc * 100)
    logger.info('分类报告:\n%s', classification_report(y_true, y_pred, digits=4))


def distill_model():
    """知识蒸馏"""
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )

    # 教师模型使用你之前在本地训练好的完美微调模型目录
    teacher_dir = Config.model_output_dir
    student_name = 'hfl/rbt3'

    tokenizer = AutoTokenizer.from_pretrained(teacher_dir)
    teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_dir)
    student_model = AutoModelForSequenceClassification.from_pretrained(student_name, num_labels=10)

    teacher_model.to(device)
    student_model.to(device)

    teacher_model.eval()
    student_model.train()

    train_corpus = get_corpus(Config.train_raw_file)
    samples = random.sample(train_corpus, k=900)
    train_dataset = TMFDataset(samples, tokenizer, max_len=Config.max_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        persistent_workers=Config.num_workers > 0
    )

    optimizer = optim.AdamW(student_model.parameters(), lr=5e-5)

    EPOCHS = 16
    temperature = 4.0  # 软化概率分布的温度
    alpha = 0.5        # 教师监督与真实标签的平衡权重

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device, dtype=torch.long)

            # 教师模型推理
            with torch.inference_mode():
                teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits

            # 学生模型前向传播
            student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = student_outputs.logits

            # 计算标准交叉熵损失 (硬标签损失)
            loss_hard = F.cross_entropy(student_logits, labels)

            # 计算蒸馏损失 (KL 散度 - 软标签损失)
            # 使用 log_softmax 和 softmax 分别处理学生和老师的输出
            loss_soft = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction='mean'
            ) * (temperature ** 2)

            # 综合损失
            loss = (1.0 - alpha) * loss_hard +  alpha * loss_soft

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logger.info('Epoch[%d/%d], Distill Loss: %.4f', epoch, EPOCHS, total_loss / len(train_loader))

    student_model.save_pretrained(Config.distilled_model_dir)
    tokenizer.save_pretrained(Config.distilled_model_dir)
    logger.info('小模型蒸馏已完成')


def evaluate_distilled_model():
    tokenizer = AutoTokenizer.from_pretrained(Config.distilled_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(Config.distilled_model_dir)

    model.eval()

    valid_corpus = get_corpus(Config.valid_raw_file)
    samples = random.sample(valid_corpus, k=1000)
    valid_dataset = TMFDataset(samples, tokenizer, max_len=Config.max_len)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=4)

    logger.info('开始执行蒸馏模型批量推理...')
    all_preds, all_trues = [], []
    with torch.inference_mode():
        for batch in valid_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label'].numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).numpy()

            all_preds.extend(preds)
            all_trues.extend(labels)

    y_true = np.array(all_trues)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    logger.info('蒸馏后整体准确率 (Accuracy): %.2f%%', acc * 100)
    logger.info('分类报告:\n%s', classification_report(y_true, y_pred, digits=4))


def prune_bert_layers():
    """结构化剪枝（切除 BERT 模型后半部 Transformer 层）"""
    tokenizer = AutoTokenizer.from_pretrained(Config.model_output_dir)
    model = AutoModelForSequenceClassification.from_pretrained(Config.model_output_dir)

    logger.info('原始模型层数: %d 层', len(model.bert.encoder.layer))
    model.bert.encoder.layer = torch.nn.ModuleList([
        model.bert.encoder.layer[i] for i in range(6)
    ])
    model.config.num_hidden_layers = 6
    model.config.num_labels = 10
    logger.info('裁剪后模型层数: %d 层', len(model.bert.encoder.layer))

    model.save_pretrained(Config.pruned_model_dir)
    tokenizer.save_pretrained(Config.pruned_model_dir)
    logger.info('模型剪枝完成')


def evaluate_pruned_model():
    tokenizer = AutoTokenizer.from_pretrained(Config.pruned_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(Config.pruned_model_dir)

    model.eval()

    valid_corpus = get_corpus(Config.valid_raw_file)
    samples = random.sample(valid_corpus, k=1000)
    valid_dataset = TMFDataset(samples, tokenizer, max_len=Config.max_len)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=4)

    logger.info('开始执行剪枝模型批量推理...')
    all_preds, all_trues = [], []
    with torch.inference_mode():
        for batch in valid_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label'].numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).numpy()

            all_preds.extend(preds)
            all_trues.extend(labels)

    y_true = np.array(all_trues)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    logger.info('剪枝后整体准确率 (Accuracy): %.2f%%', acc * 100)
    logger.info('分类报告:\n%s', classification_report(y_true, y_pred, digits=4))
