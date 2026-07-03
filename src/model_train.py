"""
model_train - 训练模型

Author: Grant Johnny
Version: 0.0.1
"""
import random
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from modelscope import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader

from src.config import Config
from src.data_pre import get_corpus

logger = logging.getLogger(__name__)


class TMFDataset(Dataset):

    def __init__(self, corpus, tokenizer, max_len=32):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        doc, label = self.corpus[index]
        inputs = self.tokenizer(
            doc,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=self.max_len
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


def train_model():
    """模型训练"""
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )

    tokenizer = AutoTokenizer.from_pretrained(Config.pretrained_model)

    train_corpus = get_corpus(Config.train_raw_file)
    if Config.train_sample_size > 0:
        samples = random.sample(train_corpus, k=Config.train_sample_size)
    else:
        samples = train_corpus

    train_dataset = TMFDataset(samples, tokenizer, max_len=Config.max_len)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=Config.num_workers,
        persistent_workers=Config.num_workers > 0,
    )

    model = AutoModelForSequenceClassification.from_pretrained(Config.pretrained_model, num_labels=10)
    model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate)

    for epoch in range(1, Config.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_func(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logger.info('Epoch[%d/%d], Loss: %.4f', epoch, Config.epochs, total_loss / len(train_loader))

    model.save_pretrained(Config.model_output_dir)
    tokenizer.save_pretrained(Config.model_output_dir)
