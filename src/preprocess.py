import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

RANDOM_SEED = 42
MAX_SEQ_LENGTH = 512
MODEL_NAME = "answerdotai/ModernBERT-base"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "dataset.xlsx"


class LogDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.attention_masks = torch.tensor(attention_masks, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }


def load_raw_data(data_path=None):
    path = Path(data_path) if data_path else DATA_PATH
    logger.info("Loading dataset from %s", path)
    return pd.read_excel(path, sheet_name="log_dataset")


def load_label_metadata(data_path=None):
    path = Path(data_path) if data_path else DATA_PATH
    return pd.read_excel(path, sheet_name="root_cause_labels")


def encode_labels(labels):
    le = LabelEncoder()
    encoded = le.fit_transform(labels)
    return encoded, le


def split_data(messages, labels, val_size=0.15, test_size=0.15):
    X_temp, X_test, y_temp, y_test = train_test_split(
        messages, labels, test_size=test_size, stratify=labels, random_state=RANDOM_SEED
    )
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, stratify=y_temp, random_state=RANDOM_SEED
    )
    logger.info("Split sizes — train: %d, val: %d, test: %d", len(X_train), len(X_val), len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test


def tokenize_messages(messages, tokenizer, max_length=MAX_SEQ_LENGTH):
    encoded = tokenizer(
        list(messages), padding="max_length", truncation=True,
        max_length=max_length, return_tensors="np",
    )
    return encoded["input_ids"], encoded["attention_mask"]


def prepare_datasets(data_path=None, val_size=0.15, test_size=0.15):
    df = load_raw_data(data_path)
    messages = df["log_message"].to_numpy()
    y, label_encoder = encode_labels(df["root_cause_label"].to_numpy())

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        messages, y, val_size=val_size, test_size=test_size
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ids, train_masks = tokenize_messages(X_train, tokenizer)
    val_ids, val_masks = tokenize_messages(X_val, tokenizer)
    test_ids, test_masks = tokenize_messages(X_test, tokenizer)

    train_ds = LogDataset(train_ids, train_masks, y_train)
    val_ds = LogDataset(val_ids, val_masks, y_val)
    test_ds = LogDataset(test_ids, test_masks, y_test)

    return train_ds, val_ds, test_ds, label_encoder


def prepare_single_message(message, tokenizer=None):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoded = tokenizer(
        message, padding="max_length", truncation=True,
        max_length=MAX_SEQ_LENGTH, return_tensors="pt",
    )
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}
