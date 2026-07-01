import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocess import prepare_datasets, MODEL_NAME

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_CONFIG = {
    "batch_size": 8,
    "epochs": 10,
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "early_stopping_patience": 3,
    "lr_decay_factor": 0.5,
    "lr_decay_patience": 2,
    "val_size": 0.15,
    "test_size": 0.15,
}


def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=-1)
    return (preds == labels).float().mean().item()


def train_epoch(model, dataloader, optimizer, scheduler, device, max_grad_norm):
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        total_loss += outputs.loss.item()
        total_acc += compute_accuracy(outputs.logits, batch["labels"])

    n = len(dataloader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def validate_epoch(model, dataloader, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += outputs.loss.item()
        total_acc += compute_accuracy(outputs.logits, batch["labels"])

    n = len(dataloader)
    return total_loss / n, total_acc / n


def save_checkpoint(model, tokenizer, label_encoder, config, history, save_path):
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    torch.save({"classes": label_encoder.classes_}, save_path / "label_encoder.pt")

    metadata = {
        "model_name": MODEL_NAME,
        "num_classes": len(label_encoder.classes_),
        "class_labels": label_encoder.classes_.tolist(),
        "training_config": config,
        "training_history": history,
        "best_val_loss": min(h["val_loss"] for h in history),
        "best_val_accuracy": max(h["val_accuracy"] for h in history),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(save_path / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def train(config=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)

    train_ds, val_ds, test_ds, label_encoder = prepare_datasets(
        val_size=cfg["val_size"], test_size=cfg["test_size"]
    )
    num_classes = len(label_encoder.classes_)
    logger.info("Loaded %d classes, %d train / %d val / %d test samples",
                num_classes, len(train_ds), len(val_ds), len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_classes
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    total_steps = len(train_loader) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    lr_decay = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg["lr_decay_factor"],
        patience=cfg["lr_decay_patience"]
    )

    history = []
    best_val_loss = float("inf")
    patience = 0

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, cfg["max_grad_norm"])
        val_loss, val_acc = validate_epoch(model, val_loader, device)
        lr_decay.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        })

        logger.info("Epoch %2d | train loss: %.4f  acc: %.4f | val loss: %.4f  acc: %.4f",
                     epoch, train_loss, train_acc, val_loss, val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            save_checkpoint(model, tokenizer, label_encoder, cfg, history, MODELS_DIR)
        else:
            patience += 1
            if patience >= cfg["early_stopping_patience"]:
                logger.info("Early stopping at epoch %d", epoch)
                break

    test_loss, test_acc = validate_epoch(model, test_loader, device)
    logger.info("Test — loss: %.4f  accuracy: %.4f", test_loss, test_acc)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_DIR / "training_history.json", "w") as f:
        json.dump({"history": history, "test": {"loss": test_loss, "accuracy": test_acc}}, f, indent=2)

    return MODELS_DIR


def main():
    parser = argparse.ArgumentParser(description="Train log classifier")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    args = parser.parse_args()

    config = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
    }
    train(config)


if __name__ == "__main__":
    main()
