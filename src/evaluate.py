import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocess import prepare_datasets

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(str(MODELS_DIR), torch_dtype=torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(MODELS_DIR))
    le_data = torch.load(MODELS_DIR / "label_encoder.pt", map_location="cpu", weights_only=False)
    return model, tokenizer, le_data["classes"], device


@torch.no_grad()
def predict_batch(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch["labels"].cpu().tolist())
    return np.array(all_preds), np.array(all_labels)


def compute_metrics(y_true, y_pred, class_names):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(class_names)), zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": round(precision[i], 4),
            "recall": round(recall[i], 4),
            "f1": round(f1[i], 4),
            "support": int(support[i]),
        }

    return {
        "accuracy": round(accuracy, 4),
        "macro_avg": {"precision": round(macro_p, 4), "recall": round(macro_r, 4), "f1": round(macro_f1, 4)},
        "weighted_avg": {"precision": round(weighted_p, 4), "recall": round(weighted_r, 4), "f1": round(weighted_f1, 4)},
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "total_samples": len(y_true),
    }


def run_evaluation():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model, _, classes, device = load_model()
    logger.info("Loaded model from %s", MODELS_DIR)

    _, _, test_ds, _ = prepare_datasets()
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    y_pred, y_true = predict_batch(model, test_loader, device)
    report = compute_metrics(y_true, y_pred, classes.tolist())

    print("\n" + "=" * 56)
    print("EVALUATION REPORT")
    print("=" * 56)
    print(f"Accuracy:     {report['accuracy']:.4f}")
    print(f"Macro F1:     {report['macro_avg']['f1']:.4f}")
    print(f"Weighted F1:  {report['weighted_avg']['f1']:.4f}")
    print(f"\n{'Class':<6} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
    print("-" * 44)
    for name, m in report["per_class"].items():
        print(f"{name:<6} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>8}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_DIR / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    cls_report = classification_report(y_true, y_pred, target_names=classes.tolist(), zero_division=0)
    with open(OUTPUTS_DIR / "classification_report.txt", "w") as f:
        f.write(cls_report)

    print(f"\nReports saved to {OUTPUTS_DIR}")
    return report


if __name__ == "__main__":
    run_evaluation()
