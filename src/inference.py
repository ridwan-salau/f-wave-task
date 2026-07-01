import argparse
import json
import logging
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocess import prepare_single_message, load_label_metadata

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(str(MODELS_DIR), torch_dtype=torch.float32).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(MODELS_DIR))
    le_data = torch.load(MODELS_DIR / "label_encoder.pt", map_location="cpu", weights_only=False)
    classes = le_data["classes"]

    label_meta = load_label_metadata()
    label_map = {}
    for _, row in label_meta.iterrows():
        label_map[row["id"]] = {
            "label": row["label"],
            "description": row["description"],
            "severity": row.get("severity", "Unknown"),
            "typical_resolution": row.get("typical_resolution", "See runbook"),
        }

    return model, tokenizer, classes, label_map, device


def extract_key_details(log_message):
    details = {}

    m = re.search(r"\b(CRITICAL|ERROR|WARN|WARNING|INFO|DEBUG)\b", log_message, re.IGNORECASE)
    if m:
        details["log_level"] = m.group(1).upper()

    m = re.search(r"\[([a-zA-Z0-9_-]+)\]", log_message)
    if m:
        details["affected_component"] = m.group(1)

    m = re.search(r"\b([2-5]\d{2})\b", log_message)
    if m:
        details["status_code"] = m.group(1)

    for pattern, key in [
        (r"client[_-]?(\w+)", "client_id"),
        (r"user[_-]?(\w+)", "user_id"),
        (r"org[_-]?(\w+)", "org_id"),
        (r"txn[_-]?(\w+)", "transaction_id"),
        (r"rec[_-]?(\w+)", "record_id"),
        (r"IP:\s*([\d.]+)", "ip_address"),
    ]:
        m = re.search(pattern, log_message, re.IGNORECASE)
        if m:
            details[key] = m.group(1)

    m = re.search(r"(\d+)\s*ms", log_message)
    if m:
        details["latency_ms"] = m.group(1)

    return details


def generate_summary(log_message, predicted_label, confidence, label_map):
    meta = label_map.get(predicted_label, {})
    details = extract_key_details(log_message)

    sev = details.get("log_level", meta.get("severity", "Unknown"))
    if sev.upper() in ("CRITICAL", "ERROR"):
        sev = "Critical"
    elif sev.upper() in ("WARN", "WARNING"):
        sev = "High"

    return {
        "predicted_root_cause": predicted_label,
        "root_cause_name": meta.get("label", "Unknown"),
        "confidence": round(confidence, 4),
        "severity": sev,
        "description": meta.get("description", ""),
        "affected_component": details.get("affected_component", "unknown"),
        "extracted_details": details,
        "suggested_action": meta.get("typical_resolution", "Investigate manually"),
    }


@torch.no_grad()
def predict_single(log_message, model=None, tokenizer=None, classes=None,
                   label_map=None, device=None, return_summary=True):
    if model is None:
        model, tokenizer, classes, label_map, device = load_model()
    if device is None:
        device = next(model.parameters()).device

    inputs = prepare_single_message(log_message, tokenizer)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).squeeze().detach().cpu().numpy()
    pred_idx = int(torch.argmax(outputs.logits, dim=-1).item())

    result = {
        "log_message": log_message,
        "predicted_label": classes[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {cls: round(float(p), 4) for cls, p in zip(classes, probs)},
    }

    if return_summary and label_map:
        result["summary"] = generate_summary(
            log_message, result["predicted_label"], result["confidence"], label_map
        )

    return result


def predict_batch(log_messages, model=None, tokenizer=None, classes=None,
                  label_map=None, device=None):
    if model is None:
        model, tokenizer, classes, label_map, device = load_model()

    return [predict_single(msg, model, tokenizer, classes, label_map, device) for msg in log_messages]


def main():
    parser = argparse.ArgumentParser(description="Run inference on log messages")
    parser.add_argument("--message", "-m", type=str, help="Single log message to classify")
    parser.add_argument("--file", "-f", type=str, help="Path to a file with log messages (one per line)")
    parser.add_argument("--test-samples", action="store_true", help="Run on the held-out test set")
    parser.add_argument("--output", "-o", type=str, default=str(OUTPUTS_DIR / "inference_results.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model, tokenizer, classes, label_map, device = load_model()
    logger.info("Model loaded, %d classes", len(classes))

    if args.message:
        result = predict_single(args.message, model, tokenizer, classes, label_map, device)
        print(f"\nPrediction: {result['predicted_label']} ({result['summary']['root_cause_name']})")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Summary: {json.dumps(result['summary'], indent=2)}")
        print(f"Probabilities: {json.dumps(result['probabilities'], indent=2)}")
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)

    elif args.file:
        file_path = Path(args.file)
        with open(file_path) as f:
            if file_path.suffix == ".jsonl":
                messages = [json.loads(line).get("log_message", "") for line in f]
            else:
                messages = [line.strip() for line in f if line.strip()]

        results = predict_batch(messages, model, tokenizer, classes, label_map, device)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} results to {args.output}")

    elif args.test_samples:
        from src.preprocess import prepare_datasets
        _, _, test_ds, _ = prepare_datasets()

        results = []
        for i in range(len(test_ds)):
            sample = test_ds[i]
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze().detach().cpu().numpy()
            pred_idx = int(torch.argmax(outputs.logits, dim=-1).item())
            true_idx = int(sample["labels"].item())
            results.append({
                "true_label": classes[true_idx],
                "predicted_label": classes[pred_idx],
                "confidence": round(float(probs[pred_idx]), 4),
                "correct": bool(pred_idx == true_idx),
            })

        correct = sum(1 for r in results if r["correct"])
        print(f"Test accuracy: {correct}/{len(results)} = {correct/len(results):.4f}")

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
