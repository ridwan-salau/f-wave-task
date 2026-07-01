import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.preprocess import prepare_datasets, prepare_single_message, load_label_metadata, MODEL_NAME
from src.train import train_epoch, validate_epoch

MODELS_DIR = Path(__file__).resolve().parent / "models"

st.set_page_config(page_title="Log Root Cause Classifier", layout="wide")
st.title("Log Root Cause Classifier")
st.caption("Fine-tunes ModernBERT to classify system error logs into 8 root cause categories.")


def load_trained_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODELS_DIR), torch_dtype=torch.float32
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(MODELS_DIR))
    le_data = torch.load(MODELS_DIR / "label_encoder.pt", map_location="cpu", weights_only=False)
    return model, tokenizer, le_data["classes"], device


def load_label_map():
    label_meta = load_label_metadata()
    label_map = {}
    for _, row in label_meta.iterrows():
        label_map[row["id"]] = {
            "label": row["label"],
            "description": row["description"],
            "severity": row.get("severity", "Unknown"),
            "typical_resolution": row.get("typical_resolution", "See runbook"),
        }
    return label_map


# Sidebar
st.sidebar.header("Training Settings")
batch_size = st.sidebar.slider("Batch size", 4, 32, 8, 4)
epochs = st.sidebar.slider("Epochs", 3, 20, 8, 1)
learning_rate = st.sidebar.number_input(
    "Learning rate", min_value=1e-6, max_value=1e-3, value=9e-5,
    format="%.1e", step=1e-6
)
model_exists = (MODELS_DIR / "model.safetensors").exists() or st.session_state.get("model_trained", False)
st.sidebar.markdown("---")
st.sidebar.write(f"Model trained: **{'Yes' if model_exists else 'No'}**")

# Tabs
tab1, tab2, tab3 = st.tabs(["Train", "Evaluate", "Inference"])

# Train
with tab1:
    st.header("Train Model")

    col1, col2 = st.columns([2, 1])
    with col1:
        train_btn = st.button("Start Training", type="primary", use_container_width=True)
    with col2:
        if model_exists:
            st.success("Model already trained ✓")

    if train_btn:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        st.info(f"Using device: {device}")

        with st.spinner("Loading data..."):
            train_ds, val_ds, test_ds, label_encoder = prepare_datasets()
            num_classes = len(label_encoder.classes_)
            st.write(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} | Classes: {num_classes}")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=num_classes
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * 0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        lr_decay = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

        progress_bar = st.progress(0, text="Training...")
        chart_placeholder = st.empty()
        status = st.empty()

        history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, scheduler, device, 1.0
            )
            val_loss, val_acc = validate_epoch(model, val_loader, device)
            lr_decay.step(val_loss)

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            df = pd.DataFrame(history).set_index("epoch")

            col_a, col_b = chart_placeholder.columns(2)
            with col_a:
                st.subheader("Loss")
                st.line_chart(df[["train_loss", "val_loss"]], height=250)
            with col_b:
                st.subheader("Accuracy")
                st.line_chart(df[["train_acc", "val_acc"]], height=250)

            progress_bar.progress(epoch / epochs, text=f"Epoch {epoch}/{epochs}")
            status.text(
                f"Epoch {epoch:2d} | train loss: {train_loss:.4f}  acc: {train_acc:.4f}"
                f" | val loss: {val_loss:.4f}  acc: {val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(MODELS_DIR))
                tokenizer.save_pretrained(str(MODELS_DIR))
                torch.save({"classes": label_encoder.classes_}, MODELS_DIR / "label_encoder.pt")

        test_loss, test_acc = validate_epoch(model, test_loader, device)
        progress_bar.progress(1.0, text="Complete ✓")
        st.success(f"Training complete. Test accuracy: {test_acc:.4f}")
        st.session_state.model_trained = True

# Evaluate
with tab2:
    st.header("Evaluation")

    if not model_exists:
        st.warning("No trained model found. Train the model first.")
    else:
        model, tokenizer, classes, device = load_trained_model()
        label_map = load_label_map()

        eval_btn = st.button("Run Evaluation", type="primary")

        if eval_btn:
            with st.spinner("Running evaluation..."):
                _, _, test_ds, _ = prepare_datasets()
                test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

                all_preds, all_labels = [], []
                model.eval()
                for batch in test_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    preds = torch.argmax(outputs.logits, dim=-1)
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(batch["labels"].cpu().tolist())

                y_pred = np.array(all_preds)
                y_true = np.array(all_labels)

                from sklearn.metrics import precision_recall_fscore_support, accuracy_score

                accuracy = accuracy_score(y_true, y_pred)
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_true, y_pred, labels=np.arange(len(classes)), zero_division=0
                )
                macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average="macro", zero_division=0
                )

                st.subheader("Overall Metrics")
                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy", f"{accuracy:.4f}")
                m2.metric("Macro F1", f"{macro_f1:.4f}")
                m3.metric("Test Samples", str(len(y_true)))

                st.subheader("Per-Class Metrics")
                rows = []
                for i, name in enumerate(classes):
                    rows.append({
                        "Class": name,
                        "Label": label_map.get(name, {}).get("label", name),
                        "Precision": f"{precision[i]:.4f}",
                        "Recall": f"{recall[i]:.4f}",
                        "F1": f"{f1[i]:.4f}",
                        "Support": int(support[i]),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(cm, cmap="Blues")
                ax.set_xticks(range(len(classes)))
                ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
                ax.set_yticks(range(len(classes)))
                ax.set_yticklabels(classes, fontsize=8)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                for i in range(len(classes)):
                    for j in range(len(classes)):
                        ax.text(j, i, cm[i, j], ha="center", va="center",
                                fontsize=9, color="white" if cm[i, j] > cm.max() / 2 else "black")
                plt.colorbar(im, ax=ax, shrink=0.8)
                st.pyplot(fig)
                plt.close()

# Inference
with tab3:
    st.header("Inference")

    if not model_exists:
        st.warning("No trained model found. Train the model first.")
    else:
        model, tokenizer, classes, device = load_trained_model()
        label_map = load_label_map()

        samples = {
            "Select a sample...": "",
            "Auth failure — missing token": "WARN [api-gateway] 401 returned to client client_3536: bearer token missing from Authorization header.",
            "DB timeout — connection pool exhausted": "CRITICAL [payments-core] ERROR [db-pool] Failed to acquire DB connection from pool: all 15 connections exhausted.",
            "Third-party API — 502 from provider": "ERROR [payout-service] Upstream provider Flutterwave returned 502. Retried 3 times. Failing over.",
            "Resource exhaustion — OOM": "CRITICAL [ml-inference] OOM error — JVM heap exhausted. Allocated: 7088MB. Killing process.",
            "Validation error — schema mismatch": "ERROR [ingestion-service] Schema validation failed: field 'amount' expected float, got string. Record ID: rec_65194.",
        }

        sample_choice = st.selectbox("Sample logs", list(samples.keys()))

        log_input = st.text_area(
            "Log message",
            value=samples[sample_choice] if sample_choice != "Select a sample..." else "",
            placeholder="Paste a log message or select a sample above...",
            height=100,
        )

        if st.button("Classify", type="primary") and log_input.strip():
            with st.spinner("Classifying..."):
                from src.inference import generate_summary

                inputs = prepare_single_message(log_input.strip(), tokenizer)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).squeeze().detach().cpu().numpy()
                pred_idx = int(torch.argmax(outputs.logits, dim=-1).item())
                confidence = float(probs[pred_idx])
                predicted_label = classes[pred_idx]
                summary = generate_summary(log_input.strip(), predicted_label, confidence, label_map)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Prediction")
                    st.metric("Root Cause", f"{predicted_label} — {label_map[predicted_label]['label']}")
                    st.metric("Confidence", f"{confidence:.4f}")
                    st.markdown("**Severity:** " + summary["severity"])
                    st.markdown("**Affected component:** " + summary["affected_component"])
                    st.markdown("**Suggested action:** " + summary["suggested_action"])

                with col2:
                    st.subheader("Class Probabilities")
                    prob_data = pd.DataFrame({
                        "Class": [f"{c} — {label_map.get(c, {}).get('label', c)}" for c in classes],
                        "Probability": probs,
                    }).sort_values("Probability", ascending=False)
                    st.bar_chart(prob_data.set_index("Class"), height=300)

                if summary["extracted_details"]:
                    st.subheader("Extracted Details")
                    st.json(summary["extracted_details"])
