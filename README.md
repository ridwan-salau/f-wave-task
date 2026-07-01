# Log Root Cause Classification

Classifies system error logs into 8 root cause categories using a fine-tuned ModernBERT model, and generates structured summaries for each entry.

## Project Structure

```
├── app.py                          # Streamlit UI (train, evaluate, inference)
├── Dockerfile                      # Docker build
├── data/dataset.xlsx               # 120 labeled log entries + label metadata
├── src/
│   ├── preprocess.py               # Data loading, splitting, tokenization
│   ├── train.py                    # ModernBERT fine-tuning
│   ├── evaluate.py                 # Accuracy, precision, recall, F1
│   └── inference.py                # Prediction + summary generation
├── models/                         # Saved model artifacts
├── outputs/                        # Evaluation reports
├── requirements.txt
└── README.md
```

## Setup

```bash
# CLI
pip install -r requirements.txt
python -m src.train --epochs 10 --lr 3e-5
python -m src.evaluate
python -m src.inference -m "ERROR [payment-gateway] Upstream provider Stripe returned 502."

# Streamlit UI
streamlit run app.py

# Docker
docker build -t log-classifier .
docker run -p 8501:8501 log-classifier
```

## Model Approach

I used ModernBERT (answerdotai/ModernBERT-base) with a classification head fine-tuned on the log messages. The main reasons:

- **Small dataset (120 samples)**: Training a model from scratch isn't feasible. ModernBERT's pre-trained representations transfer well — it already understands error semantics, timeouts, and system failures from its training corpus.
- **Semi-structured text**: Log messages mix natural language ("returned 502", "connection timed out") with technical identifiers ("client_3536", "pg-replica-02"). Subword tokenization handles both without custom preprocessing.
- **Lightweight**: 149M parameters, fine-tunes in a couple of minutes on CPU. Single saved model file with no external dependencies at inference time.

I considered TF-IDF + Logistic Regression as a baseline (~65% accuracy) but it struggled with semantically equivalent phrasings — e.g., treating "pool exhausted" and "all connections exhausted" as unrelated features. ModernBERT's contextual embeddings handle this naturally.

## Data Preprocessing

- Messages loaded from the `log_message` column, labels from `root_cause_label`
- 8 string labels encoded to integers 0–7 via `LabelEncoder`
- Stratified 70/15/15 train/val/test split (84/18/18 samples) with seed=42
- Tokenized with `AutoTokenizer` to max 512 tokens (ModernBERT supports 8,192; 512 is ample for error logs)
- No text cleaning — status codes, component names, and log levels are preserved as signal

## Evaluation Results

Test set: 18 held-out samples.

| Metric | Value |
|--------|-------|
| Accuracy | 94.44% |
| Macro F1 | 0.9333 |
| Weighted F1 | 0.9407 |

Per-class breakdown:

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| RC-01 | 1.0000 | 1.0000 | 1.0000 | 3 |
| RC-02 | 1.0000 | 1.0000 | 1.0000 | 2 |
| RC-03 | 1.0000 | 1.0000 | 1.0000 | 2 |
| RC-04 | 1.0000 | 1.0000 | 1.0000 | 2 |
| RC-05 | 1.0000 | 1.0000 | 1.0000 | 3 |
| RC-06 | 0.6667 | 1.0000 | 0.8000 | 2 |
| RC-07 | 1.0000 | 1.0000 | 1.0000 | 2 |
| RC-08 | 1.0000 | 0.5000 | 0.6667 | 2 |

The single error: one RC-08 (Network) misclassified as RC-06 (Permissions) — likely due to the log containing an IP address and connection-refused language that overlapped with auth-related patterns at this sample size.

## Tradeoffs

- Preserving raw log tokens (status codes, component names) helped accuracy but means the model is tied to this log format. A change in logging conventions would require retraining.
- ModernBERT assigns high confidence to clear-cut cases (often >0.95) but is less certain on edge cases where vocabulary overlaps between classes. A confidence threshold for human escalation would still be useful.
- RC-08 (Network) and RC-06 (Permissions) can overlap when logs involve connection-refused language combined with IP addresses or client identifiers. With only ~10 training samples per class, this is fundamentally a data issue rather than a model choice.
- Regex-based summary extraction works for this dataset's consistent format but wouldn't generalize to unstructured logs.

## Limitations

1. **120 samples total**: The dominant constraint. The model can't learn rare phrasings or edge cases. A single mislabeled example could distort a class boundary with this little data.
2. **Synthetic data bias**: The logs follow a consistent `LEVEL [component] message` pattern. Real logs are noisier — multi-line stack traces, inconsistent formatting, truncated messages.
3. **English-only**: The tokenizer and pre-training are English-specific.
4. **Capped at 512 tokens**: ModernBERT supports 8,192, but we use 512 for training/inference speed. Real stack traces could exceed even 8,192 and would need chunking.
5. **Regex-based detail extraction**: Works on this dataset but would need NER or a lightweight LLM for real-world log formats.

## Productionizing

This log classifier will be deployed as a microservice in a environment with GPU support for inference. A typical inference-optimized framework that can be used is Nvidia Triton. It supports both batch and online inference.

### Monitoring & drift
- Track prediction confidence distribution over time — a shift toward low confidence indicates data drift or new error types.
- Monitor class distribution — a sudden spike in one category likely signals a real production issue, but could also indicate concept drift.
- Log every prediction alongside engineer feedback (accepted/overridden) as ground truth for retraining.

### Scaling
- Export the model via `torch.jit.script` or ONNX for faster inference. Serve behind a thin FastAPI/gRPC service.
- Buffer incoming logs into mini-batches for efficient GPU utilization.
- The model is stateless (~574 MB). Auto-scale replicas based on queue depth rather than CPU/memory.
- Cache predictions for identical or near-identical log messages — in practice this can reduce inference volume significantly.

### Reliability
- Fall back to keyword-based classification if the model service is unavailable. Incident response should never block on ML.
- Shadow-deploy new model versions alongside production for a week before promoting.
- Version every model and include model version in inference logs for rollback capability.

### Next steps to improve the model
- Active learning: use low-confidence predictions to prioritize logs for human labeling.
- Add `service` and `severity` as additional input features alongside the log message.
- Hierarchical classification: first classify broad category (external dependency vs internal resource), then refine to specific root cause.
- Data augmentation via synonym replacement in the natural language portions of log messages for low-sample classes.
