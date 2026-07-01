
THE SCENARIO

You are joining the engineering team at a fictional technology platform that processes high volumes of digital activity. The operations team receives thousands of system error logs daily and currently investigates them manually. Your task is to build a small AI-powered prototype that:



Classifies log entries into predefined root cause categories
Generates a short, structured summary of each issue
Provides basic evaluation metrics for classification performance


Attached to this email: a synthetic dataset of log entries and a set of predefined root cause labels.


WHAT TO BUILD

Build a lightweight AI pipeline that:

Trains a classification model on the provided dataset
Predicts root cause categories for unseen log entries
Generates a concise, structured issue summary per entry
Outputs evaluation metrics


This does not need to be production-ready. We are evaluating your technical capability and engineering judgment — not the scale of your stack.


DELIVERABLES

1. Code — push to your own GitHub (or equivalent) repo and share the link. Your repo should include:

A structured project folder (not just a notebook)
Separate files or modules for training and inference logic
A requirements.txt (or equivalent)
Evaluation output from your trained model


2. README.md (required) — your README should cover:

Model approach and your reasoning behind it
Data preprocessing steps
Evaluation results: accuracy, precision, recall, and F1
Observed tradeoffs
Limitations of your solution
How you would productionize this system — covering monitoring, drift detection, scaling, and reliability


3. Demo video (required, max 5 minutes) — a narrated screen recording showing:

Training the model
Running inference
Viewing and interpreting the results