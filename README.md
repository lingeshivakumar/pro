# Dual-Branch ALBERT + GIN Email Classification (Local Execution)

A state-of-the-art multimodal architecture combining deep textual transformers (**ALBERT**) and graph neural networks (**GIN**) with learned **Gated Fusion** for email security classification:
1. **Primary Classification (3 Classes):** `HAM`, `SPAM`, and `PHISHING`
2. **Fine-Grained Phishing Subtype Classification (10 Classes):** `NOT_PHISHING`, `BEC`, `CREDENTIAL_PHISHING`, `DELIVERY_PHISHING`, `FINANCIAL_PHISHING`, `GENERIC_PHISHING`, `GOVERNMENT_PHISHING`, `MALWARE_DELIVERY`, `SURVEY_PHISHING`, `TECH_SUPPORT_PHISHING`.

This project has been modularized from [notebook6ae1b5207e.ipynb](notebook6ae1b5207e.ipynb) and configured for **seamless local execution** without requiring Kaggle or cloud resources.

---

## Project Structure

```
d:/pro/
│
├── config.py             # Global configurations, hyperparameters, device, and paths
├── models.py             # Dual-branch architecture: ALBERT + GIN + Gated Fusion + Multi-task Heads
├── dataset.py            # Graph construction (build_email_graph) & PyTorch DataLoaders
├── infer.py              # Immediate inference engine and CLI (No retraining needed!)
├── app.py                # Modern Local Web GUI Dashboard (Interactive email analysis)
├── evaluate.py           # Test set evaluation, metrics, Markdown/LaTeX table export
├── train.py              # Full multi-GPU / single-GPU / CPU training loop
├── visualize.py          # t-SNE projections and k-NN latent space connectivity plots
├── requirements.txt      # Python package dependencies
│
├── dataset/              # Local dataset folder
│   └── augmented_66k.parquet (66,000 email records)
│
└── output/               # Pre-trained artifacts & outputs
    ├── checkpoints/      # Trained model weights (best_model.pt)
    ├── metrics/          # Classification reports, training history, label mapping
    ├── tables/           # Markdown and LaTeX performance reports
    ├── images/           # Loss curves, confusion matrices, t-SNE, k-NN graph
    ├── embeddings/       # Extracted .npy embeddings (fused, graph, text)
    ├── layer_outputs/    # Detailed layer states & statistics
    └── interactive_heterogeneous_graph.html  # Standalone interactive vis.js graph
```

---

## 🚀 Quickstart: Classify Emails Instantly (No Training Required!)

The model weights ([best_model.pt](output/checkpoints/best_model.pt)) and label encodings ([label_mapping.json](output/metrics/label_mapping.json)) are already present in `output/`. You can classify emails immediately!

### 1. Run the Built-in Demo
Test the model on realistic phishing, ham, and spam samples:
```bash
python infer.py --demo
```

### 2. Classify a Custom Email via CLI
```bash
python infer.py --subject "Urgent: Your account is suspended" --body "Please click here to verify your password immediately: http://phish-secure.com" --sender "security@fake-support.com"
```

### 3. Programmatic Usage in Python
```python
from infer import EmailClassifierInference

classifier = EmailClassifierInference()
result = classifier.predict(
    subject="Action Required: Update Bank Details",
    body="Your payroll direct deposit failed. Verify your bank account credentials here: http://secure-banking-portal.ru",
    sender="payroll@company-support-auth.com",
    has_spf=False,
    has_dkim=False
)

print("Verdict:", result["primary_label"])
print("Confidence:", result["primary_confidence"])
print("Phishing Risk Score:", result["phishing_risk_score"])
print("Subtype:", result["subtype_label"])
```

---

## 📊 Run Test Set Evaluation

Evaluate the pre-trained model on the 9,900 held-out test split from `dataset/augmented_66k.parquet`:
```bash
python evaluate.py
```
This generates:
- Printed performance tables (HAM / SPAM / PHISHING & 10 Subtypes)
- Exported reports in `output/tables/` (`primary_report.md`, `primary_report.tex`, `subtype_report.md`, `subtype_report.tex`)
- Updated confusion matrix heatmaps in `output/images/`

---

## 📈 Visualizations & Latent Analysis

To recreate the t-SNE 2D projections and $k$-NN latent space similarity graphs:
```bash
python visualize.py --all
```
Open [output/interactive_heterogeneous_graph.html](output/interactive_heterogeneous_graph.html) directly in any web browser to explore interactive entity relationships (Emails, Senders, Domains, URLs).

---

## 🏋️ Optional: Retraining from Scratch

If you ever wish to fine-tune or train with new parameters:
```bash
python train.py --epochs 5 --batch-size 16
```
Training features:
- Differential learning rates (`2e-5` for ALBERT transformer backbone, `1e-3` for GIN and classification heads)
- Automatic mixed precision (`torch.amp`) on CUDA GPUs
- Dynamic cosine annealing learning rate scheduler
- Automatically saves the best checkpoint whenever validation accuracy improves.
