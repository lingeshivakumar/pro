# Dual-Branch ALBERT + GIN Email Classification with Explainable AI & Vector Memory

A state-of-the-art multimodal deep learning architecture combining deep textual transformers (**ALBERT**) and graph neural networks (**GIN**) with learned **Gated Multimodal Fusion**, an end-to-end **Explainable AI (XAI) Suite**, and a **Dynamic Vector Memory** for enterprise email security:

1. **Primary Classification (3 Classes):** `HAM`, `SPAM`, and `PHISHING`
2. **Fine-Grained Phishing Subtype Classification (10 Classes):** `NOT_PHISHING`, `BEC`, `CREDENTIAL_PHISHING`, `DELIVERY_PHISHING`, `FINANCIAL_PHISHING`, `GENERIC_PHISHING`, `GOVERNMENT_PHISHING`, `MALWARE_DELIVERY`, `SURVEY_PHISHING`, `TECH_SUPPORT_PHISHING`.
3. **Multi-Modal Explainability:** Controlled branch ablation, word-level Integrated Gradients, GNNExplainer entity mapping, per-URL multi-modal perturbation, and conversational explanation.
4. **Dynamic Vector Memory:** Zero-shot incident cataloging and $k$-NN cosine retrieval on 256-dim fused latent embeddings.

---

## 📁 Project Structure

```
pro/
│
├── config.py             # Global configurations, hyperparameters, device detection (CUDA/MPS/CPU), and paths
├── models.py             # Dual-branch architecture: ALBERT + GIN + Gated Fusion + Multi-task Heads
├── dataset.py            # Heterogeneous graph construction (build_email_graph) & PyTorch DataLoaders
├── infer.py              # Immediate inference engine and CLI (No retraining needed!)
├── xai.py                # Comprehensive Explainable AI Suite (Ablation, IG, GNNExplainer, URL attribution)
├── memory.py             # Dynamic Vector Memory (Cosine similarity retrieval & incident bank)
├── app.py                # Modern Local Web GUI Dashboard (Interactive email analysis & live XAI)
├── evaluate.py           # Test set evaluation, metrics, Markdown/LaTeX table export
├── train.py              # Full multi-GPU / single-GPU / Apple Silicon / CPU training loop
├── visualize.py          # t-SNE projections and k-NN latent space connectivity plots
├── requirements.txt      # Python package dependencies
│
├── dataset/              # Local dataset folder
│   └── augmented_66k.parquet (66,000 email records)
│
├── tests/                # Unit and integration test suite
│   └── test_xai_pipeline.py  # Tests for XAI faithfulness, stability, and pipeline integrity
│
└── output/               # Pre-trained artifacts & outputs
    ├── checkpoints/      # Trained model weights (best_model.pt)
    ├── metrics/          # Classification reports, training history, label mapping
    ├── tables/           # Markdown and LaTeX performance reports & XAI evaluation
    ├── images/           # Loss curves, confusion matrices, t-SNE, k-NN graph
    ├── embeddings/       # Extracted .npy embeddings (fused, graph, text)
    ├── layer_outputs/    # Detailed layer states & statistics
    └── interactive_heterogeneous_graph.html  # Standalone interactive vis.js graph
```

---

## 🚀 Quickstart: Classify Emails Instantly

The pre-trained model weights ([best_model.pt](output/checkpoints/best_model.pt)) and label encodings ([label_mapping.json](output/metrics/label_mapping.json)) allow instant classification without requiring any cloud APIs or retraining.

### 1. Run the Built-in Demo
Test realistic phishing, BEC, ham, and spam samples:
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
    body="Your direct deposit failed. Verify your bank account here: http://secure-portal.biz",
    sender="payroll@company-support-auth.com",
    has_spf=False,
    has_dkim=False
)

print("Verdict:", result["primary_label"])
print("Confidence:", f"{result['primary_confidence']*100:.2f}%")
print("Phishing Risk Score:", f"{result['phishing_risk_score']*100:.2f}%")
print("Subtype:", result["subtype_label"])
```

---

## 🖥️ Interactive Web GUI Dashboard

Launch the zero-dependency, local web GUI:
```bash
python app.py
```
Open **`http://localhost:7860`** in any browser:
- **Preset Scenarios:** Pre-loaded samples (PayPal Credential Phishing, CEO Wire Transfer BEC, Commercial Spam, Legitimate Corporate Ham).
- **Interactive Visual XAI:** Word importance heatmaps, GNN relational graph, and modality ablation sliders.
- **Dynamic Vector Memory:** Save flagged incidents into the local memory bank and perform zero-shot recurring campaign matching.
- **Conversational XAI Assistant:** In-browser chatbot explaining classification rationale, suspicious domains, and authentication indicators.

---

## 🔍 Explainable AI (XAI) Suite

The system includes a dedicated Explainable AI engine ([xai.py](xai.py)) designed specifically for dual-branch multimodal architectures:

### 1. Controlled Branch-Level Modality Ablation
Measures whether the model relies predominantly on linguistic cues (Text branch) or topological cues (Graph branch):
- **Neutral Baseline Substitution:** Replaces text representations with uninformative sequence embeddings and graph representations with null empty-graph baselines.
- **Causal Metric ($\Delta P$):** Quantifies exact target class probability drops:
  $$\Delta P_{\text{text}} = P(\hat{y} \mid \text{full}) - P(\hat{y} \mid \text{text-ablated})$$
  $$\Delta P_{\text{graph}} = P(\hat{y} \mid \text{full}) - P(\hat{y} \mid \text{graph-ablated})$$
- **Internal Fusion Diagnostic:** Inspects the learned gating activation $\alpha \in [0, 1]$ from the Gated Fusion layer.

### 2. Word-Level Attribution via Integrated Gradients
- Computes path-integral gradients across ALBERT embeddings:
  $$\text{Attr}_i(x) = (x_i - x_i') \times \int_{0}^{1} \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$
- Automatically aggregates SentencePiece subword tokens into coherent words, highlighting linguistic triggers (e.g., *"urgent"*, *"suspended"*, *"wire transfer"*).

### 3. Graph Explainability via GNNExplainer
- Optimizes continuous edge masks $M \in [0, 1]^{|E|}$ and node feature masks to maximize mutual information with the predicted class.
- Identifies the critical subgraphs (e.g. Sender $\leftrightarrow$ Sender Domain $\leftrightarrow$ Malicious Destination URLs).

### 4. Multi-Modal Per-URL Attribution
- Perturbs each extracted hyperlink individually across both text and graph branches.
- Measures the exact isolated risk contribution ($\Delta P_{\text{text}}$, $\Delta P_{\text{graph}}$, and $\Delta P_{\text{combined}}$) for every link in the email.

### 5. Faithfulness & Stability Protocols
The XAI outputs are validated via rigorous statistical evaluation protocols:
- **Comprehensiveness (Faithfulness):** Verifies that masking top-attribution tokens produces significantly larger probability drops than random or low-attribution tokens.
- **Invariance (Stability):** Measures Spearman rank correlation ($\rho \approx 0.956$) and top-token Jaccard overlap under harmless formatting and punctuation perturbations.

### 6. Conversational Offline XAI Agent
Ask natural language questions about any classification:
```python
from infer import EmailClassifierInference
from xai import conversational_xai_agent, compute_xai_bundle

engine = EmailClassifierInference()
result = engine.predict(subject="Urgent update", body="Click here: http://fake.com", sender="phish@evil.com")
bundle = compute_xai_bundle(engine, subject="Urgent update", body="Click here: http://fake.com", sender="phish@evil.com")

explanation = conversational_xai_agent("Why did you flag this email?", result, bundle)
print(explanation)
```

---

## 🧠 Dynamic Vector Memory

The system features an on-device episodic memory bank ([memory.py](memory.py)):
- **256-Dimensional Fused Embeddings:** Stores latent representations $z_{\text{fused}}$ of novel email threats.
- **Zero-Shot Pattern Matching:** Performs real-time cosine similarity search ($\ge 72\%$ threshold) against cataloged campaigns without fine-tuning or retraining.
- **Persistent Storage:** Stored in `output/memory/entries.json` and `output/memory/vectors.npy`.

---

## 🧪 Comprehensive Test Suite

Run the full unit and integration test suite (covers base inference, XAI ablation, Integrated Gradients, GNNExplainer, and schema stability):

```bash
python -m unittest tests/test_xai_pipeline.py -v
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

## 💻 Hardware & Cross-Platform Support

The codebase automatically detects and utilizes the best available compute device:
- **NVIDIA GPUs (CUDA):** Enables mixed-precision acceleration (`torch.amp`).
- **Apple Silicon (M1/M2/M3/M4):** Automatically utilizes Apple's **MPS** (Metal Performance Shaders) acceleration.
- **CPU:** Transparent CPU fallback for any standard laptop or environment.

---

## 🏋️ Optional: Retraining from Scratch

To fine-tune or retrain with custom parameters:
```bash
python train.py --epochs 5 --batch-size 16
```
- Differential learning rates (`2e-5` for ALBERT transformer backbone, `1e-3` for GIN and classification heads).
- Automatic mixed precision (`torch.amp`) on CUDA GPUs.
- Dynamic cosine annealing learning rate scheduler.
- Automatically checkpoints the best model on validation accuracy.
