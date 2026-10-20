# SYSTEM PROMPT & REPOSITORY CONTEXT: DUAL-BRANCH ALBERT + GIN EMAIL CLASSIFICATION

> **Instructions for ChatGPT / LLM:**
> You are acting as an expert Machine Learning Engineer, Security Data Scientist, and Research Advisor. The user has provided you with the comprehensive architecture, mathematical formulation, codebase specification, and empirical results for their project: **"Dual-Branch ALBERT + GIN Email Classification with Explainable AI & Dynamic Vector Memory"**.
> 
> Use the structured information below to answer user questions, assist with code refactoring, suggest research paper draft sections (IEEE/ACM format), optimize neural hyperparameters, debug issues, or implement new extensions (e.g., handling fine-grained subtype class imbalance, expanding graph relations, or deploying to production).

---

## 1. Executive Summary & Research Objectives

This project implements an end-to-end multimodal deep learning pipeline for advanced email security classification. It addresses key vulnerabilities in text-only phishing detectors by integrating **linguistic transformer representations** with **relational graph neural network embeddings**, synthesized through a learned **gated multimodal fusion layer**.

### Core Problem Solved:
1. **Linguistic Sophistication & Semantic Obfuscation:** Attackers craft contextually convincing phishing emails (e.g., Business Email Compromise, spear phishing) that bypass traditional regex and keyword filters.
2. **Structural & Identity Inconsistencies:** Phishing campaigns exhibit subtle topological discrepancies between the sender, sender domain, authentication mechanisms (SPF, DKIM), and embedded hyperlinks/destination domains.
3. **Multi-Task Granularity:** Beyond coarse detection (`HAM` vs `SPAM` vs `PHISHING`), security teams need fine-grained intent classification (`BEC`, `CREDENTIAL_PHISHING`, `DELIVERY_PHISHING`, `FINANCIAL_PHISHING`, etc.) and interpretable rationale.
4. **Offline Explainability & Zero-Leakage Privacy:** The inference engine, Explainable AI (XAI) suite, and vector memory run 100% locally on-device without cloud API dependencies.

---

## 2. System Architecture & Mathematical Foundations

```
Raw Email Input (Subject, Body, Sender, Receiver, URLs, SPF, DKIM)
   │
   ├───> [Text Branch] ───────────────> ALBERT-large-v2 (1024-dim) ──┐
   │                                                                 │
   └───> [Graph Branch] ──────────────> 3-Layer GIN + JK (128-dim) ──┴──> [Gated Multimodal Fusion]
                                                                                   │
                                                                           z_fused (256-dim)
                                                                                   │
                                         ┌─────────────────────────────────────────┴─────────────────────────────────────────┐
                                         ▼                                                                                    ▼
                           [Primary Classification Head]                                                        [Subtype Classification Head]
                                3 Classes: HAM / PHISHING / SPAM                                                   10 Phishing Subtypes
```

### 2.1 Branch 1: Textual Deep Representation (ALBERT)
- **Model Backbone:** `albert-large-v2` (`hidden_dim = 1024`, 24 layers sharing parameters, 16 attention heads).
- **Input Representation:** `[CLS] Subject [SEP] Body [SEP]`, tokenized with byte-pair encoding, truncated/padded to `max_len = 256`.
- **Textual Embedding:** The pooled `[CLS]` token output $h_{text} \in \mathbb{R}^{1024}$.

### 2.2 Branch 2: Relational Graph Branch (Graph Isomorphism Network - GIN)
- **Graph Formulation ($G = (V, E)$):**
  - **Node Set $V$ (up to $N = 14$ nodes):**
    - Node 0: Sender entity
    - Node 1: Receiver entity
    - Node 2: Sender domain
    - Nodes $3 \dots K$ ($K \le 13$): Extracted destination domains & hyperlinks (max 10 external destinations).
  - **Edge Set $E$ (Adjacency Matrix $A \in \mathbb{R}^{14 \times 14}$):**
    - Bidirectional connections between Sender and Receiver.
    - Edges between Sender and Sender Domain.
    - Edges between Sender Domain and Destination Domains.
    - Self-loops added to preserve self-node representations.
  - **Node Feature Vector $X \in \mathbb{R}^{14 \times 16}$:**
    - Dimension `0..3`: One-hot encoding of entity category (Sender, Receiver, Sender Domain, External URL/Domain).
    - Dimension `4`: SPF authentication flag ($1.0$ valid, $0.0$ failed/absent).
    - Dimension `5`: DKIM authentication flag ($1.0$ valid, $0.0$ failed/absent).
    - Dimension `6`: Normalized URL count ($\min(\text{num\_urls} / 10.0, 1.0)$).
    - Dimension `7`: Normalized domain count ($\min(\text{num\_domains} / 10.0, 1.0)$).
    - Dimension `8`: Normalized body length ($\min(\text{body\_len} / 5000.0, 1.0)$).
    - Dimensions `9..15`: Reserved topological attributes and zero-padding.

- **GIN Convolutional Message Passing:**
  GIN offers maximal expressive power among message-passing GNNs (equivalent to the Weisfeiler-Lehman 1-WL graph isomorphism test):
  $$h_v^{(k)} = \text{MLP}^{(k)} \left( (1 + \epsilon^{(k)}) h_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)} \right)$$
  - Implemented in batched tensor matrix operations via `torch.bmm(adj, x)`.
  - $\text{MLP}^{(k)}$ consists of `Linear(in_dim, hidden_dim) -> BatchNorm1d -> ReLU -> Linear -> BatchNorm1d -> ReLU`.
  - Hidden dimension per GIN layer = $64$.
  - Number of GIN layers = $3$.

- **Jumping Knowledge (JK) & Global Graph Readout:**
  Instead of retaining only the final layer representation, multi-scale representations across iterations are aggregated:
  $$\text{MeanPool} = \frac{1}{|V|} \sum_{v \in V} h_v^{(3)}, \quad \text{MaxPool} = \max_{v \in V} h_v^{(3)}$$
  $$h_{graph} = [\text{MeanPool} \,\|\, \text{MaxPool}] \in \mathbb{R}^{128}$$

### 2.3 Branch 3: Trainable Gated Multimodal Fusion
Rather than naive concatenation or simple summation, the network learns an input-dependent gating function that balances the decisiveness of linguistic vs structural topological features:
1. Linear projections to common embedding space ($d_{fused} = 256$):
   $$\tilde{h}_{text} = \text{Linear}_{text}(h_{text}) \in \mathbb{R}^{256}$$
   $$\tilde{h}_{graph} = \text{Linear}_{graph}(h_{graph}) \in \mathbb{R}^{256}$$
2. Dynamic sigmoid gating mechanism:
   $$g = \sigma \left( W_g [h_{text} \,\|\, h_{graph}] + b_g \right) \in (0, 1)^{256}$$
3. Gated convex combination:
   $$z_{fused} = g \odot \tilde{h}_{text} + (1 - g) \odot \tilde{h}_{graph} \in \mathbb{R}^{256}$$
   *(Note: The mean scalar value $\bar{g} = \text{mean}(g)$ directly reveals the model's text-vs-graph decision attribution).*

### 2.4 Multi-Task Classification Heads
- **Primary Head (3 classes):**
  $$\hat{y}_{prim} = \text{Softmax}(\text{Linear}(z_{fused})) \in \mathbb{R}^3 \quad (\text{HAM: 0, PHISHING: 1, SPAM: 2})$$
- **Subtype Head (10 classes):**
  $$\hat{y}_{sub} = \text{Softmax}(\text{Linear}(z_{fused})) \in \mathbb{R}^{10}$$
  Subtypes: `NOT_PHISHING (0)`, `BEC (1)`, `CREDENTIAL_PHISHING (2)`, `DELIVERY_PHISHING (3)`, `FINANCIAL_PHISHING (4)`, `GENERIC_PHISHING (5)`, `GOVERNMENT_PHISHING (6)`, `MALWARE_DELIVERY (7)`, `SURVEY_PHISHING (8)`, `TECH_SUPPORT_PHISHING (9)`.

### 2.5 Multi-Task Training Objective
$$\mathcal{L}_{total} = \mathcal{L}_{CE}(\hat{y}_{prim}, y_{prim}) + \lambda \cdot \mathcal{L}_{CE}(\hat{y}_{sub}, y_{sub})$$
- Optimized with differential learning rates:
  - $\eta_{ALBERT} = 2 \times 10^{-5}$ (fine-tuning transformer backbone with AdamW, weight decay $0.01$).
  - $\eta_{GIN} = 1 \times 10^{-3}$ (training GIN, gated fusion, and classification heads from scratch).
  - Cosine annealing learning rate scheduling and Automatic Mixed Precision (`torch.amp.autocast`).

---

## 3. Repository File Structure & Module Responsibilities

```
d:/pro/
│
├── config.py                 # Central configurations, hyperparameters, dynamic path resolver, label maps
├── models.py                 # Neural architectures: GINLayer, GINBranch, GatedMultimodalFusion, AlbertGINClassifier
├── dataset.py                # Graph construction (build_email_graph), parquet loader, PyTorch Dataset & DataLoaders
├── infer.py                  # Standalone CLI & programmatic inference engine (zero retraining needed)
├── xai.py                    # 100% offline Explainable AI suite (Attributions, Counterfactuals, Conversational Agent)
├── memory.py                 # Dynamic Vector Memory (stores 256-dim z_fused embeddings, fast cosine retrieval)
├── app.py                    # Local interactive Web GUI dashboard (port 7860, threaded Python HTTP server)
├── evaluate.py               # Comprehensive test set evaluation script, outputs Markdown & LaTeX tables
├── train.py                  # Distributed / multi-GPU / CPU training script with AMP & checkpoints
├── visualize.py              # t-SNE latent space projections, k-NN graph plots, and interactive HTML graphs
├── requirements.txt          # Python dependencies (torch, transformers, pandas, scikit-learn, networkx, etc.)
├── README.md                 # Project documentation and quickstart instructions
│
├── dataset/
│   └── augmented_66k.parquet # 66,000 augmented, balanced email records with metadata
│
└── output/
    ├── checkpoints/
    │   └── best_model.pt     # Saved PyTorch checkpoint (trained model state dict)
    ├── metrics/
    │   ├── classification_report_primary.csv
    │   ├── classification_report_subtype.csv
    │   ├── label_mapping.json
    │   ├── training_history.csv
    │   └── training_history.json
    ├── tables/
    │   ├── primary_report.md & .tex
    │   └── subtype_report.md & .tex
    ├── images/
    │   ├── confusion_matrices, loss curves, t-SNE plots, k-NN latent graphs
    ├── embeddings/           # Extracted .npy embeddings (fused 256d, graph 128d, text 1024d)
    ├── layer_outputs/        # Intermediate layer representations
    └── interactive_heterogeneous_graph.html  # Standalone interactive vis.js graph
```

---

## 4. Key Module Code & Interface Specifications

### 4.1 `config.py`
- Resolves dataset from candidate paths (`dataset/augmented_66k.parquet` or fallback paths).
- Sets `DEVICE = cuda:0` (if available) or `cpu`.
- Defines default hyperparams: `MODEL_NAME = "albert-large-v2"`, `MAX_LEN = 256`, `BATCH_SIZE = 16`, `EPOCHS = 5`.
- Defines label dictionaries for primary (3) and subtype (10) tasks.

### 4.2 `models.py`
- `GINLayer(in_dim, out_dim, eps=0.0)`: Performs $(1 + \epsilon)X + A X$ followed by a 2-layer MLP with BatchNorm and ReLU.
- `GINBranch(in_dim=16, hidden_dim=64, num_layers=3)`: Outputs 128-dim graph embedding via concatenation of mean and max pooling across node dimensions.
- `GatedMultimodalFusion(text_dim=1024, graph_dim=128, fused_dim=256)`: Computes gating weights and returns 256-dim fused vector $z_{fused}$ and gate weight tensor.
- `AlbertGINClassifier`: Wraps ALBERT transformer, GIN branch, Gated Fusion, and two classification heads (`head_primary` and `head_subtype`).
- `load_trained_model(checkpoint_path, device)`: Loads weights safely onto CPU/GPU.

### 4.3 `dataset.py`
- `build_email_graph(row)`: Takes a dictionary or dataframe row with email metadata (`domains`, `has_spf`, `has_dkim`, `num_urls`, `num_domains`, `body_length`) and builds the $[14, 16]$ node feature matrix and $[14, 14]$ adjacency matrix.
- `EmailDataset(df, tokenizer, max_len)`: PyTorch Dataset yielding `input_ids`, `attention_mask`, `node_features`, `adjacency_matrix`, `primary_label`, `subtype_label`.
- `get_dataloaders(...)`: Splits 66,000 records into train (85%, 56,100 samples) and test (15%, 9,900 samples) stratified by primary class.

### 4.4 `infer.py`
- `EmailClassifierInference`:
  - Methods:
    - `extract_domains_from_urls(urls)`
    - `extract_sender_domain(sender)`
    - `predict(subject, body, sender, receiver, urls, has_spf, has_dkim, return_xai=False, return_embeddings=False)`
  - Returns:
    - `primary_label` (`HAM`, `SPAM`, or `PHISHING`), `primary_confidence`, `primary_probabilities`
    - `phishing_risk_score` ($0.0$ to $100.0\%$)
    - `subtype_label`, `subtype_confidence`
    - `xai`: Full explainability dictionary
    - `memory_alert`: Nearest vector threat recall alert if cosine similarity $> 0.88$.

### 4.5 `xai.py` (Explainable AI Engine - 100% Offline)
1. **Multimodal Fusion Attribution:** Computes percentage reliance on text semantics vs graph structure from the gate tensor:
   $$\text{Text \%} = \bar{g} \times 100\%, \quad \text{Graph \%} = (1 - \bar{g}) \times 100\%$$
2. **Entity Attribution Breakdown:** Calculates quantitative contribution scores for Sender, Sender Domain, URLs, SPF, and DKIM based on graph features and presence.
3. **Counterfactual "What-If" Engine:**
   - Evaluates alternative scenarios in real-time:
     - *What if SPF and DKIM authentication were enabled?*
     - *What if external suspicious URLs were stripped?*
     - *What if the sender domain matched the organizational domain?*
   - Quantifies the delta change in Phishing Risk Score ($\Delta \text{Risk}$).
4. **Categorized Threat Indicators:** Rule-based heuristic pattern matching across 4 threat vectors:
   - `CREDENTIAL_HARVESTING`: Login prompts, verify password, credential tokens.
   - `URGENCY_MANIPULATION`: 24 hours, immediate action, account suspension threats.
   - `FINANCIAL_FRAUD`: Wire transfers, routing numbers, invoices, payroll requests.
   - `SUSPICIOUS_LINK`: IP-based links, high-risk TLDs (`.ru`, `.xyz`, `.top`, `.click`, `.biz`, `.work`).
5. **Reliability & Disagreement Metric:** Measures alignment between linguistic intent and topological graph signals.
6. **Conversational XAI Agent:** Local rule-based/template-driven conversational answering agent that answers security analyst queries:
   - *"Why was this classified as phishing?"*
   - *"How can the sender fix this email to reach the inbox?"*
   - *"Which URL or domain triggered the highest suspicion?"*

### 4.6 `memory.py` (Dynamic Vector Threat Memory)
- Stores 256-dimensional $z_{fused}$ latent vectors along with metadata (subject, sender, timestamp, predicted label, risk score).
- Implements sub-millisecond cosine similarity search:
  $$S_C(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
- Automatically warns analysts if an incoming email is topologically or semantically identical/similar to a previously archived attack campaign.

### 4.7 `app.py` (Local Web GUI Dashboard)
- Interactive web application served via Python's built-in `http.server` on port `7860`.
- Dark-mode responsive UI with real-time email analysis:
  - Input fields for sender, receiver, subject, body, SPF/DKIM toggles.
  - Preset loader (Credential Phishing, BEC Wire Transfer, Commercial Spam, Legitimate Internal Ham).
  - Visual dials for Phishing Risk Score.
  - Interactive Fusion Gate attribution progress bar (Text vs Graph split).
  - Graph Entity Attribution bar chart.
  - Live Counterfactual Remediation Simulator.
  - Dynamic Vector Memory manager (view, search, and commit threat vectors).
  - Built-in Conversational Security Copilot chatbox.

---

## 5. Empirical Performance & Evaluation Results

Tested on the **9,900 held-out test emails** from the 66,000 augmented dataset:

### 5.1 Primary Classification Performance (3-Class)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **HAM** | **0.8381** | **0.9494** | **0.8903** | 3,300 |
| **PHISHING** | **0.8711** | **0.9567** | **0.9119** | 3,300 |
| **SPAM** | **0.8790** | **0.6761** | **0.7643** | 3,300 |
| **Overall Accuracy** | — | — | **86.07%** | 9,900 |
| **Macro Average** | 0.8628 | 0.8607 | 0.8555 | 9,900 |
| **Weighted Average** | 0.8628 | 0.8607 | 0.8555 | 9,900 |

#### Key Empirical Insights:
- **Phishing Detection is Exceptional:** The model achieves **95.67% recall** and **91.19% F1-score** on phishing emails, minimizing dangerous false negatives.
- **Ham Recognition is Highly Robust:** **94.94% recall** ensures legitimate business correspondence is rarely flagged incorrectly.
- **Spam Tradeoff:** Spam recall is lower (67.61%) because non-malicious marketing emails share text characteristics with both ham and promotional phishing, occasionally being classified as Ham or Phishing.

### 5.2 Subtype Classification Performance (10-Class) & Imbalance Note

| Subtype Class | Precision | Recall | F1-Score | Test Support |
| :--- | :---: | :---: | :---: | :---: |
| **NOT_PHISHING** | **0.9769** | **0.9294** | **0.9526** | 6,600 |
| **GENERIC_PHISHING** | **0.5551** | **0.9729** | **0.7069** | 2,066 |
| **BEC** | 0.0000 | 0.0000 | 0.0000 | 182 |
| **CREDENTIAL_PHISHING** | 0.0000 | 0.0000 | 0.0000 | 136 |
| **DELIVERY_PHISHING** | 0.0000 | 0.0000 | 0.0000 | 188 |
| **FINANCIAL_PHISHING** | 0.0000 | 0.0000 | 0.0000 | 168 |
| **GOVERNMENT_PHISHING** | 0.0000 | 0.0000 | 0.0000 | 257 |
| **MALWARE_DELIVERY** | 0.0000 | 0.0000 | 0.0000 | 226 |
| **SURVEY_PHISHING** | 0.0000 | 0.0000 | 0.0000 | 0 |
| **TECH_SUPPORT_PHISHING**| 0.0000 | 0.0000 | 0.0000 | 77 |
| **Overall Accuracy** | — | — | **82.26%** | 9,900 |

> **Critical Machine Learning Finding for ChatGPT Research Advice:**
> The primary 3-class classification is balanced and performs with high F1 ($> 86-91\%$). However, the fine-grained 10-class subtype head suffers from severe dataset class imbalance: `NOT_PHISHING` ($6,600$) and `GENERIC_PHISHING` ($2,066$) dominate the distribution, causing the minority classes (`BEC`, `FINANCIAL_PHISHING`, etc.) to predict zero under standard Cross-Entropy Loss.
> 
> *Targeted optimization opportunities for future iterations:*
> 1. Replacing Cross-Entropy with **Focal Loss** ($\gamma = 2.0$) or Class-Balanced Loss.
> 2. Minor class oversampling with SMOTE or synthetic text generation.
> 3. Two-stage hierarchical classification (first classify Phishing vs Non-Phishing, then route Phishing cases to a specialized subtype expert network).

---

## 6. How to Run & Operate the Codebase

### 6.1 Install Dependencies
```bash
pip install -r requirements.txt
```

### 6.2 Instant Inference (CLI & Demo)
```bash
# Run built-in realistic demo suite
python infer.py --demo

# Classify single email via CLI
python infer.py --subject "Password reset required" --body "Click here to login: http://suspicious-site.xyz" --sender "admin@bank-verify.com"
```

### 6.3 Launch the Interactive Web GUI
```bash
python app.py
# Opens dashboard at http://localhost:7860
```

### 6.4 Evaluate Model & Export Tables
```bash
python evaluate.py
# Generates classification reports, markdown tables, and LaTeX tables in output/
```

### 6.5 Retrain Model from Scratch
```bash
python train.py --epochs 5 --batch-size 16 --lr-transformer 2e-5 --lr-graph 1e-3
```

---

## 7. High-Value Prompts to Use with ChatGPT

When interacting with ChatGPT using this document, you can copy-paste any of the following prompt templates:

### Prompt 1: Research Paper Methodology Section
> *"Based on the attached architecture and mathematical formulation in this project context, draft a formal IEEE/ACM Transactions-style Methodology section detailing the Dual-Branch ALBERT + GIN network, the graph entity construction, and the Gated Multimodal Fusion mechanism."*

### Prompt 2: Fixing the Subtype Class Imbalance
> *"In Section 5.2 of the project context, the fine-grained phishing subtype head exhibits severe class imbalance where minority classes have 0 recall. Provide a PyTorch implementation of a Class-Balanced Focal Loss module and show how to modify `models.py` and `train.py` to weight the loss dynamically based on inverse class frequencies."*

### Prompt 3: Enhancing the Graph Formulation
> *"Currently, the graph branch constructs up to 14 nodes (Sender, Receiver, Sender Domain, and up to 10 destination domains/URLs). How can we extend `dataset.py` and `models.py` to support heterogeneous edge types (Relational GCN or Heterogeneous Graph Transformer - HGT) to model different relation semantics (e.g., 'SENT_FROM', 'CONTAINS_LINK', 'REDIRECTS_TO')?"*

### Prompt 4: Production Deployment & ONNX Export
> *"I want to deploy this model for ultra-low-latency email gateway filtering (<15ms per email). Provide a step-by-step guide and Python script to export both the ALBERT text branch and the GIN graph branch into optimized ONNX / TensorRT runtimes."*

### Prompt 5: Explainable AI (XAI) Expansion
> *"Review the XAI suite in `xai.py`. How can we integrate Integrated Gradients (Captum) for token-level word saliency maps alongside the existing Gated Multimodal Fusion attribution and counterfactual engine?"*

---

## 8. Summary of Pre-Trained Artifacts Present in Repository

- Checkpoint: `output/checkpoints/best_model.pt` (Complete trained weights for ALBERT backbone, GIN layers, Gated Fusion, and heads).
- Metric Reports: `output/metrics/` (CSV and JSON classification metrics for all splits).
- Interactive Visualization: `output/interactive_heterogeneous_graph.html` (Interactive Vis.js graph showing entity clustering and connectivity).
- Memory Vectors: `output/memory/` (Persistent `.npy` threat vectors and `entries.json`).
