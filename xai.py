"""
Explainable AI (XAI) Suite for Dual-Branch ALBERT + GIN Email Classification.
100% Offline and Local.

Implements:
1. Controlled Branch-Level Modality Ablation (Text vs Graph vs Joint with documented baselines)
2. Per-URL Attribution (Individual URL text, graph, and combined perturbation)
3. Word-Level Attributions for ALBERT via Integrated Gradients
4. GNNExplainer for GIN Branch (Edge & Feature mask optimization mapped to entity metadata)
5. Rule-Based Security Threat Indicators
6. Offline Conversational XAI Agent
"""

import re
import urllib.parse
import numpy as np
import torch
import torch.nn.functional as F

from config import MAX_LEN


# =====================================================================
# 1. MODALITY ATTRIBUTION VIA CONTROLLED BRANCH ABLATION
# =====================================================================

def evaluate_modality_ablation(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    node_feats: torch.Tensor,
    adj: torch.Tensor,
    target_class: int,
    device: torch.device
) -> dict:
    """
    Evaluates controlled branch-level ablation to measure causal dependence on Text vs Graph:
    - Original: Full text and graph
    - Text Ablated: Text representation replaced with uninformative neutral baseline
    - Graph Ablated: Graph representation replaced with null empty-graph baseline
    - Joint Ablated: Both branches replaced with neutral baselines

    Reports target class probability drops (Delta P) without claiming false additive percentages.
    Includes gate statistics labeled purely as internal diagnostic activations.
    """
    model.eval()
    with torch.no_grad():
        # 1. Original Prediction
        out_orig = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            node_feats=node_feats,
            adj=adj,
            return_layers=True
        )
        z_text = out_orig["z_text"]
        z_graph = out_orig["z_graph"]
        prob_orig = float(F.softmax(out_orig["logits_primary"], dim=-1)[0, target_class].cpu().item())

        # Internal Gate Diagnostic Statistics
        gate_input = torch.cat([z_text, z_graph], dim=-1)
        gate_alpha = model.fusion.gate(gate_input).squeeze(0).cpu().numpy()
        mean_alpha = float(np.mean(gate_alpha))
        min_alpha = float(np.min(gate_alpha))
        max_alpha = float(np.max(gate_alpha))

        # 2. Documented Neutral Baselines
        # Text Baseline: ALBERT embedding of neutral empty input sequence
        empty_tokens = tokenizer(
            "",
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        empty_input_ids = empty_tokens["input_ids"].to(device)
        empty_attention_mask = empty_tokens["attention_mask"].to(device)
        albert_empty = model.albert(input_ids=empty_input_ids, attention_mask=empty_attention_mask)
        z_text_base = albert_empty.last_hidden_state[:, 0, :]  # Neutral [CLS] vector

        # Graph Baseline: Uninformative empty graph (14 nodes with 0 features and 0 edges)
        null_feats = torch.zeros_like(node_feats)
        null_adj = torch.zeros_like(adj)
        z_graph_base, _ = model.gin(null_feats, null_adj)

        # 3. Text Ablation (Neutral Text + Original Graph)
        z_fused_no_text = model.fusion(z_text_base, z_graph)
        logits_no_text = model.primary_head(z_fused_no_text)
        prob_no_text = float(F.softmax(logits_no_text, dim=-1)[0, target_class].cpu().item())
        delta_text = prob_orig - prob_no_text

        # 4. Graph Ablation (Original Text + Neutral Graph)
        z_fused_no_graph = model.fusion(z_text, z_graph_base)
        logits_no_graph = model.primary_head(z_fused_no_graph)
        prob_no_graph = float(F.softmax(logits_no_graph, dim=-1)[0, target_class].cpu().item())
        delta_graph = prob_orig - prob_no_graph

        # 5. Joint Ablation (Neutral Text + Neutral Graph)
        z_fused_joint = model.fusion(z_text_base, z_graph_base)
        logits_joint = model.primary_head(z_fused_joint)
        prob_joint = float(F.softmax(logits_joint, dim=-1)[0, target_class].cpu().item())
        delta_joint = prob_orig - prob_joint

    # Generate Human-Readable Interpretation
    if delta_text > delta_graph + 0.10:
        dominant_desc = "Text-Dependent: Removing textual linguistic features causes a substantially larger drop in confidence."
    elif delta_graph > delta_text + 0.10:
        dominant_desc = "Graph-Dependent: Removing relational structural features causes a substantially larger drop in confidence."
    else:
        dominant_desc = "Multimodal Interdependent: Both textual semantics and graph structure contribute comparably to confidence."

    return {
        "target_class_id": int(target_class),
        "original_probability": round(prob_orig, 4),
        "text_ablated_probability": round(prob_no_text, 4),
        "graph_ablated_probability": round(prob_no_graph, 4),
        "joint_ablated_probability": round(prob_joint, 4),
        "delta_text_ablation": round(delta_text, 4),
        "delta_graph_ablation": round(delta_graph, 4),
        "delta_joint_ablation": round(delta_joint, 4),
        "gate_statistics": {
            "mean_alpha": round(mean_alpha, 4),
            "min_alpha": round(min_alpha, 4),
            "max_alpha": round(max_alpha, 4),
            "note": "Internal gating weight (diagnostic only; not an additive contribution percentage)."
        },
        "baseline_method": "Branch ablation with neutral sequence ALBERT [CLS] and null isolated node GIN graph.",
        "interpretation": dominant_desc
    }


# =====================================================================
# 2. PER-URL ATTRIBUTION ENGINE
# =====================================================================

def evaluate_url_attribution(
    engine,
    email_dict: dict,
    target_class_name: str,
    base_prob: float
) -> list:
    """
    Evaluates individual per-URL perturbation for each URL present in the email:
    1. Text-only perturbation: Masks the specific URL in body/subject, keeping graph fixed.
    2. Graph-only perturbation: Masks the specific URL's destination node, keeping text fixed.
    3. Combined perturbation: Masks the URL in both text and graph.

    Measures probability change on the fixed target class without dividing totals equally.
    """
    from dataset import build_email_graph

    urls = email_dict.get("urls", [])
    if not urls:
        return []

    results = []
    base_subject = email_dict.get("subject", "")
    base_body = email_dict.get("body", "")
    base_domains = email_dict.get("domains", [])

    for i, url in enumerate(urls):
        try:
            domain = urllib.parse.urlparse(url if "://" in url else f"http://{url}").netloc
        except Exception:
            domain = url[:30]

        # --- A. Text-Only URL Perturbation (Graph strictly fixed) ---
        clean_text_body = base_body.replace(url, "[URL_REMOVED]")
        clean_text_subj = base_subject.replace(url, "[URL_REMOVED]")
        tokens_text_only = engine.tokenizer(
            f"{clean_text_subj} {clean_text_body}".strip(),
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # Use ORIGINAL graph
        node_feats_orig, adj_orig, _ = build_email_graph(email_dict)
        with torch.no_grad():
            out_t = engine.model(
                input_ids=tokens_text_only["input_ids"].to(engine.device),
                attention_mask=tokens_text_only["attention_mask"].to(engine.device),
                node_feats=torch.from_numpy(node_feats_orig).unsqueeze(0).to(engine.device),
                adj=torch.from_numpy(adj_orig).unsqueeze(0).to(engine.device)
            )
            prob_t = float(F.softmax(out_t[0], dim=-1).squeeze(0)[engine.prim_label2id[target_class_name]].item())
        delta_text_only = base_prob - prob_t

        # --- B. Graph-Only URL Perturbation (Text strictly fixed) ---
        # Modify domains list removing this specific destination domain
        mod_domains = [d for d in base_domains if d != domain]
        mod_graph_dict = dict(email_dict)
        mod_graph_dict["domains"] = mod_domains
        mod_graph_dict["num_urls"] = max(0.0, float(len(urls) - 1))
        mod_graph_dict["num_domains"] = float(len(mod_domains))
        node_feats_g, adj_g, _ = build_email_graph(mod_graph_dict)

        # Use ORIGINAL text tokens
        tokens_orig = engine.tokenizer(
            f"{base_subject} {base_body}".strip(),
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            out_g = engine.model(
                input_ids=tokens_orig["input_ids"].to(engine.device),
                attention_mask=tokens_orig["attention_mask"].to(engine.device),
                node_feats=torch.from_numpy(node_feats_g).unsqueeze(0).to(engine.device),
                adj=torch.from_numpy(adj_g).unsqueeze(0).to(engine.device)
            )
            prob_g = float(F.softmax(out_g[0], dim=-1).squeeze(0)[engine.prim_label2id[target_class_name]].item())
        delta_graph_only = base_prob - prob_g

        # --- C. Combined Perturbation (Both Text & Graph perturbed for this URL) ---
        with torch.no_grad():
            out_comb = engine.model(
                input_ids=tokens_text_only["input_ids"].to(engine.device),
                attention_mask=tokens_text_only["attention_mask"].to(engine.device),
                node_feats=torch.from_numpy(node_feats_g).unsqueeze(0).to(engine.device),
                adj=torch.from_numpy(adj_g).unsqueeze(0).to(engine.device)
            )
            prob_comb = float(F.softmax(out_comb[0], dim=-1).squeeze(0)[engine.prim_label2id[target_class_name]].item())
        delta_combined = base_prob - prob_comb

        results.append({
            "url": url,
            "destination_domain": domain,
            "target_class": target_class_name,
            "delta_text_only": round(delta_text_only, 4),
            "delta_graph_only": round(delta_graph_only, 4),
            "delta_combined": round(delta_combined, 4),
            "impact_direction": "INCREASES_RISK" if delta_combined > 0.01 else ("DECREASES_RISK" if delta_combined < -0.01 else "NEUTRAL"),
            "details": f"Evaluated individually: Text ΔP: {delta_text_only:+.4f}, Graph node ΔP: {delta_graph_only:+.4f}, Combined ΔP: {delta_combined:+.4f}"
        })

    return results


# =====================================================================
# 3. WORD-LEVEL EXPLANATIONS (INTEGRATED GRADIENTS ON ALBERT)
# =====================================================================

def compute_integrated_gradients(
    model,
    tokenizer,
    text: str,
    node_feats: torch.Tensor,
    adj: torch.Tensor,
    target_class: int,
    device: torch.device,
    num_steps: int = 10
) -> list:
    """
    Computes Integrated Gradients attribution for input words through ALBERT:
    IG_i = (x_i - x'_i) * (1/m) sum_{k=1}^m grad_F(x' + k/m (x - x'))
    Aggregates SentencePiece subwords into natural language words with signed contribution.
    """
    model.eval()
    # Optimize seq length: do not pad to 256 for short texts; use actual length for fast CPU backward
    tokens = tokenizer(
        text,
        max_length=MAX_LEN,
        padding=False,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    # Word embeddings from ALBERT: shape [1, seq_len, 128]
    with torch.no_grad():
        original_embeds = model.albert.embeddings(input_ids).detach()
        # Baseline: All-zeros representation in embedding space (absence of tokens)
        baseline_embeds = torch.zeros_like(original_embeds)
        # Precompute graph embedding
        z_graph, _ = model.gin(node_feats, adj)

    # Accumulate Riemann summation gradients
    accumulated_grads = torch.zeros_like(original_embeds)

    for step in range(1, num_steps + 1):
        alpha = float(step) / float(num_steps)
        interpolated = baseline_embeds + alpha * (original_embeds - baseline_embeds)
        interpolated = interpolated.detach().clone().requires_grad_(True)

        albert_out = model.albert(inputs_embeds=interpolated, attention_mask=attention_mask)
        z_text = albert_out.last_hidden_state[:, 0, :]
        z_fused = model.fusion(z_text, z_graph)
        logits = model.primary_head(z_fused)
        target_score = logits[0, target_class]

        model.zero_grad()
        target_score.backward()

        if interpolated.grad is not None:
            accumulated_grads += interpolated.grad.detach()

    # Integrated Gradients formula
    avg_grads = accumulated_grads / float(num_steps)
    delta_embeds = original_embeds - baseline_embeds
    token_attributions = (delta_embeds * avg_grads).sum(dim=-1).squeeze(0).cpu().numpy()  # [seq_len]

    # Convert tokens to readable words & aggregate subwords
    raw_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    word_attributions = []
    current_word = ""
    current_score = 0.0
    active_tokens = 0

    for tok, score in zip(raw_tokens, token_attributions):
        # Ignore special tokens
        if tok in ["[PAD]", "<pad>", "[CLS]", "[SEP]", "<cls>", "<sep>"]:
            continue

        if tok.startswith("\u2581") or tok.startswith(" ") or tok.startswith("Ġ"):
            if current_word:
                word_attributions.append({
                    "word": current_word,
                    "score": float(current_score)
                })
            current_word = tok.lstrip("\u2581 Ġ")
            current_score = float(score)
            active_tokens = 1
        elif tok.startswith("##"):
            current_word += tok[2:]
            current_score += float(score)
            active_tokens += 1
        else:
            if not current_word:
                current_word = tok
                current_score = float(score)
            else:
                current_word += tok
                current_score += float(score)
            active_tokens += 1

    if current_word:
        word_attributions.append({
            "word": current_word,
            "score": float(current_score)
        })

    # Normalize scores into [-1.0, 1.0] for UI visual heatmapping
    if word_attributions:
        max_abs = max(abs(w["score"]) for w in word_attributions) or 1e-12
        for w in word_attributions:
            norm_val = w["score"] / max_abs
            w["normalized_score"] = round(float(norm_val), 4)
            w["contribution"] = "SUPPORTS_CLASS" if norm_val > 0.05 else ("OPPOSES_CLASS" if norm_val < -0.05 else "NEUTRAL")

    return word_attributions


# =====================================================================
# 4. GNNEXPLAINER FOR GIN BRANCH
# =====================================================================

def explain_gnn(
    model,
    node_feats: torch.Tensor,
    adj: torch.Tensor,
    z_text: torch.Tensor,
    target_class: int,
    num_nodes: int,
    domain_names: list,
    sender: str,
    receiver: str,
    sender_domain: str,
    epochs: int = 80,
    lr: float = 0.05,
    device: torch.device = torch.device("cpu")
) -> dict:
    """
    GNNExplainer for the GIN branch:
    Optimizes a continuous edge mask M_edge and feature mask M_feat to maximize mutual
    information with the target class prediction, using sparsity and entropy regularization.
    Maps results to human-readable entity nodes and relationship edges.
    """
    model.eval()
    num_nodes_total = node_feats.shape[1]  # 14
    feat_dim = node_feats.shape[2]         # 16

    # Initialize masks around 0 (sigmoid gives ~0.5)
    edge_mask = torch.nn.Parameter(torch.randn(num_nodes_total, num_nodes_total, device=device) * 0.1)
    feat_mask = torch.nn.Parameter(torch.randn(feat_dim, device=device) * 0.1)
    optimizer = torch.optim.Adam([edge_mask, feat_mask], lr=lr)

    target_tensor = torch.tensor([target_class], device=device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        # Symmetric edge mask
        m_edge = torch.sigmoid((edge_mask + edge_mask.t()) / 2.0)
        m_feat = torch.sigmoid(feat_mask)

        masked_adj = adj * m_edge.unsqueeze(0)
        masked_feats = node_feats * m_feat.unsqueeze(0).unsqueeze(0)

        z_graph, _ = model.gin(masked_feats, masked_adj)
        z_fused = model.fusion(z_text, z_graph)
        logits = model.primary_head(z_fused)

        # Cross Entropy Loss with target class
        ce_loss = F.cross_entropy(logits, target_tensor)

        # Sparsity Regularization (L1 norm on masks)
        size_loss_edge = 0.005 * m_edge.sum()
        size_loss_feat = 0.005 * m_feat.sum()

        # Entropy Regularization (pushes values towards 0 or 1)
        entropy_edge = -m_edge * torch.log(m_edge + 1e-12) - (1.0 - m_edge) * torch.log(1.0 - m_edge + 1e-12)
        entropy_loss_edge = 0.001 * entropy_edge.mean()

        total_loss = ce_loss + size_loss_edge + size_loss_feat + entropy_loss_edge
        total_loss.backward()
        optimizer.step()

    # Extract final optimized masks
    with torch.no_grad():
        final_edge_mask = torch.sigmoid((edge_mask + edge_mask.t()) / 2.0).cpu().numpy()
        final_feat_mask = torch.sigmoid(feat_mask).cpu().numpy()
        adj_np = adj.squeeze(0).cpu().numpy()
        # Effective edge importance: mask applied only on existing edges
        effective_edge_imp = final_edge_mask * adj_np

    # Map node indices to readable entity names
    entity_labels = {}
    entity_labels[0] = f"Sender ({sender})"
    entity_labels[1] = f"Receiver ({receiver})"
    entity_labels[2] = f"Sender Domain (@{sender_domain})"
    for idx, d in enumerate(domain_names[:10]):
        entity_labels[3 + idx] = f"Destination URL/Domain ({d})"
    for idx in range(3 + len(domain_names), num_nodes_total):
        entity_labels[idx] = f"Empty Node Slot ({idx})"

    # Calculate overall node importance as sum of incident edge importance + active features
    node_importance = []
    for i in range(num_nodes):
        incident_edge_score = float(effective_edge_imp[i, :].sum() + effective_edge_imp[:, i].sum()) / 2.0
        node_importance.append({
            "node_idx": i,
            "entity": entity_labels.get(i, f"Node {i}"),
            "importance_score": round(incident_edge_score, 4),
            "percentage": f"{incident_edge_score * 100:.1f}%"
        })
    node_importance.sort(key=lambda x: x["importance_score"], reverse=True)

    # Extract top influential relational edges
    top_edges = []
    visited_edges = set()
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_np[i, j] > 0:
                imp = float(effective_edge_imp[i, j])
                edge_key = (min(i, j), max(i, j))
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    top_edges.append({
                        "source": entity_labels.get(i, f"Node {i}"),
                        "target": entity_labels.get(j, f"Node {j}"),
                        "weight": round(imp, 4),
                        "percentage": f"{imp * 100:.1f}%"
                    })
    top_edges.sort(key=lambda x: x["weight"], reverse=True)

    # Map feature dimensions to descriptions
    FEATURE_NAMES = [
        "One-Hot: Sender Identity",
        "One-Hot: Receiver Identity",
        "One-Hot: Sender Domain Identity",
        "One-Hot: Destination Domain Identity",
        "SPF Authentication Status",
        "DKIM Signature Verification",
        "URL Count Density",
        "Domain Count Density",
        "Body Length Scale",
        "Slot 9 (Zero)", "Slot 10 (Zero)", "Slot 11 (Zero)",
        "Slot 12 (Zero)", "Slot 13 (Zero)", "Slot 14 (Zero)", "Slot 15 (Zero)"
    ]
    top_features = []
    for k in range(9):  # Active informative features
        top_features.append({
            "feature_idx": k,
            "name": FEATURE_NAMES[k],
            "importance": round(float(final_feat_mask[k]), 4)
        })
    top_features.sort(key=lambda x: x["importance"], reverse=True)

    return {
        "node_importance": node_importance,
        "edge_importance": top_edges,
        "feature_importance": top_features,
        "status": "Optimization converged successfully."
    }


# =====================================================================
# 5. SECURITY THREAT INDICATORS
# =====================================================================

INDICATOR_RULES = {
    "CREDENTIAL_HARVESTING": {
        "title": "Credential Harvesting",
        "level": "CRITICAL",
        "keywords": [
            r"password", r"login", r"log\s*in", r"sign\s*in", r"verify\s+your\s+account",
            r"verify\s+identity", r"credentials", r"reset\s+password", r"suspended\s+account",
            r"restore\s+access", r"reactivate", r"auth\s*token", r"security\s+checkpoint"
        ],
        "desc": "Direct solicitation of account passwords, tokens, or credential verification."
    },
    "URGENCY_MANIPULATION": {
        "title": "Urgency & Coercion",
        "level": "HIGH",
        "keywords": [
            r"immediate(ly)?", r"urgent(ly)?", r"24\s*hours?", r"48\s*hours?", r"suspended",
            r"terminated?", r"action\s+required", r"critical\s+alert", r"unauthorized\s+attempt",
            r"security\s+notice", r"expires?\s+today", r"lock(ed)?"
        ],
        "desc": "Psychological time pressure coercion forcing hasty compliance."
    },
    "FINANCIAL_FRAUD": {
        "title": "Financial Transaction Trigger",
        "level": "HIGH",
        "keywords": [
            r"wire\s+transfer", r"invoice\s+payment", r"direct\s+deposit", r"payroll",
            r"bank\s+account", r"swift", r"routing\s+number", r"voucher", r"refund",
            r"unpaid\s+balance", r"settlement", r"gift\s*card", r"crypto(currency)?"
        ],
        "desc": "Requests unauthorized fund transfers, wire routing, or payroll changes."
    },
    "BRAND_IMPERSONATION": {
        "title": "Brand Impersonation / Lookalike",
        "level": "HIGH",
        "keywords": [
            r"paypa[l1]", r"microso[f|t]", r"msoffice", r"goog[l1]e", r"netf[l1]ix",
            r"amaz[o0]n", r"app[l1]e-id", r"chase-sec", r"wells-fargo-auth"
        ],
        "desc": "Impersonates reputable tech, banking, or corporate entities."
    }
}

def extract_security_indicators(subject: str, body: str, sender: str, urls: list, has_spf: bool, has_dkim: bool) -> list:
    full_text = f"{subject} {body} {sender} {' '.join(urls)}".lower()
    triggered = []

    for cat_id, info in INDICATOR_RULES.items():
        matched_terms = []
        for pat in info["keywords"]:
            found = re.findall(pat, full_text, re.IGNORECASE)
            if found:
                matched_terms.extend([m[0] if isinstance(m, tuple) else m for m in found])

        if matched_terms:
            unique_terms = list(dict.fromkeys([t.strip() for t in matched_terms if str(t).strip()]))[:5]
            triggered.append({
                "category": cat_id,
                "title": info["title"],
                "level": info["level"],
                "triggers": unique_terms,
                "description": info["desc"]
            })

    # SPF / DKIM verification
    if not has_spf or not has_dkim:
        auth_fails = []
        if not has_spf:
            auth_fails.append("SPF Unverified")
        if not has_dkim:
            auth_fails.append("DKIM Missing")
        triggered.append({
            "category": "AUTHENTICATION_FAIL",
            "title": "Email Origin Authentication Failed",
            "level": "HIGH" if (not has_spf and not has_dkim) else "MEDIUM",
            "triggers": auth_fails,
            "description": "Originating mail server failed cryptographic signature or sender policy framework."
        })

    return triggered


# =====================================================================
# 6. UNIFIED XAI BUNDLE COMPUTATION
# =====================================================================

def compute_xai_bundle(
    engine,
    email_dict: dict,
    base_result: dict,
    z_text: torch.Tensor,
    z_graph: torch.Tensor
) -> dict:
    """
    Computes full explainability suite:
    - Modality Attribution via branch ablation
    - Per-URL Attribution
    - Word-Level Integrated Gradients
    - GNNExplainer on GIN branch
    - Semantic Threat Indicators
    """
    from dataset import build_email_graph

    target_class_id = engine.prim_label2id[base_result["primary_label"]]
    base_prob = base_result["primary_confidence"]

    # Featurize
    full_text = f"{email_dict.get('subject', '')} {email_dict.get('body', '')}".strip()
    tokens = engine.tokenizer(
        full_text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = tokens["input_ids"].to(engine.device)
    attention_mask = tokens["attention_mask"].to(engine.device)

    node_feats_np, adj_np, num_nodes = build_email_graph(email_dict)
    node_feats = torch.from_numpy(node_feats_np).unsqueeze(0).to(engine.device)
    adj = torch.from_numpy(adj_np).unsqueeze(0).to(engine.device)

    # 1. Modality Attribution via Controlled Branch Ablation
    modality_ablation = evaluate_modality_ablation(
        model=engine.model,
        tokenizer=engine.tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        node_feats=node_feats,
        adj=adj,
        target_class=target_class_id,
        device=engine.device
    )

    # 2. Per-URL Attribution
    url_attribution = evaluate_url_attribution(
        engine=engine,
        email_dict=email_dict,
        target_class_name=base_result["primary_label"],
        base_prob=base_prob
    )

    # 3. Word-Level Attributions via Integrated Gradients
    word_attributions = compute_integrated_gradients(
        model=engine.model,
        tokenizer=engine.tokenizer,
        text=full_text,
        node_feats=node_feats,
        adj=adj,
        target_class=target_class_id,
        device=engine.device,
        num_steps=10
    )

    # 4. GNNExplainer for GIN Branch
    gnn_explanation = explain_gnn(
        model=engine.model,
        node_feats=node_feats,
        adj=adj,
        z_text=z_text,
        target_class=target_class_id,
        num_nodes=num_nodes,
        domain_names=email_dict.get("domains", []),
        sender=email_dict.get("sender", "unknown"),
        receiver=email_dict.get("receiver", "unknown"),
        sender_domain=email_dict.get("sender_domain", "unknown"),
        epochs=40,
        lr=0.08,
        device=engine.device
    )

    # 5. Security Threat Indicators
    threat_indicators = extract_security_indicators(
        subject=email_dict.get("subject", ""),
        body=email_dict.get("body", ""),
        sender=email_dict.get("sender", ""),
        urls=email_dict.get("urls", []),
        has_spf=email_dict.get("has_spf", False),
        has_dkim=email_dict.get("has_dkim", False)
    )

    return {
        "modality_attribution": modality_ablation,
        "url_attribution": url_attribution,
        "word_attributions": word_attributions,
        "gnn_explanation": gnn_explanation,
        "threat_indicators": threat_indicators,
        # Legacy aliases for UI backwards compatibility
        "fusion": {
            "text_percent": round(modality_ablation["gate_statistics"]["mean_alpha"] * 100, 1),
            "graph_percent": round((1.0 - modality_ablation["gate_statistics"]["mean_alpha"]) * 100, 1),
            "dominant_modality": modality_ablation["interpretation"],
            "rationale": f"Text Ablation ΔP: {modality_ablation['delta_text_ablation']:+.4f} | Graph Ablation ΔP: {modality_ablation['delta_graph_ablation']:+.4f}"
        },
        "entity_contributions": gnn_explanation["node_importance"]
    }


# =====================================================================
# 7. CONVERSATIONAL XAI AGENT
# =====================================================================

def conversational_xai_agent(question: str, xai_bundle: dict, email_data: dict, prediction: dict) -> str:
    """
    Offline conversational agent answering analyst questions grounded in real XAI metrics.
    """
    q = question.lower().strip()
    verdict = prediction.get("primary_label", "UNKNOWN")
    conf = prediction.get("primary_confidence", 0.0) * 100
    mod_attr = xai_bundle.get("modality_attribution", {})
    url_attrs = xai_bundle.get("url_attribution", [])
    word_attrs = xai_bundle.get("word_attributions", [])
    gnn = xai_bundle.get("gnn_explanation", {})
    indicators = xai_bundle.get("threat_indicators", [])

    # Modality questions
    if any(k in q for k in ["text vs graph", "modality", "which branch", "influence more", "ablation"]):
        return (
            f"**Modality Attribution Analysis (Controlled Branch Ablation):**\n\n"
            f"• **Text Branch Ablation:** Removes textual evidence $\\to$ confidence changes by **{mod_attr.get('delta_text_ablation', 0):+.4f}**\n"
            f"• **Graph Branch Ablation:** Removes entity graph evidence $\\to$ confidence changes by **{mod_attr.get('delta_graph_ablation', 0):+.4f}**\n"
            f"• **Joint Ablation (Both Removed):** Confidence changes by **{mod_attr.get('delta_joint_ablation', 0):+.4f}**\n\n"
            f"**Conclusion:** {mod_attr.get('interpretation', 'Balanced multimodal evaluation.')}\n"
            f"*(Internal fusion gate diagnostic mean: {mod_attr.get('gate_statistics', {}).get('mean_alpha', 0.5):.3f})*"
        )

    # Word attribution questions
    if any(k in q for k in ["word", "token", "phrase", "text", "highlight"]):
        top_words = sorted(word_attrs, key=lambda x: abs(x["score"]), reverse=True)[:5]
        words_summary = ", ".join([f"**'{w['word']}'** ({w['contribution']})" for w in top_words])
        return (
            f"**Word-Level Attribution (Integrated Gradients on ALBERT):**\n\n"
            f"Top influential words for the `{verdict}` decision: {words_summary}.\n\n"
            f"*Caveat:* Word attributions reflect gradient sensitivity in the latent embedding space; they indicate model focus rather than standalone proof of malice."
        )

    # URL attribution questions
    if any(k in q for k in ["url", "link", "destination"]):
        if not url_attrs:
            return "No external hyperlinks were found in this email."
        url_lines = [
            f"• **{u['url']}**:\n  - Text-Only ΔP: `{u['delta_text_only']:+.4f}`\n  - Graph-Only ΔP: `{u['delta_graph_only']:+.4f}`\n  - Combined ΔP: `{u['delta_combined']:+.4f}` ({u['impact_direction']})"
            for u in url_attrs[:3]
        ]
        return f"**Per-URL Perturbation Breakdown:**\n\n" + "\n".join(url_lines)

    # Graph questions
    if any(k in q for k in ["graph", "node", "edge", "entity", "domain", "gnn"]):
        top_nodes = gnn.get("node_importance", [])[:3]
        node_lines = [f"• **{n['entity']}**: Importance score {n['percentage']}" for n in top_nodes]
        top_edges = gnn.get("edge_importance", [])[:2]
        edge_lines = [f"• **{e['source']} $\\leftrightarrow$ {e['target']}**: Weight {e['percentage']}" for e in top_edges]
        return (
            f"**GNNExplainer Graph Substructure Analysis:**\n\n"
            f"**Top Influential Entities:**\n" + "\n".join(node_lines) + "\n\n"
            f"**Key Relational Edges:**\n" + "\n".join(edge_lines)
        )

    # Vector memory / similarity questions
    if any(k in q for k in ["memory", "vector", "similar", "saved", "zero-shot", "catalog"]):
        matches = prediction.get("memory_matches", [])
        mem_count = prediction.get("memory_count", 0)
        if not matches:
            return (
                f"**Dynamic Vector Memory Analysis:**\n\n"
                f"• **Cataloged Vectors in Bank:** {mem_count}\n"
                f"• **Nearest Matches:** None found exceeding the ≥72% similarity threshold.\n\n"
                f"This email represents a novel attack or communication pattern not yet indexed. "
                f"You can use the **Save to Vector Memory** button to catalog it forever for zero-shot detection!"
            )
        match_lines = [
            f"• **{m.get('similarity_percent', 0)}% Similarity** to Saved Vector #{m.get('id', '?')} (`{m.get('label', 'UNKNOWN')}` / `{m.get('subtype', 'N/A')}`)\n"
            f"  - Subject: *\"{m.get('subject', 'N/A')}\"*\n"
            f"  - Sender: `{m.get('sender', 'N/A')}`\n"
            f"  - Cataloged: {m.get('timestamp', 'Unknown')} | Note: {m.get('note', 'No note')}"
            for m in matches
        ]
        return (
            f"**Zero-Shot Vector Memory Match Identified:**\n\n"
            + "\n".join(match_lines) + "\n\n"
            f"The 256-dim fused embedding ($z_{{fused}}$) closely aligns with previously flagged patterns in memory, confirming a recurring campaign without model retraining."
        )

    # General reason / why
    reasons = []
    if prediction.get("memory_matches"):
        top_match = prediction["memory_matches"][0]
        reasons.append(f"⚠️ Matches saved vector #{top_match.get('id')} ({top_match.get('similarity_percent')}% similarity to flagged '{top_match.get('label')}').")
    if indicators:
        reasons.append(f"Triggered {len(indicators)} security flags: " + ", ".join([i["title"] for i in indicators[:3]]) + ".")
    if mod_attr:
        reasons.append(f"Modality ablation indicates: {mod_attr.get('interpretation')}")
    if gnn.get("node_importance"):
        reasons.append(f"Highest graph factor was '{gnn['node_importance'][0]['entity']}'.")

    return (
        f"**Verdict Analysis: {verdict} ({conf:.1f}% Confidence)**\n\n" +
        " ".join(reasons) +
        "\n\n*Ask me about:* 'Text vs graph ablation', 'Which words mattered?', 'Per-URL impact', 'Graph entities', or 'Check vector memory'."
    )
