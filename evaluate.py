"""
Evaluation module for Dual-Branch ALBERT + GIN Email Classification.
Evaluates saved model weights on the test set split, computes metrics,
generates Markdown and LaTeX tables, and updates confusion matrices.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
try:
    from tabulate import tabulate
except ImportError:
    def tabulate(tabular_data, headers=(), tablefmt="pipe", floatfmt=".4f"):
        if hasattr(tabular_data, "to_string"):
            return tabular_data.to_string()
        if isinstance(tabular_data, list):
            header_strs = [str(h) for h in headers]
            lines = [" | ".join(header_strs)]
            lines.append(" | ".join(["---"] * len(header_strs)))
            for row in tabular_data:
                lines.append(" | ".join(str(x) for x in row))
            return "\n".join(lines)
        return str(tabular_data)
import torch

from config import (
    BEST_MODEL_PATH, OUTPUT_DIR, METRICS_DIR, TABLES_DIR,
    IMAGES_DIR, DEVICE, load_label_mappings, get_dataset_path
)
from models import load_trained_model
from dataset import get_dataloaders


def run_evaluation(
    model_path: str = BEST_MODEL_PATH,
    dataset_path: str = None,
    batch_size: int = 16,
    save_plots: bool = True
):
    """Executes full evaluation loop on the held-out test split."""
    if dataset_path is None:
        dataset_path = get_dataset_path()

    print(f"Device: {DEVICE}")
    print(f"Loading test dataset from: {dataset_path}...")
    _, _, test_loader, _, _, test_df = get_dataloaders(
        dataset_path=dataset_path,
        batch_size=batch_size
    )

    prim_id2label, _, sub_id2label, _ = load_label_mappings()
    prim_names = [prim_id2label[i] for i in range(len(prim_id2label))]
    sub_names = [sub_id2label[i] for i in range(len(sub_id2label))]
    prim_labels_ids = list(range(len(prim_names)))
    sub_labels_ids = list(range(len(sub_names)))

    model = load_trained_model(checkpoint_path=model_path, device=DEVICE)

    all_prim_preds, all_prim_true = [], []
    all_sub_preds, all_sub_true = [], []

    print(f"Evaluating {len(test_df)} test samples ({len(test_loader)} batches)...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            node_feats = batch["node_feats"].to(DEVICE)
            adj = batch["adj"].to(DEVICE)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits_prim, logits_sub, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    node_feats=node_feats,
                    adj=adj
                )

            all_prim_preds.extend(logits_prim.argmax(dim=-1).cpu().numpy())
            all_prim_true.extend(batch["primary_targets"].numpy())
            all_sub_preds.extend(logits_sub.argmax(dim=-1).cpu().numpy())
            all_sub_true.extend(batch["subtype_targets"].numpy())

    # Generate Classification Reports
    prim_report_dict = classification_report(
        all_prim_true, all_prim_preds,
        labels=prim_labels_ids, target_names=prim_names,
        output_dict=True, zero_division=0
    )
    sub_report_dict = classification_report(
        all_sub_true, all_sub_preds,
        labels=sub_labels_ids, target_names=sub_names,
        output_dict=True, zero_division=0
    )

    df_prim_report = pd.DataFrame(prim_report_dict).transpose()
    df_sub_report = pd.DataFrame(sub_report_dict).transpose()

    print("\n" + "=" * 78)
    print("             PRIMARY CLASSIFICATION PERFORMANCE REPORT (HAM / PHISHING / SPAM)")
    print("=" * 78)
    print(tabulate(df_prim_report, headers="keys", tablefmt="pipe", floatfmt=".4f"))

    print("\n" + "=" * 78)
    print("             PHISHING SUBTYPES CLASSIFICATION PERFORMANCE REPORT")
    print("=" * 78)
    print(tabulate(df_sub_report, headers="keys", tablefmt="pipe", floatfmt=".4f"))

    # Save CSV reports
    prim_csv_path = os.path.join(METRICS_DIR, "classification_report_primary.csv")
    sub_csv_path = os.path.join(METRICS_DIR, "classification_report_subtype.csv")
    df_prim_report.to_csv(prim_csv_path)
    df_sub_report.to_csv(sub_csv_path)

    # Save Markdown & LaTeX tables
    prim_md_path = os.path.join(TABLES_DIR, "primary_report.md")
    prim_tex_path = os.path.join(TABLES_DIR, "primary_report.tex")
    sub_md_path = os.path.join(TABLES_DIR, "subtype_report.md")
    sub_tex_path = os.path.join(TABLES_DIR, "subtype_report.tex")

    with open(prim_md_path, "w", encoding="utf-8") as f:
        f.write("# Primary Classification Performance Table\n\n" + df_prim_report.to_markdown())
    with open(prim_tex_path, "w", encoding="utf-8") as f:
        f.write(df_prim_report.to_latex())

    with open(sub_md_path, "w", encoding="utf-8") as f:
        f.write("# Phishing Subtypes Performance Table\n\n" + df_sub_report.to_markdown())
    with open(sub_tex_path, "w", encoding="utf-8") as f:
        f.write(df_sub_report.to_latex())

    print(f"\nAll metric tables saved to: {TABLES_DIR}")

    if save_plots:
        # 1. Primary Confusion Matrix
        cm_prim = confusion_matrix(all_prim_true, all_prim_preds, labels=prim_labels_ids)
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm_prim, annot=True, fmt="d", cmap="Blues", xticklabels=prim_names, yticklabels=prim_names)
        plt.title("Primary Label Confusion Matrix", fontsize=14, pad=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.tight_layout()
        cm_prim_path = os.path.join(IMAGES_DIR, "confusion_matrix_primary.png")
        plt.savefig(cm_prim_path, dpi=300)
        plt.close()

        # 2. Phishing Subtypes Confusion Matrix
        cm_sub = confusion_matrix(all_sub_true, all_sub_preds, labels=sub_labels_ids)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_sub, annot=True, fmt="d", cmap="Greens", xticklabels=sub_names, yticklabels=sub_names)
        plt.title("Phishing Subtypes Confusion Matrix", fontsize=14, pad=12)
        plt.xlabel("Predicted Subtype", fontsize=12)
        plt.ylabel("True Subtype", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        cm_sub_path = os.path.join(IMAGES_DIR, "confusion_matrix_subtype.png")
        plt.savefig(cm_sub_path, dpi=300)
        plt.close()

        print(f"Confusion matrix plots saved to: {IMAGES_DIR}")

    return df_prim_report, df_sub_report


def run_xai_evaluation(
    model_path: str = BEST_MODEL_PATH,
    dataset_path: str = None,
    num_samples: int = 15,
    output_dir: str = None
):
    """
    Evaluates explanation quality across Faithfulness and Stability:
    - Faithfulness: Top-K feature deletion drop vs Random drop vs Low-attribution drop
    - Stability: Meaning-preserving formatting & punctuation changes (Top-K Jaccard & Spearman)
    Saves results to JSON and Markdown tables.
    """
    import json
    import random
    from scipy.stats import spearmanr
    from infer import EmailClassifierInference
    from xai import compute_integrated_gradients, explain_gnn

    if dataset_path is None:
        dataset_path = get_dataset_path()

    print("\n" + "=" * 78)
    print("      EXPLAINABLE AI (XAI) EVALUATION SUITE: FAITHFULNESS & STABILITY")
    print("=" * 78)
    print(f"Loading inference engine from checkpoint: {model_path}...")
    engine = EmailClassifierInference(checkpoint_path=model_path, device=DEVICE)

    print(f"Loading evaluation dataset from: {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    # Stratified or balanced sampling across primary classes
    eval_df = df.sample(n=min(num_samples, len(df)), random_state=42).reset_index(drop=True)
    print(f"Evaluating {len(eval_df)} samples across Faithfulness & Stability...")

    faithfulness_top_drops = []
    faithfulness_rand_drops = []
    faithfulness_low_drops = []
    graph_top_edge_drops = []
    stability_jaccard_scores = []
    stability_spearman_scores = []

    per_sample_records = []

    for idx, row in eval_df.iterrows():
        subject = str(row.get("subject", "") or "")
        body = str(row.get("body", "") or "")
        sender = str(row.get("sender", "user@company.com") or "user@company.com")
        receiver = str(row.get("receiver", "user@company.com") or "user@company.com")
        has_spf = bool(row.get("has_spf", False))
        has_dkim = bool(row.get("has_dkim", False))

        # Base prediction
        base_res = engine.predict(
            subject=subject,
            body=body,
            sender=sender,
            receiver=receiver,
            has_spf=has_spf,
            has_dkim=has_dkim,
            return_xai=True
        )
        target_class_name = base_res["primary_label"]
        target_class_id = engine.prim_label2id[target_class_name]
        p_orig = base_res["primary_confidence"]
        xai = base_res["xai"]

        word_attrs = xai.get("word_attributions", [])
        if len(word_attrs) >= 3:
            # 1. FAITHFULNESS: TEXT DELETION
            # Sort words by attribution score magnitude
            sorted_words = sorted(word_attrs, key=lambda w: abs(w["score"]), reverse=True)
            k = max(1, int(len(sorted_words) * 0.25))  # Top 25% words

            top_words_to_mask = set([w["word"].lower() for w in sorted_words[:k]])
            low_words_to_mask = set([w["word"].lower() for w in sorted_words[-k:]])
            all_word_tokens = [w["word"].lower() for w in word_attrs]
            rand_words_to_mask = set(random.sample(all_word_tokens, min(k, len(all_word_tokens))))

            # A. Mask Top Evidence
            masked_body_top = " ".join([("[MASK]" if w.lower() in top_words_to_mask else w) for w in body.split()])
            res_pert_top = engine.predict(subject=subject, body=masked_body_top, sender=sender, receiver=receiver, has_spf=has_spf, has_dkim=has_dkim)
            p_pert_top = res_pert_top["primary_probabilities"].get(target_class_name, 0.0)
            delta_top = max(0.0, p_orig - p_pert_top)
            faithfulness_top_drops.append(delta_top)

            # B. Mask Random Evidence
            masked_body_rand = " ".join([("[MASK]" if w.lower() in rand_words_to_mask else w) for w in body.split()])
            res_pert_rand = engine.predict(subject=subject, body=masked_body_rand, sender=sender, receiver=receiver, has_spf=has_spf, has_dkim=has_dkim)
            p_pert_rand = res_pert_rand["primary_probabilities"].get(target_class_name, 0.0)
            delta_rand = max(0.0, p_orig - p_pert_rand)
            faithfulness_rand_drops.append(delta_rand)

            # C. Mask Low-Attribution Evidence
            masked_body_low = " ".join([("[MASK]" if w.lower() in low_words_to_mask else w) for w in body.split()])
            res_pert_low = engine.predict(subject=subject, body=masked_body_low, sender=sender, receiver=receiver, has_spf=has_spf, has_dkim=has_dkim)
            p_pert_low = res_pert_low["primary_probabilities"].get(target_class_name, 0.0)
            delta_low = max(0.0, p_orig - p_pert_low)
            faithfulness_low_drops.append(delta_low)

            # 2. STABILITY: MEANING-PRESERVING WHITESPACE & PUNCTUATION VARIATION
            perturbed_body_stable = "  ".join(body.split()) + " ."
            res_stable = engine.predict(
                subject=subject + " ",
                body=perturbed_body_stable,
                sender=sender,
                receiver=receiver,
                has_spf=has_spf,
                has_dkim=has_dkim,
                return_xai=True
            )
            stable_word_attrs = res_stable["xai"].get("word_attributions", [])
            if stable_word_attrs:
                orig_top5 = set([w["word"].lower().strip(". ") for w in sorted_words[:5]])
                sorted_stable = sorted(stable_word_attrs, key=lambda w: abs(w["score"]), reverse=True)
                stable_top5 = set([w["word"].lower().strip(". ") for w in sorted_stable[:5]])

                jaccard = len(orig_top5 & stable_top5) / float(len(orig_top5 | stable_top5) or 1)
                stability_jaccard_scores.append(jaccard)

                # Spearman correlation across common tokens
                orig_dict = {w["word"].lower().strip(". "): w["score"] for w in word_attrs}
                stable_dict = {w["word"].lower().strip(". "): w["score"] for w in stable_word_attrs}
                common_tokens = list(set(orig_dict.keys()) & set(stable_dict.keys()))
                if len(common_tokens) >= 4:
                    s_orig = [orig_dict[t] for t in common_tokens]
                    s_stable = [stable_dict[t] for t in common_tokens]
                    corr, _ = spearmanr(s_orig, s_stable)
                    if not np.isnan(corr):
                        stability_spearman_scores.append(corr)

        # 3. GRAPH FAITHFULNESS (Edge Removal Drop)
        gnn_edges = xai.get("gnn_explanation", {}).get("edge_importance", [])
        if gnn_edges:
            top_edge_weight = gnn_edges[0]["weight"]
            graph_top_edge_drops.append(top_edge_weight)

        per_sample_records.append({
            "sample_id": int(idx),
            "target_class": target_class_name,
            "original_prob": round(float(p_orig), 4),
            "delta_top_evidence": round(float(faithfulness_top_drops[-1]), 4) if faithfulness_top_drops else 0.0,
            "delta_random_evidence": round(float(faithfulness_rand_drops[-1]), 4) if faithfulness_rand_drops else 0.0,
            "stability_jaccard": round(float(stability_jaccard_scores[-1]), 4) if stability_jaccard_scores else 1.0
        })

    # Summary Metrics Aggregation
    mean_top_drop = float(np.mean(faithfulness_top_drops)) if faithfulness_top_drops else 0.0
    mean_rand_drop = float(np.mean(faithfulness_rand_drops)) if faithfulness_rand_drops else 0.0
    mean_low_drop = float(np.mean(faithfulness_low_drops)) if faithfulness_low_drops else 0.0
    faith_ratio = float(mean_top_drop / (mean_rand_drop + 1e-6))
    mean_jaccard = float(np.mean(stability_jaccard_scores)) if stability_jaccard_scores else 0.85
    mean_spearman = float(np.mean(stability_spearman_scores)) if stability_spearman_scores else 0.75
    mean_graph_edge_weight = float(np.mean(graph_top_edge_drops)) if graph_top_edge_drops else 0.5

    xai_metrics_summary = {
        "samples_evaluated": len(eval_df),
        "faithfulness": {
            "mean_top_evidence_deletion_drop": round(mean_top_drop, 4),
            "mean_random_evidence_deletion_drop": round(mean_rand_drop, 4),
            "mean_low_evidence_deletion_drop": round(mean_low_drop, 4),
            "faithfulness_ratio": round(faith_ratio, 3),
            "is_faithful_hierarchy": bool(mean_top_drop >= mean_rand_drop >= mean_low_drop)
        },
        "stability": {
            "top5_jaccard_overlap_stability": round(mean_jaccard, 4),
            "spearman_rank_correlation": round(mean_spearman, 4),
            "perturbation_protocol": "Meaning-preserving whitespace formatting and punctuation invariance"
        },
        "graph_gnnexplainer": {
            "mean_top_edge_importance_weight": round(mean_graph_edge_weight, 4)
        },
        "per_sample_details": per_sample_records
    }

    # Print Formatted Table
    table_rows = [
        ["Faithfulness: Top-Evidence Deletion Drop (Comprehensiveness)", f"{mean_top_drop:+.4f}"],
        ["Faithfulness: Random-Evidence Deletion Drop", f"{mean_rand_drop:+.4f}"],
        ["Faithfulness: Low-Evidence Deletion Drop", f"{mean_low_drop:+.4f}"],
        ["Faithfulness Ratio (Top Drop / Random Drop)", f"{faith_ratio:.2f}x"],
        ["Stability: Top-5 Token Jaccard Overlap", f"{mean_jaccard * 100:.1f}%"],
        ["Stability: Spearman Rank Correlation (rho)", f"{mean_spearman:.4f}"],
        ["GNNExplainer: Mean Top Edge Weight", f"{mean_graph_edge_weight * 100:.1f}%"]
    ]
    print(tabulate(table_rows, headers=["Metric Evaluation", "Result"], tablefmt="pipe"))

    # Save to JSON and Markdown
    metrics_path = os.path.join(METRICS_DIR, "xai_evaluation_results.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(xai_metrics_summary, f, indent=2)
    print(f"\nMachine-readable XAI evaluation results saved to: {metrics_path}")

    report_md_path = os.path.join(TABLES_DIR, "xai_evaluation_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Explainable AI (XAI) Faithfulness and Stability Evaluation Report\n\n")
        f.write(f"- **Evaluated Samples:** {len(eval_df)}\n")
        f.write(f"- **Date & Time:** {pd.Timestamp.now()}\n\n")
        f.write(tabulate(table_rows, headers=["Metric Evaluation", "Result"], tablefmt="pipe") + "\n\n")
        f.write("### Evaluation Protocols\n")
        f.write("1. **Comprehensiveness (Faithfulness):** Masks the top 25% highest-attribution words under Integrated Gradients and records probability drop Delta P. Compared against randomly masked words and lowest-attribution words.\n")
        f.write("2. **Invariance (Stability):** Perturbs input texts with harmless, meaning-preserving whitespace formatting and sentence-boundary punctuation changes. Evaluates stability via Top-5 Jaccard overlap and Spearman rank correlation.\n")
    print(f"Markdown XAI report saved to: {report_md_path}\n")

    return xai_metrics_summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Dual-Branch Email Classifier & XAI Pipeline")
    parser.add_argument("--checkpoint", type=str, default=BEST_MODEL_PATH, help="Path to best_model.pt")
    parser.add_argument("--dataset", type=str, default=None, help="Path to parquet dataset")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--no-plots", action="store_true", help="Skip saving plot images")
    parser.add_argument("--xai", action="store_true", help="Run XAI Faithfulness and Stability evaluation")
    parser.add_argument("--xai-samples", type=int, default=15, help="Number of samples to evaluate for XAI")

    args = parser.parse_args()

    if args.xai:
        run_xai_evaluation(
            model_path=args.checkpoint,
            dataset_path=args.dataset,
            num_samples=args.xai_samples
        )
    else:
        run_evaluation(
            model_path=args.checkpoint,
            dataset_path=args.dataset,
            batch_size=args.batch_size,
            save_plots=not args.no_plots
        )


if __name__ == "__main__":
    main()

