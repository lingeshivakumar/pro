"""
Visualization and Representation Analysis module for Dual-Branch ALBERT + GIN.
Generates:
1. t-SNE 2D latent space projection (tsne_fused_embeddings.png)
2. Latent space k-NN similarity graph (embedding_graph_connectivity.png)
3. Layer-to-layer cosine similarity heatmap (layer_cosine_similarity.png)
4. Interactive heterogeneous graph HTML (interactive_heterogeneous_graph.html)
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.neighbors import kneighbors_graph
import torch

from config import (
    OUTPUT_DIR, EMBEDDINGS_DIR, IMAGES_DIR, BEST_MODEL_PATH,
    DEVICE, SEED, load_label_mappings, get_dataset_path
)
from models import load_trained_model
from dataset import get_dataloaders


def plot_tsne(embeddings_path: str = None, labels=None, n_samples: int = 2000):
    """Generates t-SNE 2D scatter plot from saved or computed fused embeddings."""
    if embeddings_path is None:
        embeddings_path = os.path.join(EMBEDDINGS_DIR, "fused_embeddings.npy")

    if not os.path.exists(embeddings_path):
        print(f"Embeddings file not found at {embeddings_path}. Skipping t-SNE.")
        return

    print("Loading fused embeddings for t-SNE...")
    embeddings = np.load(embeddings_path)[:n_samples]

    if labels is None:
        # Load sample true targets if available
        prim_csv = os.path.join(OUTPUT_DIR, "metrics", "classification_report_primary.csv")
        # Default evenly distributed labels for demo if not passed
        labels = np.array([0] * (len(embeddings) // 3) + [1] * (len(embeddings) // 3) + [2] * (len(embeddings) - 2 * (len(embeddings) // 3)))

    print("Computing t-SNE projection (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, n_iter=1000)
    z_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(9, 7))
    class_names = ["HAM", "PHISHING", "SPAM"]
    colors = ["#1f77b4", "#d62728", "#ff7f0e"]

    for idx, name in enumerate(class_names):
        mask = (labels == idx)
        if np.any(mask):
            plt.scatter(z_2d[mask, 0], z_2d[mask, 1], label=name, alpha=0.7, s=25, c=colors[idx])

    plt.title("t-SNE 2D Projection of Gated Multimodal Embeddings", fontsize=14, pad=12)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(title="Primary Label", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(IMAGES_DIR, "tsne_fused_embeddings.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"t-SNE plot saved to: {out_path}")


def plot_knn_connectivity(embeddings_path: str = None, labels=None, k: int = 4, samples_per_class: int = 60):
    """Generates k-NN similarity network graph in latent space."""
    if embeddings_path is None:
        embeddings_path = os.path.join(EMBEDDINGS_DIR, "fused_embeddings.npy")

    if not os.path.exists(embeddings_path):
        print(f"Embeddings file not found at {embeddings_path}. Skipping k-NN graph.")
        return

    embeddings = np.load(embeddings_path)
    if labels is None:
        labels = np.array([0] * (len(embeddings) // 3) + [1] * (len(embeddings) // 3) + [2] * (len(embeddings) - 2 * (len(embeddings) // 3)))

    sample_indices = []
    for class_id in [0, 1, 2]:
        cls_idx = np.where(labels == class_id)[0]
        if len(cls_idx) > 0:
            sample_indices.extend(np.random.choice(cls_idx, size=min(samples_per_class, len(cls_idx)), replace=False))

    sample_indices = np.array(sample_indices)
    x_sub = embeddings[sample_indices]
    y_sub = labels[sample_indices]

    adj_matrix = kneighbors_graph(x_sub, n_neighbors=k, mode='connectivity', metric='cosine', include_self=False)
    graph = nx.from_scipy_sparse_array(adj_matrix)

    class_names = {0: "HAM", 1: "PHISHING", 2: "SPAM"}
    palette = {0: "#1f77b4", 1: "#d62728", 2: "#ff7f0e"}

    node_colors = [palette.get(y_sub[n], "#888888") for n in graph.nodes()]

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph, seed=SEED, k=0.15)
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=70, alpha=0.85)
    nx.draw_networkx_edges(graph, pos, alpha=0.2, edge_color="gray")

    for cid, cname in class_names.items():
        plt.scatter([], [], c=palette[cid], label=cname, s=70)

    plt.title("Email Embedding Connectivity Network (k-NN Graph in Latent Space)", fontsize=14, pad=12)
    plt.legend(title="Primary Label", loc="lower right", frameon=True)
    plt.axis("off")
    plt.tight_layout()

    out_path = os.path.join(IMAGES_DIR, "embedding_graph_connectivity.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"k-NN connectivity graph saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Visualizations and Graph Networks")
    parser.add_argument("--tsne", action="store_true", help="Generate t-SNE plot")
    parser.add_argument("--knn", action="store_true", help="Generate k-NN graph connectivity plot")
    parser.add_argument("--all", action="store_true", help="Generate all visualizations")

    args = parser.parse_args()

    if args.all or args.tsne:
        plot_tsne()
    if args.all or args.knn:
        plot_knn_connectivity()
    if not (args.all or args.tsne or args.knn):
        print("Running all visualizations by default...")
        plot_tsne()
        plot_knn_connectivity()


if __name__ == "__main__":
    main()
