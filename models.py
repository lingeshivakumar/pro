"""
Model architecture module for Dual-Branch ALBERT + GIN Email Classification.
Includes:
- GINLayer: Batched Graph Isomorphism Network convolution
- GINBranch: 3-layer GIN with Jumping Knowledge (JK) readout
- GatedMultimodalFusion: Trainable dynamic gating between textual & graph embeddings
- AlbertGINClassifier: Unified multi-task dual-branch classification network
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from config import MODEL_NAME, BEST_MODEL_PATH, DEVICE, load_label_mappings


class GINLayer(nn.Module):
    """
    Batched Graph Isomorphism Network (GIN) Convolution Layer with 2-layer MLP.
    Supports batched adjacency matrices of shape [B, N, N].
    """
    def __init__(self, in_dim: int, out_dim: int, eps: float = 0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.Tensor([eps]))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D], adj: [B, N, N]
        # Permutation-invariant sum aggregation via batched matrix multiply
        agg = torch.bmm(adj, x)
        out = (1.0 + self.eps) * x + agg
        b, n, d = out.shape
        out_flat = self.mlp(out.view(b * n, d))
        return out_flat.view(b, n, -1)


class GINBranch(nn.Module):
    """
    3-Layer Batched GIN with Jumping Knowledge (JK) multi-layer representations.
    Global graph readout concatenates Mean-pooling and Max-pooling across nodes.
    """
    def __init__(self, in_dim: int = 16, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GINLayer(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GINLayer(hidden_dim, hidden_dim))
        self.out_dim = hidden_dim * 2  # Mean pool (64) + Max pool (64) = 128

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        layer_reps = [x]
        curr = x
        for layer in self.layers:
            curr = layer(curr, adj)
            layer_reps.append(curr)

        # Global Graph Readout (Mean + Max pooling per graph across nodes dim 1)
        mean_pool = curr.mean(dim=1)
        max_pool = curr.max(dim=1)[0]
        graph_embed = torch.cat([mean_pool, max_pool], dim=-1)
        return graph_embed, layer_reps


class GatedMultimodalFusion(nn.Module):
    """
    Learned Gated Multimodal Fusion combining text and graph embeddings.
    Dynamically weights linguistic and graph structure features.
    """
    def __init__(self, text_dim: int = 1024, graph_dim: int = 128, fused_dim: int = 256):
        super().__init__()
        self.proj_text = nn.Linear(text_dim, fused_dim)
        self.proj_graph = nn.Linear(graph_dim, fused_dim)
        self.gate = nn.Sequential(
            nn.Linear(text_dim + graph_dim, fused_dim),
            nn.Sigmoid()
        )
        self.out_proj = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, z_text: torch.Tensor, z_graph: torch.Tensor) -> torch.Tensor:
        h_t = self.proj_text(z_text)
        h_g = self.proj_graph(z_graph)
        alpha = self.gate(torch.cat([z_text, z_graph], dim=-1))
        fused = alpha * h_t + (1.0 - alpha) * h_g
        return self.out_proj(fused)


class AlbertGINClassifier(nn.Module):
    """
    Dual-Branch Architecture:
    - Pretrained ALBERT transformer for deep textual semantics
    - GIN for email entity relationship topology
    - Gated fusion layer
    - Dual multi-task classification heads (Primary & Subtype)
    """
    def __init__(
        self,
        albert_model_name: str = MODEL_NAME,
        num_primary: int = 3,
        num_subtypes: int = 10,
        fused_dim: int = 256
    ):
        super().__init__()
        self.albert = AutoModel.from_pretrained(albert_model_name)
        albert_dim = self.albert.config.hidden_size  # 1024 for albert-large-v2
        self.gin = GINBranch(in_dim=16, hidden_dim=64, num_layers=3)
        self.fusion = GatedMultimodalFusion(text_dim=albert_dim, graph_dim=128, fused_dim=fused_dim)

        # Multi-task Classification Heads
        self.primary_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_primary)
        )
        self.subtype_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_subtypes)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        node_feats: torch.Tensor,
        adj: torch.Tensor,
        return_layers: bool = False
    ):
        # 1. ALBERT Forward Pass
        albert_out = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=return_layers
        )
        z_text = albert_out.last_hidden_state[:, 0, :]  # [CLS] representation [B, 1024]

        # 2. Batched GIN Forward Pass
        z_graph, gin_layer_reps = self.gin(node_feats, adj)  # [B, 128]

        # 3. Gated Fusion
        z_fused = self.fusion(z_text, z_graph)  # [B, 256]

        # 4. Multi-Task Predictions
        logits_primary = self.primary_head(z_fused)
        logits_subtype = self.subtype_head(z_fused)

        if return_layers:
            return {
                "logits_primary": logits_primary,
                "logits_subtype": logits_subtype,
                "z_text": z_text,
                "z_graph": z_graph,
                "z_fused": z_fused,
                "transformer_layers": albert_out.hidden_states,
                "gin_layers": gin_layer_reps
            }

        return logits_primary, logits_subtype, z_fused


def load_trained_model(checkpoint_path: str = BEST_MODEL_PATH, device: torch.device = DEVICE) -> AlbertGINClassifier:
    """
    Instantiates the model and loads weights from the checkpoint file.
    Automatically handles 'module.' prefix stripping from multi-GPU DataParallel training.
    """
    prim_id2label, _, sub_id2label, _ = load_label_mappings()
    model = AlbertGINClassifier(
        albert_model_name=MODEL_NAME,
        num_primary=len(prim_id2label),
        num_subtypes=len(sub_id2label),
        fused_dim=256
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)

    # Clean 'module.' prefixes if trained with DataParallel
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        cleaned_state_dict[new_key] = v

    model.load_state_dict(cleaned_state_dict)
    model.to(device)
    model.eval()
    print("Model successfully loaded and set to eval mode.")
    return model
