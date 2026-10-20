"""
Dataset and Graph Construction module for Dual-Branch ALBERT + GIN.
Transforms raw email metadata and textual content into:
- ALBERT tokenized sequences [input_ids, attention_mask]
- Attributed entity graphs [node_features, adjacency_matrix]
"""

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from config import (
    MODEL_NAME, MAX_LEN, BATCH_SIZE, SEED,
    get_dataset_path, load_label_mappings
)


def build_email_graph(row) -> tuple:
    """
    Constructs an attributed entity graph for a single email record.
    Nodes:
        0: Sender
        1: Receiver
        2: Sender Domain
        3..K: Destination Domains / URLs (max 10 external nodes)
    Node Features: 16-dim vector containing one-hot entity type, SPF, DKIM,
                   normalized URL counts, domain counts, and body lengths.
    Returns:
        features: np.ndarray [14, 16]
        adj: np.ndarray [14, 14]
        num_nodes: int
    """
    # Parse destination domains safely
    raw_domains = row.get("domains", [])
    try:
        if isinstance(raw_domains, str):
            domains_list = json.loads(raw_domains) if raw_domains.startswith("[") else [raw_domains]
        elif isinstance(raw_domains, (list, tuple, np.ndarray)):
            domains_list = list(raw_domains)
        else:
            domains_list = []
    except Exception:
        domains_list = []

    # Filter out empty or None domains
    domains_list = [str(d).strip() for d in domains_list if d and str(d).strip()]
    domains_list = domains_list[:10]  # Cap to max 10 destination nodes
    num_nodes = 3 + len(domains_list)

    MAX_GRAPH_NODES = 14
    features = np.zeros((MAX_GRAPH_NODES, 16), dtype=np.float32)
    adj = np.zeros((MAX_GRAPH_NODES, MAX_GRAPH_NODES), dtype=np.float32)

    # Node 0: Sender
    features[0, 0] = 1.0
    features[0, 4] = float(row.get("has_spf", 0) if pd.notnull(row.get("has_spf", 0)) else 0)
    features[0, 5] = float(row.get("has_dkim", 0) if pd.notnull(row.get("has_dkim", 0)) else 0)
    num_urls = float(row.get("num_urls", 0) if pd.notnull(row.get("num_urls", 0)) else 0)
    num_domains = float(row.get("num_domains", 0) if pd.notnull(row.get("num_domains", 0)) else 0)
    body_len = float(row.get("body_length", 0) if pd.notnull(row.get("body_length", 0)) else 0)

    features[0, 6] = min(num_urls / 10.0, 1.0)
    features[0, 7] = min(num_domains / 10.0, 1.0)
    features[0, 8] = min(body_len / 5000.0, 1.0)

    # Node 1: Receiver
    features[1, 1] = 1.0
    features[1, 8] = features[0, 8]

    # Node 2: Sender Domain
    features[2, 2] = 1.0
    features[2, 4] = features[0, 4]
    features[2, 5] = features[0, 5]

    # Node 3..K: Destination Domains / URLs
    for i in range(len(domains_list)):
        idx = 3 + i
        features[idx, 3] = 1.0
        features[idx, 6] = 1.0 / (i + 1)
        # Edge between Receiver and Destination Domain
        adj[1, idx] = 1.0
        adj[idx, 1] = 1.0

    # Bidirectional Core Edges
    # Sender <-> Sender Domain
    adj[0, 2] = 1.0
    adj[2, 0] = 1.0
    # Sender <-> Receiver
    adj[0, 1] = 1.0
    adj[1, 0] = 1.0

    return features, adj, num_nodes


class EmailMultiModalDataset(Dataset):
    """PyTorch Dataset producing tokenized textual inputs and structural graph tensors."""
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_len: int = MAX_LEN):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Combine subject and body text
        subject = str(row.get("subject", "") or "")
        body = str(row.get("body", "") or "")
        full_text = f"{subject} {body}".strip()
        if not full_text:
            full_text = "Empty email content."

        # Tokenize for ALBERT
        tokens = self.tokenizer(
            full_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Build Graph tensors
        node_feats, adj, num_nodes = build_email_graph(row)

        item = {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "node_feats": torch.from_numpy(node_feats),
            "adj": torch.from_numpy(adj),
            "num_nodes": num_nodes
        }

        if "primary_target" in row:
            item["primary_target"] = torch.tensor(row["primary_target"], dtype=torch.long)
        if "subtype_target" in row:
            item["subtype_target"] = torch.tensor(row["subtype_target"], dtype=torch.long)

        return item


def collate_multimodal_batch(batch: list) -> dict:
    """Collates a list of dataset samples into batch tensors."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    node_feats = torch.stack([b["node_feats"] for b in batch])
    adj = torch.stack([b["adj"] for b in batch])

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "node_feats": node_feats,
        "adj": adj,
        "batch_size": len(batch)
    }

    if "primary_target" in batch[0]:
        out["primary_targets"] = torch.stack([b["primary_target"] for b in batch])
    if "subtype_target" in batch[0]:
        out["subtype_targets"] = torch.stack([b["subtype_target"] for b in batch])

    return out


def load_dataset_dataframe(dataset_path: str = None) -> pd.DataFrame:
    """Loads parquet dataset and assigns numerical targets."""
    if dataset_path is None:
        dataset_path = get_dataset_path()

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Parquet dataset not found at: {dataset_path}")

    print(f"Loading dataset from: {dataset_path}...")
    df = pd.read_parquet(dataset_path)

    _, prim_label2id, _, sub_label2id = load_label_mappings()
    if "primary_label" in df.columns:
        df["primary_target"] = df["primary_label"].map(prim_label2id)
    if "phishing_subtype" in df.columns:
        df["subtype_target"] = df["phishing_subtype"].map(sub_label2id)

    print(f"Loaded {len(df)} records. Columns: {list(df.columns)}")
    return df


def get_dataloaders(
    dataset_path: str = None,
    batch_size: int = BATCH_SIZE,
    max_len: int = MAX_LEN,
    num_workers: int = 0
) -> tuple:
    """
    Performs stratified 70% train / 15% validation / 15% test splits
    and returns DataLoaders and Split DataFrames.
    """
    df = load_dataset_dataframe(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=df["primary_target"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["primary_target"]
    )

    print(f"Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    train_ds = EmailMultiModalDataset(train_df, tokenizer, max_len=max_len)
    val_ds = EmailMultiModalDataset(val_df, tokenizer, max_len=max_len)
    test_ds = EmailMultiModalDataset(test_df, tokenizer, max_len=max_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_multimodal_batch, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_multimodal_batch, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_multimodal_batch, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, train_df, val_df, test_df
