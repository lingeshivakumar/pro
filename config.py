"""
Configuration module for Dual-Branch ALBERT + GIN Email Classification.
Handles file paths, hyperparameters, device management, and label mappings.
"""

import os
import json
import torch

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
EMBEDDINGS_DIR = os.path.join(OUTPUT_DIR, "embeddings")
LAYER_OUTPUTS_DIR = os.path.join(OUTPUT_DIR, "layer_outputs")

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
LABEL_MAPPING_PATH = os.path.join(METRICS_DIR, "label_mapping.json")

# Ensure output directories exist
for directory in [
    OUTPUT_DIR, CHECKPOINT_DIR, METRICS_DIR, TABLES_DIR,
    IMAGES_DIR, EMBEDDINGS_DIR, LAYER_OUTPUTS_DIR
]:
    os.makedirs(directory, exist_ok=True)

# Dataset path resolution
DATASET_CANDIDATES = [
    os.path.join(BASE_DIR, "dataset", "augmented_66k.parquet"),
    os.path.join(BASE_DIR, "augmented_66k.parquet"),
    "C:/Users/acer/Desktop/rev2tasks/dataset/augmented_dataset/augmented_66k.parquet"
]

def get_dataset_path():
    """Finds the dataset file across common local candidate paths."""
    for path in DATASET_CANDIDATES:
        if os.path.exists(path) and os.path.isfile(path):
            return path
    return DATASET_CANDIDATES[0]

# Hardware and Execution Settings
SEED = 42
DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available()
    else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
)
USE_CUDA = torch.cuda.is_available()
NUM_GPUS = torch.cuda.device_count()

# Model Hyperparameters
MODEL_NAME = "albert-large-v2"
MAX_LEN = 256
BATCH_SIZE = 16 if NUM_GPUS >= 2 else (8 if USE_CUDA else 4)
EPOCHS = 5

# Learning rates and optimization
LR_TRANSFORMER = 2e-5
LR_GRAPH_AND_HEADS = 1e-3
WEIGHT_DECAY = 0.01

# Default label mappings
DEFAULT_PRIMARY_LABELS = {"HAM": 0, "PHISHING": 1, "SPAM": 2}
DEFAULT_SUBTYPE_LABELS = {
    "NOT_PHISHING": 0,
    "BEC": 1,
    "CREDENTIAL_PHISHING": 2,
    "DELIVERY_PHISHING": 3,
    "FINANCIAL_PHISHING": 4,
    "GENERIC_PHISHING": 5,
    "GOVERNMENT_PHISHING": 6,
    "MALWARE_DELIVERY": 7,
    "SURVEY_PHISHING": 8,
    "TECH_SUPPORT_PHISHING": 9
}

def load_label_mappings():
    """Loads label mappings from saved JSON if present, otherwise returns defaults."""
    if os.path.exists(LABEL_MAPPING_PATH):
        try:
            with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            prim_id2label = {int(k): v for k, v in data["primary_labels"].items()}
            prim_label2id = {v: int(k) for k, v in data["primary_labels"].items()}
            sub_id2label = {int(k): v for k, v in data["subtype_labels"].items()}
            sub_label2id = {v: int(k) for k, v in data["subtype_labels"].items()}
            return prim_id2label, prim_label2id, sub_id2label, sub_label2id
        except Exception as e:
            print(f"[Warning] Failed loading label mapping from {LABEL_MAPPING_PATH}: {e}")
    
    prim_id2label = {v: k for k, v in DEFAULT_PRIMARY_LABELS.items()}
    prim_label2id = DEFAULT_PRIMARY_LABELS
    sub_id2label = {v: k for k, v in DEFAULT_SUBTYPE_LABELS.items()}
    sub_label2id = DEFAULT_SUBTYPE_LABELS
    return prim_id2label, prim_label2id, sub_id2label, sub_label2id
