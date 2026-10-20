"""
Dynamic Vector Memory module for Dual-Branch ALBERT + GIN Email Classification.
Implements instant retrieval-augmented threat memory:
- Stores 256-dimensional fused embeddings (z_fused) of known/flagged emails
- Performs sub-millisecond cosine similarity search across stored threat vectors
- Alerts the user when an unseen email matches a previously encountered attack or safe email
"""

import os
import json
import time
import numpy as np

from config import OUTPUT_DIR

MEMORY_DIR = os.path.join(OUTPUT_DIR, "memory")
VECTORS_FILE = os.path.join(MEMORY_DIR, "vectors.npy")
ENTRIES_FILE = os.path.join(MEMORY_DIR, "entries.json")


class VectorMemory:
    """Manages persistent 256-dimensional email embedding memory."""

    def __init__(self, memory_dir: str = MEMORY_DIR):
        self.memory_dir = memory_dir
        self.vectors_file = os.path.join(memory_dir, "vectors.npy")
        self.entries_file = os.path.join(memory_dir, "entries.json")
        os.makedirs(self.memory_dir, exist_ok=True)

        self.vectors = np.empty((0, 256), dtype=np.float32)
        self.entries = []
        self._load()

    def _load(self):
        """Loads vectors and metadata from disk if available."""
        if os.path.exists(self.vectors_file) and os.path.exists(self.entries_file):
            try:
                self.vectors = np.load(self.vectors_file).astype(np.float32)
                with open(self.entries_file, "r", encoding="utf-8") as f:
                    self.entries = json.load(f)
                # Verify consistency
                if len(self.vectors) != len(self.entries):
                    min_len = min(len(self.vectors), len(self.entries))
                    self.vectors = self.vectors[:min_len]
                    self.entries = self.entries[:min_len]
            except Exception as e:
                print(f"[VectorMemory] Warning loading memory: {e}. Starting fresh.")
                self.vectors = np.empty((0, 256), dtype=np.float32)
                self.entries = []

    def _save(self):
        """Saves current state to disk."""
        np.save(self.vectors_file, self.vectors)
        with open(self.entries_file, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2)

    def add(
        self,
        embedding: np.ndarray,
        sender: str,
        subject: str,
        label: str,
        subtype: str = "CUSTOM",
        note: str = ""
    ) -> dict:
        """
        Adds a new 256-dim embedding vector and metadata to the memory bank.
        """
        vec = np.asarray(embedding, dtype=np.float32).flatten()
        if vec.shape[0] != 256:
            raise ValueError(f"Expected 256-dim embedding vector, got shape {vec.shape}")

        # Normalize vector for fast cosine similarity via dot product
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm

        entry_id = len(self.entries) + 1
        entry = {
            "id": entry_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sender": sender,
            "subject": subject,
            "label": label.upper(),
            "subtype": subtype,
            "note": note or f"Cataloged as {label.upper()}"
        }

        if len(self.vectors) == 0:
            self.vectors = np.expand_dims(vec, axis=0)
        else:
            self.vectors = np.vstack([self.vectors, vec])

        self.entries.append(entry)
        self._save()
        return entry

    def query(self, embedding: np.ndarray, top_k: int = 3, threshold: float = 0.70) -> list:
        """
        Finds nearest neighbors to the query embedding using cosine similarity.
        Returns matches with similarity >= threshold.
        """
        if len(self.vectors) == 0:
            return []

        vec = np.asarray(embedding, dtype=np.float32).flatten()
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm

        # Batched cosine similarities via dot product: [N, 256] x [256] -> [N]
        sims = np.dot(self.vectors, vec)

        # Sort descending
        top_indices = np.argsort(-sims)[:top_k]

        matches = []
        for idx in top_indices:
            score = float(sims[idx])
            if score >= threshold:
                entry = self.entries[idx].copy()
                entry["similarity_percent"] = round(score * 100, 1)
                entry["raw_similarity"] = score
                matches.append(entry)

        return matches

    def count(self) -> int:
        return len(self.entries)

    def list_all(self, limit: int = 50) -> list:
        return list(reversed(self.entries))[:limit]

    def clear(self):
        self.vectors = np.empty((0, 256), dtype=np.float32)
        self.entries = []
        self._save()
