"""
Training module for Dual-Branch ALBERT + GIN Email Classification.
Implements:
- Multi-task joint loss (Primary Cross-Entropy + 0.5 * Subtype Cross-Entropy)
- Differential learning rates (lower for pretrained ALBERT, higher for GIN & heads)
- Automatic Mixed Precision (AMP)
- Multi-GPU DataParallel support
- Checkpointing and history logging
"""

import os
import time
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from config import (
    MODEL_NAME, BATCH_SIZE, EPOCHS, LR_TRANSFORMER, LR_GRAPH_AND_HEADS,
    WEIGHT_DECAY, BEST_MODEL_PATH, METRICS_DIR, IMAGES_DIR,
    DEVICE, NUM_GPUS, load_label_mappings, get_dataset_path
)
from models import AlbertGINClassifier
from dataset import get_dataloaders


def train_model(
    dataset_path: str = None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr_albert: float = LR_TRANSFORMER,
    lr_heads: float = LR_GRAPH_AND_HEADS,
    checkpoint_out: str = BEST_MODEL_PATH
):
    """Executes the training and validation loops."""
    if dataset_path is None:
        dataset_path = get_dataset_path()

    print("=" * 75)
    print(f"Device        : {DEVICE} (Available GPUs: {NUM_GPUS})")
    print(f"Dataset       : {dataset_path}")
    print(f"Epochs        : {epochs} | Batch Size: {batch_size}")
    print("=" * 75)

    prim_id2label, _, sub_id2label, _ = load_label_mappings()
    train_loader, val_loader, _, train_df, val_df, _ = get_dataloaders(
        dataset_path=dataset_path,
        batch_size=batch_size
    )

    model = AlbertGINClassifier(
        albert_model_name=MODEL_NAME,
        num_primary=len(prim_id2label),
        num_subtypes=len(sub_id2label),
        fused_dim=256
    )
    model.to(DEVICE)

    if NUM_GPUS >= 2:
        print(f"Enabling DataParallel across {NUM_GPUS} GPUs...")
        train_model_wrapper = nn.DataParallel(model)
    else:
        train_model_wrapper = model

    # Differential Learning Rates
    optimizer_params = [
        {
            "params": [p for n, p in model.albert.named_parameters() if p.requires_grad],
            "lr": lr_albert,
            "weight_decay": WEIGHT_DECAY
        },
        {
            "params": [p for n, p in model.gin.named_parameters() if p.requires_grad] +
                      [p for n, p in model.fusion.named_parameters() if p.requires_grad] +
                      [p for n, p in model.primary_head.named_parameters() if p.requires_grad] +
                      [p for n, p in model.subtype_head.named_parameters() if p.requires_grad],
            "lr": lr_heads,
            "weight_decay": 1e-4
        }
    ]

    optimizer = torch.optim.AdamW(optimizer_params)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    crit_primary = nn.CrossEntropyLoss()
    crit_subtype = nn.CrossEntropyLoss()

    history = {
        "epoch": [], "train_loss": [], "train_prim_acc": [],
        "val_loss": [], "val_prim_acc": [], "val_sub_acc": []
    }
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        total_train_loss = 0.0
        correct_train_prim = 0
        total_samples = 0

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            node_feats = batch["node_feats"].to(DEVICE)
            adj = batch["adj"].to(DEVICE)
            y_prim = batch["primary_targets"].to(DEVICE)
            y_sub = batch["subtype_targets"].to(DEVICE)
            bsz = batch["batch_size"]

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits_prim, logits_sub, _ = train_model_wrapper(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    node_feats=node_feats,
                    adj=adj
                )
                loss_prim = crit_primary(logits_prim, y_prim)
                loss_sub = crit_subtype(logits_sub, y_sub)
                loss = loss_prim + 0.5 * loss_sub

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item() * bsz
            preds_prim = logits_prim.argmax(dim=-1)
            correct_train_prim += (preds_prim == y_prim).sum().item()
            total_samples += bsz

            if step % 100 == 0 or step == len(train_loader):
                acc = (correct_train_prim / total_samples) * 100
                print(f"Epoch [{epoch}/{epochs}] Step [{step}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} | Train Acc: {acc:.2f}%")

        # Validation Phase
        model.eval()
        total_val_loss = 0.0
        correct_val_prim = 0
        correct_val_sub = 0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                node_feats = batch["node_feats"].to(DEVICE)
                adj = batch["adj"].to(DEVICE)
                y_prim = batch["primary_targets"].to(DEVICE)
                y_sub = batch["subtype_targets"].to(DEVICE)
                bsz = batch["batch_size"]

                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    logits_prim, logits_sub, _ = train_model_wrapper(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        node_feats=node_feats,
                        adj=adj
                    )
                    loss_prim = crit_primary(logits_prim, y_prim)
                    loss_sub = crit_subtype(logits_sub, y_sub)
                    loss = loss_prim + 0.5 * loss_sub

                total_val_loss += loss.item() * bsz
                correct_val_prim += (logits_prim.argmax(dim=-1) == y_prim).sum().item()
                correct_val_sub += (logits_sub.argmax(dim=-1) == y_sub).sum().item()
                val_samples += bsz

        train_loss = total_train_loss / total_samples
        train_acc = correct_train_prim / total_samples
        val_loss = total_val_loss / val_samples
        val_prim_acc = correct_val_prim / val_samples
        val_sub_acc = correct_val_sub / val_samples
        elapsed = time.time() - start_time

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_prim_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_prim_acc"].append(val_prim_acc)
        history["val_sub_acc"].append(val_sub_acc)

        print("-" * 75)
        print(f"Epoch {epoch} Completed in {elapsed:.1f}s | "
              f"Val Loss: {val_loss:.4f} | Val Prim Acc: {val_prim_acc * 100:.2f}% | "
              f"Val Sub Acc: {val_sub_acc * 100:.2f}%")

        if val_prim_acc > best_val_acc:
            best_val_acc = val_prim_acc
            print(f">>> New Best Validation Accuracy ({best_val_acc * 100:.2f}%)! Saving checkpoint...")
            torch.save(model.state_dict(), checkpoint_out)

    # Save training history files
    hist_csv = os.path.join(METRICS_DIR, "training_history.csv")
    hist_json = os.path.join(METRICS_DIR, "training_history.json")
    pd.DataFrame(history).to_csv(hist_csv, index=False)
    with open(hist_json, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Save curves plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history["epoch"], history["train_loss"], "o-", label="Train Loss", color="#1f77b4")
    axes[0].plot(history["epoch"], history["val_loss"], "s--", label="Val Loss", color="#ff7f0e")
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(history["epoch"], [a * 100 for a in history["train_prim_acc"]], "o-", label="Train Prim Acc", color="#2ca02c")
    axes[1].plot(history["epoch"], [a * 100 for a in history["val_prim_acc"]], "s--", label="Val Prim Acc", color="#d62728")
    axes[1].plot(history["epoch"], [a * 100 for a in history["val_sub_acc"]], "^-.", label="Val Subtype Acc", color="#9467bd")
    axes[1].set_title("Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    curves_path = os.path.join(IMAGES_DIR, "loss_accuracy_curves.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()

    print(f"Training completed. Checkpoint saved to: {checkpoint_out}")


def main():
    parser = argparse.ArgumentParser(description="Train Dual-Branch Email Classifier")
    parser.add_argument("--dataset", type=str, default=None, help="Path to parquet dataset")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size per step")
    parser.add_argument("--lr-albert", type=float, default=LR_TRANSFORMER, help="Learning rate for ALBERT")
    parser.add_argument("--lr-heads", type=float, default=LR_GRAPH_AND_HEADS, help="Learning rate for GIN and Heads")

    args = parser.parse_args()
    train_model(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_albert=args.lr_albert,
        lr_heads=args.lr_heads
    )


if __name__ == "__main__":
    main()
