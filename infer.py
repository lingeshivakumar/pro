"""
Standalone Inference module for Dual-Branch ALBERT + GIN Email Classification.
Allows immediate classification of arbitrary emails using the saved best_model.pt
checkpoint without requiring any retraining.
"""

import sys
import os
import re
import argparse
import urllib.parse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Reconfigure stdout for utf-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


from config import (
    MODEL_NAME, MAX_LEN, BEST_MODEL_PATH, DEVICE,
    load_label_mappings
)
from models import AlbertGINClassifier, load_trained_model
from dataset import build_email_graph
from memory import VectorMemory


class EmailClassifierInference:
    """Inference engine with dynamic vector memory for dual-branch email classification."""
    def __init__(self, checkpoint_path: str = BEST_MODEL_PATH, device: torch.device = None):
        if device is None:
            self.device = DEVICE
        else:
            self.device = device

        self.prim_id2label, self.prim_label2id, self.sub_id2label, self.sub_label2id = load_label_mappings()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = load_trained_model(checkpoint_path=checkpoint_path, device=self.device)
        self.memory = VectorMemory()

    def extract_domains_from_urls(self, urls: list) -> list:
        """Extracts unique domain names from a list of URLs."""
        domains = []
        for u in urls:
            try:
                parsed = urllib.parse.urlparse(u if "://" in u else f"http://{u}")
                netloc = parsed.netloc.split(":")[0].lower()
                if netloc and netloc not in domains:
                    domains.append(netloc)
            except Exception:
                continue
        return domains

    def extract_sender_domain(self, sender: str) -> str:
        """Extracts the domain from an email address."""
        if "@" in sender:
            return sender.split("@")[-1].strip().lower()
        return "unknown.domain"

    def predict(
        self,
        subject: str = "",
        body: str = "",
        sender: str = "sender@example.com",
        receiver: str = "user@example.com",
        urls: list = None,
        has_spf: bool = False,
        has_dkim: bool = False,
        return_xai: bool = False,
        **kwargs
    ) -> dict:
        """
        Classifies an email and returns predictions, probabilities, and risk score.
        If return_xai=True, generates a full multimodal explainability bundle.
        """
        if urls is None:
            url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
            urls = re.findall(url_pattern, f"{subject} {body}")

        destination_domains = self.extract_domains_from_urls(urls)
        sender_domain = self.extract_sender_domain(sender)

        row_dict = {
            "sender": sender,
            "receiver": receiver,
            "sender_domain": sender_domain,
            "domains": destination_domains,
            "has_spf": 1.0 if has_spf else 0.0,
            "has_dkim": 1.0 if has_dkim else 0.0,
            "num_urls": float(len(urls)),
            "num_domains": float(len(destination_domains)),
            "body_length": float(len(body)),
            "subject": subject,
            "body": body,
            "urls": urls
        }

        # 1. Tokenize textual branch
        full_text = f"{subject} {body}".strip()
        tokens = self.tokenizer(
            full_text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 2. Featurize graph branch
        node_feats, adj, num_nodes = build_email_graph(row_dict)

        # 3. Prepare tensors on target device
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)
        node_feats_t = torch.from_numpy(node_feats).unsqueeze(0).to(self.device)
        adj_t = torch.from_numpy(adj).unsqueeze(0).to(self.device)

        # 4. Model Forward Pass
        with torch.no_grad():
            if return_xai:
                out_layers = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    node_feats=node_feats_t,
                    adj=adj_t,
                    return_layers=True
                )
                logits_prim = out_layers["logits_primary"]
                logits_sub = out_layers["logits_subtype"]
                z_fused = out_layers["z_fused"]
                z_text = out_layers["z_text"]
                z_graph = out_layers["z_graph"]
            else:
                logits_prim, logits_sub, z_fused = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    node_feats=node_feats_t,
                    adj=adj_t
                )
                z_text, z_graph = None, None

            probs_prim = F.softmax(logits_prim, dim=-1).squeeze(0).cpu().numpy()
            probs_sub = F.softmax(logits_sub, dim=-1).squeeze(0).cpu().numpy()
            z_fused_np = z_fused.squeeze(0).cpu().numpy()

        pred_prim_id = int(np.argmax(probs_prim))
        pred_prim_label = self.prim_id2label.get(pred_prim_id, f"CLASS_{pred_prim_id}")
        pred_prim_conf = float(probs_prim[pred_prim_id])

        pred_sub_id = int(np.argmax(probs_sub))
        pred_sub_label = self.sub_id2label.get(pred_sub_id, f"SUBTYPE_{pred_sub_id}")
        pred_sub_conf = float(probs_sub[pred_sub_id])

        # Phishing risk score: direct phishing probability * 100
        phish_idx = [k for k, v in self.prim_id2label.items() if v == "PHISHING"]
        risk_score = float(probs_prim[phish_idx[0]] * 100) if phish_idx else 0.0

        # Query dynamic vector memory for nearest matching cataloged threats/safe emails
        memory_matches = self.memory.query(z_fused_np, top_k=3, threshold=0.72)

        base_res = {
            "primary_label": pred_prim_label,
            "primary_confidence": pred_prim_conf,
            "primary_probabilities": {
                self.prim_id2label.get(i, f"CLASS_{i}"): float(prob)
                for i, prob in enumerate(probs_prim)
            },
            "phishing_risk_score": risk_score,
            "subtype_label": pred_sub_label,
            "subtype_confidence": pred_sub_conf,
            "subtype_probabilities": {
                self.sub_id2label.get(i, f"SUBTYPE_{i}"): float(prob)
                for i, prob in enumerate(probs_sub)
            },
            "memory_matches": memory_matches,
            "memory_count": self.memory.count(),
            "embedding": z_fused_np.tolist(),
            "graph_metadata": {
                "num_nodes": num_nodes,
                "sender_domain": sender_domain,
                "destination_domains": destination_domains,
                "detected_urls": len(urls)
            }
        }

        if return_xai and z_text is not None and z_graph is not None:
            from xai import compute_xai_bundle
            xai_bundle = compute_xai_bundle(
                engine=self,
                email_dict=row_dict,
                base_result=base_res,
                z_text=z_text,
                z_graph=z_graph
            )
            base_res["xai"] = xai_bundle
            base_res["xai_bundle"] = xai_bundle

        return base_res

    def flag_email_in_memory(
        self,
        embedding: list,
        sender: str,
        subject: str,
        label: str,
        subtype: str = "CUSTOM",
        note: str = ""
    ) -> dict:
        """Persistently catalogs an email's 256-dim embedding and metadata into vector memory."""
        return self.memory.add(
            embedding=np.array(embedding, dtype=np.float32),
            sender=sender,
            subject=subject,
            label=label,
            subtype=subtype,
            note=note
        )


def print_prediction_card(res: dict, title: str = "Email Analysis Result"):
    """Prints a styled visual summary of classification results."""
    print("\n" + "=" * 65)
    print(f"             {title.upper()}")
    print("=" * 65)
    
    label = res["primary_label"]
    conf = res["primary_confidence"] * 100
    risk = res["phishing_risk_score"]
    subtype = res["subtype_label"]
    sub_conf = res["subtype_confidence"] * 100

    print(f"Primary Verdict    : {label} (Confidence: {conf:.1f}%)")
    print(f"Phishing Risk Score: {risk:.1f} / 100.0")
    print(f"Predicted Subtype  : {subtype} (Confidence: {sub_conf:.1f}%)")
    print("-" * 65)
    print("Primary Probability Breakdown:")
    for k, v in res["primary_probabilities"].items():
        bar = "#" * int(v * 25)
        print(f"  {k:10s} : [{bar:25s}] {v * 100:5.1f}%")
    
    print("-" * 65)
    print(f"Graph Features     : {res['graph_metadata']['num_nodes']} graph nodes | "
          f"{res['graph_metadata']['detected_urls']} URLs detected | "
          f"Sender Domain: {res['graph_metadata']['sender_domain']}")
    print("=" * 65 + "\n")


def run_demo():
    """Runs demonstration predictions on realistic sample emails."""
    print("Initializing inference engine...")
    engine = EmailClassifierInference()

    test_samples = [
        {
            "title": "Sample 1: Urgent Account Security Phishing",
            "subject": "CRITICAL ALERT: Your Microsoft 365 Account Will Be Suspended",
            "body": "Dear Employee, We detected abnormal login attempts. Click here immediately to verify your credentials: http://msoffice-verify-login.ru/secure and restore your account or it will be terminated within 24 hours.",
            "sender": "security-alert@office365-updates-auth.com",
            "receiver": "john.doe@company.com",
            "has_spf": False,
            "has_dkim": False
        },
        {
            "title": "Sample 2: Legitimate Internal Meeting Notification",
            "subject": "Project sync meeting tomorrow at 10:00 AM",
            "body": "Hi team, please find attached the weekly roadmap review agenda. Let's meet in Conference Room B or via our internal Google Meet link. Best regards, Sarah.",
            "sender": "sarah.jenkins@company.com",
            "receiver": "dev-team@company.com",
            "has_spf": True,
            "has_dkim": True
        },
        {
            "title": "Sample 3: Commercial Spam Offer",
            "subject": "EXCLUSIVE DISCOUNT: Save 70% on Luxury Watches Today Only!",
            "body": "Don't miss our flash blowout sale! Premium Swiss watches at unbelievable prices. Free shipping on all orders over $50. Unsubscribe anytime by visiting our website.",
            "sender": "promotions@watches-flashdeals.net",
            "receiver": "shopper@mail.com",
            "has_spf": False,
            "has_dkim": True
        }
    ]

    for sample in test_samples:
        res = engine.predict(
            subject=sample["subject"],
            body=sample["body"],
            sender=sample["sender"],
            receiver=sample["receiver"],
            has_spf=sample["has_spf"],
            has_dkim=sample["has_dkim"]
        )
        print_prediction_card(res, title=sample["title"])


def main():
    parser = argparse.ArgumentParser(description="Dual-Branch Email Classification Inference CLI")
    parser.add_argument("--subject", type=str, default="", help="Email Subject")
    parser.add_argument("--body", type=str, default="", help="Email Body Text")
    parser.add_argument("--sender", type=str, default="user@example.com", help="Sender email address")
    parser.add_argument("--receiver", type=str, default="recipient@example.com", help="Receiver email address")
    parser.add_argument("--spf", action="store_true", help="Set flag if SPF authentication passed")
    parser.add_argument("--dkim", action="store_true", help="Set flag if DKIM authentication passed")
    parser.add_argument("--demo", action="store_true", help="Run demonstration on sample emails")

    args = parser.parse_args()

    if args.demo or (not args.subject and not args.body):
        run_demo()
        return

    engine = EmailClassifierInference()
    result = engine.predict(
        subject=args.subject,
        body=args.body,
        sender=args.sender,
        receiver=args.receiver,
        has_spf=args.spf,
        has_dkim=args.dkim
    )
    print_prediction_card(result)


if __name__ == "__main__":
    main()
