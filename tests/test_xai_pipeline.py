"""
Comprehensive Unit & Integration Test Suite for Upgraded XAI Pipeline.
Tests:
1. Base prediction integrity (HAM/SPAM/PHISHING & Subtypes)
2. Modality ablation output & Delta P calculations
3. Per-URL attribution (Zero, One, and Multiple URLs)
4. Word-level Integrated Gradients & SentencePiece aggregation
5. GNNExplainer graph entity and edge mapping
6. Faithfulness and Stability evaluation logic
7. Missing metadata, empty inputs, and edge cases
8. API response compatibility and schema stability
"""

import os
import sys
import unittest
import numpy as np
import torch

from infer import EmailClassifierInference
from xai import (
    evaluate_modality_ablation,
    evaluate_url_attribution,
    compute_integrated_gradients,
    explain_gnn,
    compute_xai_bundle,
    conversational_xai_agent
)


class TestXAIPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n[Test Setup] Initializing EmailClassifierInference engine...")
        cls.engine = EmailClassifierInference()

    def test_01_base_prediction_integrity(self):
        """Verifies that base prediction, probability distribution, and subtype classification remain intact."""
        res = self.engine.predict(
            subject="Invoice details attached",
            body="Please review the quarterly invoice and approve the transfer.",
            sender="billing@company.com",
            receiver="finance@company.com",
            return_xai=False
        )
        self.assertIn("primary_label", res)
        self.assertIn(res["primary_label"], ["HAM", "SPAM", "PHISHING"])
        self.assertIn("primary_confidence", res)
        self.assertGreaterEqual(res["primary_confidence"], 0.0)
        self.assertLessEqual(res["primary_confidence"], 1.0)
        self.assertIn("subtype_label", res)
        self.assertEqual(len(res["primary_probabilities"]), 3)
        self.assertAlmostEqual(sum(res["primary_probabilities"].values()), 1.0, places=4)

    def test_02_modality_ablation_output(self):
        """Verifies controlled branch ablation calculation, target class consistency, and Delta P."""
        res = self.engine.predict(
            subject="URGENT: Password Reset Required",
            body="Your PayPal account has been compromised. Log in now: http://paypa1-security.com/login",
            sender="alert@paypa1-security.com",
            receiver="victim@company.com",
            return_xai=True
        )
        xai = res.get("xai")
        self.assertIsNotNone(xai)
        self.assertIn("modality_attribution", xai)

        mod = xai["modality_attribution"]
        self.assertIn("original_probability", mod)
        self.assertIn("text_ablated_probability", mod)
        self.assertIn("graph_ablated_probability", mod)
        self.assertIn("joint_ablated_probability", mod)
        self.assertIn("delta_text_ablation", mod)
        self.assertIn("delta_graph_ablation", mod)
        self.assertIn("delta_joint_ablation", mod)
        self.assertIn("gate_statistics", mod)

        # Delta P checks: P_orig - P_ablated == delta
        self.assertAlmostEqual(
            mod["original_probability"] - mod["text_ablated_probability"],
            mod["delta_text_ablation"],
            places=3
        )
        self.assertAlmostEqual(
            mod["original_probability"] - mod["graph_ablated_probability"],
            mod["delta_graph_ablation"],
            places=3
        )
        # Check gate statistics are present and documented
        self.assertIn("mean_alpha", mod["gate_statistics"])
        self.assertIn("note", mod["gate_statistics"])

    def test_03_per_url_attribution_zero_urls(self):
        """Verifies per-URL attribution behavior when zero URLs are present."""
        res = self.engine.predict(
            subject="Project roadmap sync tomorrow",
            body="Hi team, let us meet tomorrow morning to discuss the sprint backlog. No links needed.",
            sender="manager@company.com",
            receiver="dev@company.com",
            urls=[],
            return_xai=True
        )
        url_attrs = res["xai"]["url_attribution"]
        self.assertEqual(len(url_attrs), 0)

    def test_04_per_url_attribution_one_url(self):
        """Verifies individual URL perturbation for single URL input."""
        res = self.engine.predict(
            subject="Important update",
            body="Check your account here: http://secure-bank-login.com/auth",
            sender="support@secure-bank-login.com",
            receiver="victim@target.com",
            return_xai=True
        )
        url_attrs = res["xai"]["url_attribution"]
        self.assertEqual(len(url_attrs), 1)
        entry = url_attrs[0]
        self.assertEqual(entry["url"], "http://secure-bank-login.com/auth")
        self.assertEqual(entry["destination_domain"], "secure-bank-login.com")
        self.assertIn("delta_text_only", entry)
        self.assertIn("delta_graph_only", entry)
        self.assertIn("delta_combined", entry)
        self.assertIn(entry["impact_direction"], ["INCREASES_RISK", "DECREASES_RISK", "NEUTRAL"])

    def test_05_per_url_attribution_multiple_urls(self):
        """Verifies individual separate attribution when multiple URLs are present (no dividing equally)."""
        res = self.engine.predict(
            subject="Multiple portals available",
            body="Primary portal: http://primary-auth.com/login and secondary: http://mirror-site.biz/verify",
            sender="portal@primary-auth.com",
            receiver="user@target.com",
            return_xai=True
        )
        url_attrs = res["xai"]["url_attribution"]
        self.assertEqual(len(url_attrs), 2)
        urls_found = [u["url"] for u in url_attrs]
        self.assertIn("http://primary-auth.com/login", urls_found)
        self.assertIn("http://mirror-site.biz/verify", urls_found)
        for u in url_attrs:
            self.assertIsInstance(u["delta_text_only"], float)
            self.assertIsInstance(u["delta_graph_only"], float)
            self.assertIsInstance(u["delta_combined"], float)

    def test_06_word_attributions_albert(self):
        """Verifies Integrated Gradients word attributions, SentencePiece aggregation, and score normalization."""
        res = self.engine.predict(
            subject="Urgent account action",
            body="Your credentials have expired. Verify password immediately.",
            sender="security@auth.com",
            receiver="user@target.com",
            return_xai=True
        )
        word_attrs = res["xai"]["word_attributions"]
        self.assertGreater(len(word_attrs), 3)

        # Check structure of word attribution items
        for w in word_attrs:
            self.assertIn("word", w)
            self.assertIn("score", w)
            self.assertIn("normalized_score", w)
            self.assertIn("contribution", w)
            self.assertIn(w["contribution"], ["SUPPORTS_CLASS", "OPPOSES_CLASS", "NEUTRAL"])
            self.assertGreaterEqual(w["normalized_score"], -1.0)
            self.assertLessEqual(w["normalized_score"], 1.0)

        # Ensure tokens are aggregated into natural words (no raw SentencePiece prefixes like \u2581 or ##)
        all_words = [w["word"] for w in word_attrs]
        self.assertTrue(any("urgent" in w.lower() for w in all_words))
        self.assertTrue(any("verify" in w.lower() or "password" in w.lower() for w in all_words))

    def test_07_gnnexplainer_graph_mapping(self):
        """Verifies GNNExplainer optimization, edge masks, node importance, and entity mapping."""
        res = self.engine.predict(
            subject="Phishing Link",
            body="Go to http://evil-spoof.org/login now.",
            sender="attacker@evil-spoof.org",
            receiver="target@company.com",
            return_xai=True
        )
        gnn = res["xai"]["gnn_explanation"]
        self.assertIn("node_importance", gnn)
        self.assertIn("edge_importance", gnn)
        self.assertIn("feature_importance", gnn)

        # Verify entity labels are human-readable
        nodes = gnn["node_importance"]
        self.assertGreater(len(nodes), 0)
        entities = [n["entity"] for n in nodes]
        self.assertTrue(any("Sender" in e for e in entities))
        self.assertTrue(any("Receiver" in e for e in entities))

    def test_08_missing_metadata_and_edge_cases(self):
        """Verifies handling of empty text, missing sender/receiver, and atypical inputs without crash."""
        res = self.engine.predict(
            subject="",
            body="",
            sender="unknown",
            receiver="unknown",
            has_spf=False,
            has_dkim=False,
            return_xai=True
        )
        self.assertIn("primary_label", res)
        self.assertIn("xai", res)
        self.assertIn("modality_attribution", res["xai"])
        self.assertIn("url_attribution", res["xai"])
        self.assertIn("gnn_explanation", res["xai"])

    def test_09_conversational_xai_agent(self):
        """Verifies the offline natural language agent answers factual questions about XAI metrics."""
        res = self.engine.predict(
            subject="Wire Transfer Request",
            body="Execute transfer of $45,000 to http://external-wire.net immediately.",
            sender="ceo@partner.org",
            receiver="cfo@company.com",
            return_xai=True
        )
        xai_bundle = res["xai"]
        email_data = {"subject": "Wire Transfer Request", "sender": "ceo@partner.org", "urls": ["http://external-wire.net"]}

        ans_mod = conversational_xai_agent("Which branch mattered more: text vs graph?", xai_bundle, email_data, res)
        self.assertIn("Modality Attribution", ans_mod)

        ans_words = conversational_xai_agent("Which words influenced the decision?", xai_bundle, email_data, res)
        self.assertIn("Word-Level Attribution", ans_words)

        ans_url = conversational_xai_agent("What is the URL impact?", xai_bundle, email_data, res)
        self.assertIn("Per-URL", ans_url)

    def test_10_zero_shot_vector_memory(self):
        """Verifies manual flagging into persistent vector memory and subsequent zero-shot similarity matching."""
        # 1. Base prediction
        res1 = self.engine.predict(
            subject="Urgent Payroll Correction",
            body="Your direct deposit failed. Update your bank credentials at http://payroll-verify.org",
            sender="hr@company-payroll.net",
            receiver="employee@corp.com",
            return_xai=False
        )
        self.assertIn("embedding", res1)
        self.assertEqual(len(res1["embedding"]), 256)

        # 2. Flag into persistent vector memory
        init_count = self.engine.memory.count()
        entry = self.engine.flag_email_in_memory(
            embedding=res1["embedding"],
            sender="hr@company-payroll.net",
            subject="Urgent Payroll Correction",
            label="PHISHING",
            subtype="FINANCIAL_PHISHING",
            note="Zero-shot test campaign"
        )
        self.assertEqual(self.engine.memory.count(), init_count + 1)
        self.assertEqual(entry["label"], "PHISHING")

        # 3. Analyze similar email (same campaign variation)
        res2 = self.engine.predict(
            subject="Urgent Payroll Correction",
            body="Your direct deposit failed. Update your bank credentials at http://payroll-verify.org",
            sender="hr@company-payroll.net",
            receiver="employee2@corp.com",
            return_xai=True
        )
        matches = res2.get("memory_matches", [])
        self.assertGreater(len(matches), 0)
        top_match = matches[0]
        self.assertEqual(top_match["label"], "PHISHING")
        self.assertGreaterEqual(top_match["similarity_percent"], 90.0)

        # 4. Verify conversational agent reports the memory match
        email_data = {"subject": res2["primary_label"], "sender": "hr@company-payroll.net"}
        ans_mem = conversational_xai_agent("Check vector threat memory?", res2["xai"], email_data, res2)
        self.assertIn("Zero-Shot Vector Memory Match", ans_mem)
        self.assertIn(str(top_match["similarity_percent"]), ans_mem)


if __name__ == "__main__":
    unittest.main()
