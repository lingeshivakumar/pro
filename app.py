"""
Local Web GUI Dashboard for Dual-Branch ALBERT + GIN Email Classification
with Explainable AI (XAI) Suite & Dynamic Vector Memory.

Usage:
    python app.py
"""

import sys
import os
import json
import webbrowser
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

class ThreadedHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from infer import EmailClassifierInference
from xai import conversational_xai_agent

PORT = 7860
classifier = None

PRESETS = {
    "phishing": {
        "name": "Credential Phishing (PayPal Spoof)",
        "sender": "service-security@paypa1-update-center.com",
        "receiver": "user@corporate-network.com",
        "subject": "URGENT: Your account has been suspended! Verify your credentials",
        "body": "Dear customer, We detected unauthorized attempts to access your account. Your account is temporarily locked. Please log in immediately at http://login-paypa1-account-security.com/verification to re-activate your profile. Failure to do so within 24 hours will result in permanent suspension.",
        "spf": False,
        "dkim": False
    },
    "bec": {
        "name": "Business Email Compromise (CEO Wire Transfer)",
        "sender": "ceo-execoffice@company-partner-hq.org",
        "receiver": "cfo@company.com",
        "subject": "Confidential: Immediate Wire Transfer Required for Project Omega",
        "body": "Hi, I am currently in a board meeting and cannot take calls. We need to execute an urgent invoice payment of $48,500 to finalize the supplier acquisition. Please wire the funds to the account details attached in the link: http://wire-transfer-portal.finance-partner.net. Confirm once completed.",
        "spf": False,
        "dkim": False
    },
    "spam": {
        "name": "Commercial Marketing Spam",
        "sender": "exclusive-deals@bestmarketingoffers24.biz",
        "receiver": "user@gmail.com",
        "subject": "Claim Your 85% Discount Voucher - Limited Quantity Available!",
        "body": "Congratulations! You have been selected for our VIP seasonal clearance. Get up to 85% off all electronics, luxury watches, and smart gear. Click here to claim your coupon code now: http://deals.bestmarketingoffers24.biz/voucher-claim. Unsubscribe at http://optout.marketing24.biz",
        "spf": True,
        "dkim": False
    },
    "ham": {
        "name": "Legitimate Internal Work Email (HAM)",
        "sender": "sarah.jenkins@company.com",
        "receiver": "alex.morgan@company.com",
        "subject": "Sprint Planning & Architecture Sync - Tomorrow 10:00 AM",
        "body": "Hi Alex, Just following up on the quarterly roadmap. I have reviewed the pull request for the API gateway integration and left a few minor comments. Let's sync tomorrow at 10 AM to discuss the database migration plan. The agenda document is on your Google Drive folder.",
        "spf": True,
        "dkim": True
    }
}

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Email Security Shield — ALBERT + GIN + XAI Suite</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #070b14;
      --card-bg: rgba(15, 23, 42, 0.78);
      --card-border: rgba(255, 255, 255, 0.08);
      --text: #f3f4f6;
      --text-muted: #94a3b8;
      --primary: #6366f1;
      --primary-hover: #4f46e5;
      --phish: #ef4444;
      --phish-glow: rgba(239, 68, 68, 0.28);
      --ham: #10b981;
      --ham-glow: rgba(16, 185, 129, 0.25);
      --spam: #f59e0b;
      --spam-glow: rgba(245, 158, 11, 0.25);
      --accent: #38bdf8;
      --purple: #a855f7;
      --pink: #f43f5e;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
    body {
      background: radial-gradient(circle at 15% 10%, #111a33 0%, var(--bg) 75%);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      padding: 24px;
    }

    .container {
      max-width: 1400px;
      margin: 0 auto;
      width: 100%;
    }

    /* Header */
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--card-border);
      flex-wrap: wrap;
      gap: 16px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
    }
    .brand-icon {
      width: 48px;
      height: 48px;
      background: linear-gradient(135deg, #6366f1, #38bdf8);
      border-radius: 14px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 24px;
      box-shadow: 0 8px 24px rgba(99, 102, 241, 0.35);
    }
    .brand-title h1 {
      font-size: 22px;
      font-weight: 800;
      letter-spacing: -0.02em;
      background: linear-gradient(to right, #fff, #cbd5e1);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .brand-title p {
      font-size: 13px;
      color: var(--text-muted);
    }
    .header-badges {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .badge-status {
      display: flex;
      align-items: center;
      gap: 8px;
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.3);
      padding: 6px 14px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      color: #34d399;
    }
    .status-dot {
      width: 8px;
      height: 8px;
      background: #10b981;
      border-radius: 50%;
      box-shadow: 0 0 10px #10b981;
    }
    .badge-xai {
      display: flex;
      align-items: center;
      gap: 8px;
      background: rgba(56, 189, 248, 0.12);
      border: 1px solid rgba(56, 189, 248, 0.3);
      padding: 6px 14px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      color: #7dd3fc;
    }
    .badge-memory {
      display: flex;
      align-items: center;
      gap: 8px;
      background: rgba(168, 85, 247, 0.12);
      border: 1px solid rgba(168, 85, 247, 0.3);
      padding: 6px 14px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      color: #c084fc;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .badge-memory:hover {
      background: rgba(168, 85, 247, 0.22);
      transform: translateY(-1px);
    }

    /* Main Grid */
    .dashboard-grid {
      display: grid;
      grid-template-columns: 1.02fr 0.98fr;
      gap: 28px;
      align-items: start;
    }
    @media (max-width: 1024px) {
      .dashboard-grid { grid-template-columns: 1fr; }
    }

    /* Cards */
    .card {
      background: var(--card-bg);
      backdrop-filter: blur(20px);
      border: 1px solid var(--card-border);
      border-radius: 20px;
      padding: 26px;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.35);
    }
    .card-title {
      font-size: 16px;
      font-weight: 700;
      margin-bottom: 18px;
      display: flex;
      align-items: center;
      gap: 10px;
      color: #f1f5f9;
    }

    /* Presets bar */
    .presets-container {
      margin-bottom: 22px;
    }
    .presets-label {
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 10px;
      display: block;
    }
    .preset-pills {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .preset-btn {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--card-border);
      color: #cbd5e1;
      padding: 8px 14px;
      border-radius: 10px;
      font-size: 13px;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .preset-btn:hover {
      background: rgba(99, 102, 241, 0.15);
      border-color: rgba(99, 102, 241, 0.4);
      color: #fff;
      transform: translateY(-1px);
    }

    /* Form Fields */
    .form-group {
      margin-bottom: 18px;
    }
    .form-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }
    label {
      display: block;
      font-size: 13px;
      font-weight: 500;
      color: #cbd5e1;
      margin-bottom: 6px;
    }
    input[type="text"], textarea {
      width: 100%;
      background: rgba(10, 15, 28, 0.7);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 12px 14px;
      color: #fff;
      font-size: 14px;
      outline: none;
      transition: all 0.2s ease;
    }
    input[type="text"]:focus, textarea:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
    }
    textarea {
      resize: vertical;
      min-height: 140px;
      line-height: 1.5;
    }

    /* Switches */
    .switches-row {
      display: flex;
      gap: 20px;
      margin-bottom: 22px;
      background: rgba(0, 0, 0, 0.2);
      padding: 12px 16px;
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.04);
    }
    .switch-label {
      display: flex;
      align-items: center;
      gap: 10px;
      cursor: pointer;
      font-size: 13px;
      color: #cbd5e1;
      user-select: none;
    }
    .switch-label input {
      display: none;
    }
    .slider {
      width: 38px;
      height: 22px;
      background: #334155;
      border-radius: 999px;
      position: relative;
      transition: background 0.25s ease;
    }
    .slider::before {
      content: "";
      position: absolute;
      width: 16px;
      height: 16px;
      left: 3px;
      top: 3px;
      background: #fff;
      border-radius: 50%;
      transition: transform 0.25s ease;
    }
    input:checked + .slider {
      background: var(--primary);
    }
    input:checked + .slider::before {
      transform: translateX(16px);
    }

    /* Submit Button */
    .btn-analyze {
      width: 100%;
      background: linear-gradient(135deg, #6366f1, #3b82f6);
      color: #fff;
      border: none;
      padding: 15px;
      border-radius: 14px;
      font-size: 15px;
      font-weight: 700;
      letter-spacing: 0.02em;
      cursor: pointer;
      transition: all 0.25s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      box-shadow: 0 10px 25px rgba(99, 102, 241, 0.35);
    }
    .btn-analyze:hover {
      background: linear-gradient(135deg, #4f46e5, #2563eb);
      transform: translateY(-2px);
      box-shadow: 0 14px 30px rgba(99, 102, 241, 0.45);
    }
    .btn-analyze:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
    }

    /* Output Card States */
    .empty-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 480px;
      text-align: center;
      color: var(--text-muted);
    }
    .empty-icon {
      font-size: 54px;
      margin-bottom: 16px;
      opacity: 0.6;
    }

    /* Results Layout */
    .results-container {
      display: none;
      animation: fadeIn 0.35s ease forwards;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* Verdict Banner */
    .verdict-banner {
      border-radius: 16px;
      padding: 22px;
      margin-bottom: 22px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      position: relative;
      overflow: hidden;
    }
    .verdict-banner::before {
      content: "";
      position: absolute;
      top: 0; right: 0; bottom: 0; left: 0;
      background: linear-gradient(135deg, rgba(255,255,255,0.06), transparent);
    }
    .verdict-info h2 {
      font-size: 26px;
      font-weight: 800;
      letter-spacing: -0.01em;
      margin-bottom: 4px;
    }
    .verdict-info p {
      font-size: 13px;
      opacity: 0.85;
    }
    .risk-circle {
      width: 72px;
      height: 72px;
      border-radius: 50%;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      background: rgba(0, 0, 0, 0.45);
      border: 3px solid;
      font-weight: 800;
    }
    .risk-circle span {
      font-size: 18px;
    }
    .risk-circle small {
      font-size: 9px;
      text-transform: uppercase;
      opacity: 0.7;
    }

    /* Verdict themes */
    .theme-phishing {
      background: linear-gradient(135deg, #7f1d1d, #450a0a);
      border: 1px solid #dc2626;
      box-shadow: 0 10px 30px var(--phish-glow);
    }
    .theme-phishing .risk-circle { border-color: #ef4444; color: #fca5a5; }

    .theme-ham {
      background: linear-gradient(135deg, #064e3b, #022c22);
      border: 1px solid #059669;
      box-shadow: 0 10px 30px var(--ham-glow);
    }
    .theme-ham .risk-circle { border-color: #10b981; color: #6ee7b7; }

    .theme-spam {
      background: linear-gradient(135deg, #78350f, #451a03);
      border: 1px solid #d97706;
      box-shadow: 0 10px 30px var(--spam-glow);
    }
    .theme-spam .risk-circle { border-color: #f59e0b; color: #fcd34d; }

    /* Probabilities bars */
    .metric-section { margin-bottom: 14px; }
    .metric-header {
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      font-weight: 600;
      margin-bottom: 6px;
      color: #cbd5e1;
    }
    .bar-bg {
      width: 100%;
      height: 10px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 999px;
      overflow: hidden;
      position: relative;
    }
    .bar-fill {
      height: 100%;
      border-radius: 999px;
      transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .fill-phish { background: linear-gradient(90deg, #f87171, #ef4444); }
    .fill-ham   { background: linear-gradient(90deg, #34d399, #10b981); }
    .fill-spam  { background: linear-gradient(90deg, #fbbf24, #f59e0b); }

    /* Meta tags */
    .chips-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 14px;
      margin-bottom: 22px;
    }
    .chip {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 10px 14px;
    }
    .chip-label {
      font-size: 11px;
      color: var(--text-muted);
      text-transform: uppercase;
      font-weight: 600;
      margin-bottom: 4px;
    }
    .chip-val {
      font-size: 14px;
      font-weight: 600;
      color: #fff;
      font-family: 'JetBrains Mono', monospace;
    }

    /* XAI SECTION COMPONENTS */
    .xai-section-divider {
      margin: 24px 0 14px 0;
      padding-top: 16px;
      border-top: 1px solid rgba(255, 255, 255, 0.08);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .xai-section-title {
      font-size: 14px;
      font-weight: 700;
      color: #e2e8f0;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .xai-tag {
      font-size: 10px;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 6px;
      background: rgba(99, 102, 241, 0.2);
      color: #a5b4fc;
      border: 1px solid rgba(99, 102, 241, 0.4);
      text-transform: uppercase;
    }

    /* 1. Word Attribution Heatmap */
    .word-heatmap-box {
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 16px;
      margin-bottom: 16px;
      line-height: 2.0;
      max-height: 240px;
      overflow-y: auto;
    }
    .word-token {
      display: inline-block;
      padding: 2px 6px;
      margin: 2px 3px;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s ease;
      font-size: 13px;
      color: #fff;
      border: 1px solid transparent;
      user-select: none;
    }
    .word-token:hover {
      transform: scale(1.08);
      border-color: rgba(255, 255, 255, 0.4);
      z-index: 5;
    }
    .word-legend {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 11px;
      color: var(--text-muted);
      margin-top: 8px;
      padding: 0 4px;
    }
    .word-legend-items {
      display: flex;
      gap: 12px;
    }
    .legend-box {
      width: 12px;
      height: 12px;
      border-radius: 3px;
      display: inline-block;
      margin-right: 4px;
      vertical-align: middle;
    }
    .box-pos { background: rgba(239, 68, 68, 0.6); }
    .box-neg { background: rgba(16, 185, 129, 0.6); }
    .box-neu { background: rgba(148, 163, 184, 0.2); }
    .word-disclaimer {
      font-size: 11px;
      color: #94a3b8;
      font-style: italic;
      line-height: 1.4;
      margin-top: 8px;
      padding: 8px 12px;
      background: rgba(255, 255, 255, 0.02);
      border-radius: 8px;
    }

    /* 2. Modality Ablation Grid */
    .ablation-box {
      background: rgba(0, 0, 0, 0.28);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 16px;
      margin-bottom: 16px;
    }
    .ablation-grid {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 10px;
      margin-bottom: 12px;
    }
    .ablation-card {
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 12px;
      text-align: center;
    }
    .ablation-label {
      font-size: 11px;
      color: var(--text-muted);
      text-transform: uppercase;
      font-weight: 700;
      margin-bottom: 4px;
    }
    .ablation-delta {
      font-size: 16px;
      font-weight: 800;
      font-family: 'JetBrains Mono', monospace;
    }
    .delta-drop { color: #f87171; }
    .delta-none { color: #94a3b8; }
    .ablation-sub {
      font-size: 11px;
      color: var(--text-muted);
      margin-top: 2px;
    }
    .ablation-rationale {
      font-size: 12px;
      color: #cbd5e1;
      line-height: 1.45;
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid rgba(255, 255, 255, 0.06);
    }
    .gate-stat-badge {
      font-size: 11px;
      color: #a5b4fc;
      font-family: 'JetBrains Mono', monospace;
    }

    /* 3. Per-URL Attribution Table */
    .url-table-container {
      background: rgba(0, 0, 0, 0.28);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 12px 14px;
      margin-bottom: 16px;
      overflow-x: auto;
    }
    .url-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    .url-table th {
      text-align: left;
      padding: 8px 10px;
      color: var(--text-muted);
      text-transform: uppercase;
      font-size: 11px;
      font-weight: 700;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .url-table td {
      padding: 10px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      color: #e2e8f0;
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
    }

    /* 4. GNNExplainer Subgraph */
    .gnn-box {
      background: rgba(0, 0, 0, 0.28);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 14px 16px;
      margin-bottom: 16px;
    }
    .gnn-nodes-list {
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-bottom: 12px;
    }
    .gnn-node-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 8px 12px;
      background: rgba(255, 255, 255, 0.02);
      border-radius: 8px;
      font-size: 12px;
    }

    /* 5. Security Threat Badges */
    .threat-indicators-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-bottom: 16px;
    }
    .threat-indicator-card {
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .threat-indicator-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .threat-category-title {
      font-size: 13px;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .severity-pill {
      font-size: 10px;
      font-weight: 800;
      padding: 2px 8px;
      border-radius: 6px;
      text-transform: uppercase;
      font-family: 'JetBrains Mono', monospace;
    }
    .sev-critical { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .sev-high     { background: rgba(249, 115, 22, 0.2); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.4); }
    .sev-medium   { background: rgba(234, 179, 8, 0.2);  color: #fde047; border: 1px solid rgba(234, 179, 8, 0.4); }
    .sev-low      { background: rgba(56, 189, 248, 0.2); color: #7dd3fc; border: 1px solid rgba(56, 189, 248, 0.4); }
    .threat-evidence {
      font-size: 12px;
      color: #cbd5e1;
      line-height: 1.4;
    }

    /* 6. Conversational XAI Agent */
    .xai-chat-card {
      background: rgba(10, 15, 28, 0.7);
      border: 1px solid rgba(99, 102, 241, 0.3);
      border-radius: 16px;
      padding: 18px;
      margin-top: 14px;
    }
    .chat-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 12px;
    }
    .chat-header-title {
      font-size: 14px;
      font-weight: 700;
      color: #c7d2fe;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .suggested-prompts {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 12px;
    }
    .prompt-chip {
      background: rgba(99, 102, 241, 0.12);
      border: 1px solid rgba(99, 102, 241, 0.25);
      color: #c7d2fe;
      font-size: 11px;
      padding: 5px 10px;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .prompt-chip:hover {
      background: rgba(99, 102, 241, 0.25);
      border-color: rgba(99, 102, 241, 0.5);
      color: #fff;
      transform: translateY(-1px);
    }
    .chat-history {
      max-height: 220px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 10px;
      padding: 6px 0;
      margin-bottom: 12px;
    }
    .chat-bubble {
      max-width: 90%;
      padding: 10px 14px;
      border-radius: 12px;
      font-size: 13px;
      line-height: 1.45;
      word-wrap: break-word;
    }
    .bubble-user {
      align-self: flex-end;
      background: linear-gradient(135deg, #4f46e5, #6366f1);
      color: #fff;
      border-bottom-right-radius: 4px;
    }
    .bubble-agent {
      align-self: flex-start;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.08);
      color: #f1f5f9;
      border-bottom-left-radius: 4px;
    }
    .chat-input-row { display: flex; gap: 8px; }
    .chat-input-box {
      flex: 1;
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 10px 14px;
      color: #fff;
      font-size: 13px;
      outline: none;
    }
    .chat-input-box:focus { border-color: var(--primary); }
    .chat-send-btn {
      background: var(--primary);
      border: none;
      color: #fff;
      padding: 10px 16px;
      border-radius: 10px;
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
      transition: background 0.2s ease;
    }
    .chat-send-btn:hover { background: var(--primary-hover); }

    /* Spinner */
    .spinner {
      width: 20px;
      height: 20px;
      border: 3px solid rgba(255, 255, 255, 0.3);
      border-top-color: #fff;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      display: none;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* Vector Memory Styling */
    .memory-alert-box {
      background: rgba(168, 85, 247, 0.12);
      border: 1px solid rgba(168, 85, 247, 0.45);
      border-radius: 14px;
      padding: 14px 18px;
      margin-bottom: 16px;
      backdrop-filter: blur(10px);
      box-shadow: 0 4px 20px rgba(168, 85, 247, 0.15);
      animation: fadeIn 0.4s ease;
    }
    .memory-alert-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 8px;
    }
    .memory-alert-title {
      font-size: 13px;
      font-weight: 700;
      color: #e9d5ff;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .memory-match-badge {
      font-size: 11px;
      font-weight: 700;
      padding: 3px 10px;
      border-radius: 999px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .memory-match-phish { background: rgba(239, 68, 68, 0.25); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.5); }
    .memory-match-ham   { background: rgba(16, 185, 129, 0.25); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.5); }
    .memory-match-spam  { background: rgba(245, 158, 11, 0.25); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.5); }
    .memory-match-item {
      font-size: 12px;
      color: #cbd5e1;
      padding: 10px 12px;
      background: rgba(0, 0, 0, 0.35);
      border-radius: 8px;
      margin-top: 6px;
      border-left: 3px solid #c084fc;
      line-height: 1.5;
    }

    .memory-flag-card {
      background: rgba(168, 85, 247, 0.08);
      border: 1px solid rgba(168, 85, 247, 0.3);
      border-radius: 16px;
      padding: 16px;
      margin-top: 18px;
      margin-bottom: 18px;
    }
    .memory-flag-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 8px;
    }
    .memory-flag-title {
      font-size: 13px;
      font-weight: 700;
      color: #e9d5ff;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .memory-actions-row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .btn-flag {
      flex: 1;
      min-width: 140px;
      padding: 10px 14px;
      border-radius: 10px;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
      border: 1px solid transparent;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
    }
    .btn-flag:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
    .btn-flag-phish { background: rgba(239, 68, 68, 0.18); border-color: rgba(239, 68, 68, 0.4); color: #fca5a5; }
    .btn-flag-phish:hover { background: rgba(239, 68, 68, 0.32); color: #fff; }
    .btn-flag-ham   { background: rgba(16, 185, 129, 0.18); border-color: rgba(16, 185, 129, 0.4); color: #6ee7b7; }
    .btn-flag-ham:hover   { background: rgba(16, 185, 129, 0.32); color: #fff; }
    .btn-flag-spam  { background: rgba(245, 158, 11, 0.18); border-color: rgba(245, 158, 11, 0.4); color: #fcd34d; }
    .btn-flag-spam:hover  { background: rgba(245, 158, 11, 0.32); color: #fff; }

    /* Modal styling */
    .modal-overlay {
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0, 0, 0, 0.78);
      backdrop-filter: blur(8px);
      z-index: 9999;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }
    .modal-card {
      background: #0f172a;
      border: 1px solid rgba(168, 85, 247, 0.4);
      border-radius: 20px;
      width: 100%;
      max-width: 760px;
      max-height: 85vh;
      display: flex;
      flex-direction: column;
      box-shadow: 0 25px 50px rgba(0, 0, 0, 0.7);
      overflow: hidden;
    }
    .modal-header {
      padding: 18px 24px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: rgba(255, 255, 255, 0.02);
    }
    .modal-body {
      padding: 20px 24px;
      overflow-y: auto;
      flex: 1;
    }
    .modal-close {
      background: transparent;
      border: none;
      color: #94a3b8;
      font-size: 24px;
      line-height: 1;
      cursor: pointer;
      padding: 4px;
    }
    .modal-close:hover { color: #fff; }
    .memory-entry-card {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 12px 16px;
      margin-bottom: 10px;
      transition: border-color 0.2s ease;
    }
    .memory-entry-card:hover { border-color: rgba(168, 85, 247, 0.4); }
  </style>
</head>
<body>

  <div class="container">
    <header>
      <div class="brand">
        <div class="brand-icon">🛡️</div>
        <div class="brand-title">
          <h1>Email Security Shield</h1>
          <p>Dual-Branch ALBERT + GIN with Comprehensive XAI Suite</p>
        </div>
      </div>
      <div class="header-badges">
        <div class="badge-memory" id="headerMemoryBadge" onclick="openMemoryModal()" title="View Stored Zero-Shot Vector Threat Memory Bank">
          🧠 Threat Memory: <span id="headerMemoryCount">0</span>
        </div>
        <div class="badge-xai">
          🔍 XAI Explainability Active
        </div>
        <div class="badge-status">
          <span class="status-dot"></span> Neural Weights Active
        </div>
      </div>
    </header>

    <div class="dashboard-grid">
      <!-- Input Panel -->
      <div class="card">
        <div class="card-title">
          <span>✉️</span> Inspect Incoming Email
        </div>

        <div class="presets-container">
          <span class="presets-label">Quick Test Presets</span>
          <div class="preset-pills">
            <button type="button" class="preset-btn" onclick="loadPreset('phishing')">🎣 PayPal Phishing</button>
            <button type="button" class="preset-btn" onclick="loadPreset('bec')">💼 CEO Wire Fraud (BEC)</button>
            <button type="button" class="preset-btn" onclick="loadPreset('spam')">📢 Marketing Spam</button>
            <button type="button" class="preset-btn" onclick="loadPreset('ham')">✅ Project Meeting (HAM)</button>
          </div>
        </div>

        <form id="emailForm" onsubmit="analyzeEmail(event)">
          <div class="form-row form-group">
            <div>
              <label for="sender">Sender Email Address</label>
              <input type="text" id="sender" placeholder="e.g. alert@security-verify.com" required>
            </div>
            <div>
              <label for="receiver">Recipient Email Address</label>
              <input type="text" id="receiver" placeholder="e.g. user@company.com" required>
            </div>
          </div>

          <div class="form-group">
            <label for="subject">Email Subject Line</label>
            <input type="text" id="subject" placeholder="e.g. Immediate action required on your account" required>
          </div>

          <div class="form-group">
            <label for="body">Email Message Content (Body)</label>
            <textarea id="body" placeholder="Paste the full email text content or hyperlinks here..." required></textarea>
          </div>

          <div class="switches-row">
            <label class="switch-label">
              <input type="checkbox" id="spf">
              <span class="slider"></span>
              <span>SPF Authentication Passed</span>
            </label>
            <label class="switch-label">
              <input type="checkbox" id="dkim">
              <span class="slider"></span>
              <span>DKIM Signature Verified</span>
            </label>
          </div>

          <button type="submit" id="submitBtn" class="btn-analyze">
            <span class="spinner" id="spinner"></span>
            <span id="btnText">Analyze Email with ALBERT + GIN + XAI</span>
          </button>
        </form>
      </div>

      <!-- Output Panel -->
      <div class="card">
        <div class="card-title">
          <span>⚡</span> Multimodal Intelligence Verdict
        </div>

        <div class="empty-state" id="emptyState">
          <div class="empty-icon">🔍</div>
          <h3>Awaiting Email Input</h3>
          <p>Fill out the fields on the left or select a preset to compute real-time classification, word-level Integrated Gradients, controlled modality ablation, per-URL impact, and GNNExplainer graph features.</p>
        </div>

        <div class="results-container" id="resultsContainer">
          <!-- Zero-Shot Vector Threat Memory Match Alert -->
          <div id="memoryMatchAlert" class="memory-alert-box" style="display:none;"></div>

          <!-- Primary Verdict Banner -->
          <div class="verdict-banner" id="verdictBanner">
            <div class="verdict-info">
              <h2 id="verdictText">PHISHING</h2>
              <p id="verdictDesc">Detected with high semantic and structural risk</p>
            </div>
            <div class="risk-circle" id="riskCircle">
              <span id="riskScore">98%</span>
              <small>Risk</small>
            </div>
          </div>

          <!-- Probabilities -->
          <div class="card-title" style="margin-top: 10px; font-size: 14px;">
            <span>📊</span> Primary Category Confidence
          </div>

          <div class="metric-section">
            <div class="metric-header">
              <span>Phishing Probability</span>
              <span id="phishVal">0%</span>
            </div>
            <div class="bar-bg"><div class="bar-fill fill-phish" id="phishBar" style="width: 0%"></div></div>
          </div>

          <div class="metric-section">
            <div class="metric-header">
              <span>Legitimate (HAM) Probability</span>
              <span id="hamVal">0%</span>
            </div>
            <div class="bar-bg"><div class="bar-fill fill-ham" id="hamBar" style="width: 0%"></div></div>
          </div>

          <div class="metric-section">
            <div class="metric-header">
              <span>Commercial Spam Probability</span>
              <span id="spamVal">0%</span>
            </div>
            <div class="bar-bg"><div class="bar-fill fill-spam" id="spamBar" style="width: 0%"></div></div>
          </div>

          <!-- Structural Chips -->
          <div class="chips-grid">
            <div class="chip">
              <div class="chip-label">Phishing Subtype</div>
              <div class="chip-val" id="subtypeVal">-</div>
            </div>
            <div class="chip">
              <div class="chip-label">Sender Domain</div>
              <div class="chip-val" id="domainVal">-</div>
            </div>
            <div class="chip">
              <div class="chip-label">Hyperlinks Detected</div>
              <div class="chip-val" id="urlsVal">0 URLs</div>
            </div>
            <div class="chip">
              <div class="chip-label">Entity Graph Nodes</div>
              <div class="chip-val" id="nodesVal">0 Nodes</div>
            </div>
          </div>

          <!-- Human-in-the-Loop Zero-Shot Vector Archival Card -->
          <div class="memory-flag-card">
            <div class="memory-flag-header">
              <div class="memory-flag-title">
                <span>🧠</span> Zero-Shot Vector Memory & Human-in-the-Loop Flagging
              </div>
              <span class="xai-tag" style="background:rgba(168,85,247,0.2);color:#d8b4fe;border-color:rgba(168,85,247,0.4);">Permanent Store</span>
            </div>
            <p style="font-size:12px;color:var(--text-muted);margin-bottom:12px;line-height:1.45;">
              Catalog this email's 256-dimensional multimodal embedding (<code style="color:#c084fc;font-family:monospace;">z_fused</code>) directly into persistent on-disk memory. When future campaign variants or whitelisted emails appear, the system recognizes them via vector similarity without model retraining.
            </p>
            <div style="display:flex;gap:8px;margin-bottom:10px;">
              <input type="text" id="memoryNoteInput" placeholder="Optional analyst note (e.g. 'Payroll spear-phishing campaign variant', 'Trusted vendor notice')" style="flex:1;background:rgba(0,0,0,0.35);border:1px solid var(--card-border);border-radius:10px;padding:8px 12px;color:#fff;font-size:12px;">
            </div>
            <div class="memory-actions-row">
              <button type="button" class="btn-flag btn-flag-phish" onclick="flagEmailToMemory('PHISHING')">
                🚨 Flag as PHISHING
              </button>
              <button type="button" class="btn-flag btn-flag-ham" onclick="flagEmailToMemory('HAM')">
                ✅ Flag as Legitimate (HAM)
              </button>
              <button type="button" class="btn-flag btn-flag-spam" onclick="flagEmailToMemory('SPAM')">
                ⚠️ Flag as SPAM
              </button>
            </div>
            <div id="memorySaveStatus" style="display:none;margin-top:10px;font-size:12px;border-radius:8px;padding:10px 14px;"></div>
          </div>

          <!-- ============================================================= -->
          <!-- 1. WORD-LEVEL ATTRIBUTIONS (INTEGRATED GRADIENTS ON ALBERT) -->
          <!-- ============================================================= -->
          <div class="xai-section-divider">
            <div class="xai-section-title">
              <span>🔤</span> Word-Level Attributions (ALBERT Branch)
            </div>
            <span class="xai-tag">Integrated Gradients</span>
          </div>

          <div class="word-heatmap-box" id="wordHeatmapContainer">
            <!-- Dynamically populated with highlighted word tokens -->
          </div>
          <div class="word-legend">
            <div class="word-legend-items">
              <span><span class="legend-box box-pos"></span> Supports Predicted Class</span>
              <span><span class="legend-box box-neg"></span> Opposes Class (Authentic)</span>
              <span><span class="legend-box box-neu"></span> Neutral / Insignificant</span>
            </div>
            <span id="wordCountLabel">0 words evaluated</span>
          </div>
          <div class="word-disclaimer">
            Highlighted words indicate how the text model's output changes with respect to its input representation. They are model attributions, not proof that a word is malicious.
          </div>

          <!-- ============================================================= -->
          <!-- 2. CONTROLLED BRANCH-LEVEL MODALITY ABLATION -->
          <!-- ============================================================= -->
          <div class="xai-section-divider">
            <div class="xai-section-title">
              <span>⚖️</span> Modality Attribution (Controlled Branch Ablation)
            </div>
            <span class="xai-tag">Causal Ablation</span>
          </div>

          <div class="ablation-box">
            <div class="ablation-grid">
              <div class="ablation-card">
                <div class="ablation-label">Text Branch Ablated</div>
                <div class="ablation-delta" id="ablationDeltaText">-</div>
                <div class="ablation-sub" id="ablationProbText">Prob: -</div>
              </div>
              <div class="ablation-card">
                <div class="ablation-label">Graph Branch Ablated</div>
                <div class="ablation-delta" id="ablationDeltaGraph">-</div>
                <div class="ablation-sub" id="ablationProbGraph">Prob: -</div>
              </div>
              <div class="ablation-card">
                <div class="ablation-label">Joint Both Ablated</div>
                <div class="ablation-delta" id="ablationDeltaJoint">-</div>
                <div class="ablation-sub" id="ablationProbJoint">Prob: -</div>
              </div>
            </div>
            <div class="ablation-rationale" id="ablationInterpretationText">
              Evaluating controlled branch-level ablation...
            </div>
            <div style="margin-top:8px;font-size:11px;color:#94a3b8;display:flex;justify-content:space-between;">
              <span>Baseline: Neutral ALBERT [CLS] & null isolated GIN graph</span>
              <span class="gate-stat-badge" id="gateStatBadge">Gate α: -</span>
            </div>
          </div>

          <!-- ============================================================= -->
          <!-- 3. PER-URL PERTURBATION ATTRIBUTION -->
          <!-- ============================================================= -->
          <div class="xai-section-divider">
            <div class="xai-section-title">
              <span>🔗</span> Per-URL Relational & Linguistic Impact
            </div>
            <span class="xai-tag">Per-URL Sensitivity</span>
          </div>

          <div class="url-table-container">
            <div id="urlTableWrapper">
              <!-- Dynamically populated table of URLs -->
            </div>
          </div>

          <!-- ============================================================= -->
          <!-- 4. GNNEXPLAINER GRAPH ENTITY & EDGE ATTRIBUTION -->
          <!-- ============================================================= -->
          <div class="xai-section-divider">
            <div class="xai-section-title">
              <span>🕸️</span> GNNExplainer Subgraph & Feature Attribution
            </div>
            <span class="xai-tag">GIN Explainer</span>
          </div>

          <div class="gnn-box">
            <div style="font-size:12px;font-weight:700;color:#cbd5e1;margin-bottom:8px;">Top Influential Graph Entities:</div>
            <div class="gnn-nodes-list" id="gnnNodesList">
              <!-- Dynamically populated -->
            </div>
            <div style="font-size:12px;font-weight:700;color:#cbd5e1;margin-top:12px;margin-bottom:8px;">Key Relational Graph Edges:</div>
            <div class="gnn-nodes-list" id="gnnEdgesList">
              <!-- Dynamically populated -->
            </div>
          </div>

          <!-- ============================================================= -->
          <!-- 5. SECURITY THREAT INDICATORS -->
          <!-- ============================================================= -->
          <div class="xai-section-divider">
            <div class="xai-section-title">
              <span>🚨</span> Semantic Threat Indicators
            </div>
            <span class="xai-tag">Behavioral Flags</span>
          </div>
          <div class="threat-indicators-list" id="threatIndicatorsList">
            <!-- Dynamically populated -->
          </div>

          <!-- ============================================================= -->
          <!-- 6. INTERACTIVE CONVERSATIONAL ANALYST AGENT -->
          <!-- ============================================================= -->
          <div class="xai-chat-card">
            <div class="chat-header">
              <div class="chat-header-title">
                <span>🤖</span> Ask XAI Security Analyst
              </div>
              <span style="font-size:11px;color:#a5b4fc;font-weight:600;">100% Offline AI</span>
            </div>

            <div class="suggested-prompts">
              <div class="prompt-chip" onclick="askPresetQuestion('Text vs graph ablation?')">⚖️ Text vs Graph ablation?</div>
              <div class="prompt-chip" onclick="askPresetQuestion('Which words influenced the decision?')">🔤 Influential words?</div>
              <div class="prompt-chip" onclick="askPresetQuestion('Explain the URL impact breakdown')">🔗 URL impact breakdown?</div>
              <div class="prompt-chip" onclick="askPresetQuestion('Which graph entities contributed most?')">🕸️ GNN graph entities?</div>
              <div class="prompt-chip" onclick="askPresetQuestion('Check vector threat memory?')">🧠 Threat memory match?</div>
            </div>

            <div class="chat-history" id="chatHistory">
              <div class="chat-bubble bubble-agent">
                👋 Hello! I am your explainable email security analyst. Once you analyze an email, you can inspect word-level heatmaps, branch ablations, per-URL impacts, or ask me detailed questions!
              </div>
            </div>

            <div class="chat-input-row">
              <input type="text" id="chatInput" class="chat-input-box" placeholder="Ask a question about this classification..." onkeydown="handleChatKey(event)">
              <button type="button" class="chat-send-btn" id="chatSendBtn" onclick="sendChatMessage()">Send</button>
            </div>
          </div>

        </div>
      </div>
    </div>
  </div>

  <!-- Persistent Vector Memory Bank Modal -->
  <div class="modal-overlay" id="memoryModal" onclick="if(event.target===this) closeMemoryModal()">
    <div class="modal-card">
      <div class="modal-header">
        <div style="display:flex;align-items:center;gap:12px;">
          <span style="font-size:24px;">🧠</span>
          <div>
            <h3 style="color:#f1f5f9;font-size:16px;margin:0;">Persistent Vector Memory Bank</h3>
            <span style="font-size:11px;color:#94a3b8;">Zero-Shot Representation Storage for Rapid Attack Recall</span>
          </div>
        </div>
        <button type="button" class="modal-close" onclick="closeMemoryModal()">&times;</button>
      </div>
      <div class="modal-body" id="memoryModalBody">
        <!-- populated dynamically -->
      </div>
    </div>
  </div>

  <script>
    const PRESETS = """ + json.dumps(PRESETS) + """;
    let lastAnalysisResult = null;
    let lastPayload = null;

    function loadPreset(key) {
      const p = PRESETS[key];
      if (!p) return;
      document.getElementById('sender').value = p.sender;
      document.getElementById('receiver').value = p.receiver;
      document.getElementById('subject').value = p.subject;
      document.getElementById('body').value = p.body;
      document.getElementById('spf').checked = p.spf;
      document.getElementById('dkim').checked = p.dkim;
    }

    async function analyzeEmail(e) {
      e.preventDefault();
      const btn = document.getElementById('submitBtn');
      const spinner = document.getElementById('spinner');
      const btnText = document.getElementById('btnText');

      btn.disabled = true;
      spinner.style.display = 'block';
      btnText.innerText = 'Extracting Graph & Computing XAI Suite...';

      const payload = {
        sender: document.getElementById('sender').value,
        receiver: document.getElementById('receiver').value,
        subject: document.getElementById('subject').value,
        body: document.getElementById('body').value,
        has_spf: document.getElementById('spf').checked,
        has_dkim: document.getElementById('dkim').checked
      };
      lastPayload = payload;

      try {
        const response = await fetch('/api/classify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const res = await response.json();
        lastAnalysisResult = { ...res, ...payload };
        renderResults(res);
      } catch (err) {
        alert('Error communicating with classification backend: ' + err);
      } finally {
        btn.disabled = false;
        spinner.style.display = 'none';
        btnText.innerText = 'Analyze Email with ALBERT + GIN + XAI';
      }
    }

    function renderResults(res) {
      document.getElementById('emptyState').style.display = 'none';
      document.getElementById('resultsContainer').style.display = 'block';

      const banner = document.getElementById('verdictBanner');
      banner.className = 'verdict-banner';

      const verdict = res.primary_label;
      document.getElementById('verdictText').innerText = verdict;

      if (verdict === 'PHISHING') {
        banner.classList.add('theme-phishing');
        document.getElementById('verdictDesc').innerText = `Classified as Phishing (${(res.primary_confidence * 100).toFixed(1)}% confidence)`;
      } else if (verdict === 'HAM') {
        banner.classList.add('theme-ham');
        document.getElementById('verdictDesc').innerText = `Legitimate authentic communication (${(res.primary_confidence * 100).toFixed(1)}% confidence)`;
      } else {
        banner.classList.add('theme-spam');
        document.getElementById('verdictDesc').innerText = `Commercial bulk or unsolicited spam (${(res.primary_confidence * 100).toFixed(1)}% confidence)`;
      }

      document.getElementById('riskScore').innerText = Math.round(res.phishing_risk_score) + '%';

      // Probabilities Bars
      const phishProb = (res.primary_probabilities['PHISHING'] || 0) * 100;
      const hamProb = (res.primary_probabilities['HAM'] || 0) * 100;
      const spamProb = (res.primary_probabilities['SPAM'] || 0) * 100;

      document.getElementById('phishVal').innerText = phishProb.toFixed(1) + '%';
      document.getElementById('phishBar').style.width = phishProb + '%';

      document.getElementById('hamVal').innerText = hamProb.toFixed(1) + '%';
      document.getElementById('hamBar').style.width = hamProb + '%';

      document.getElementById('spamVal').innerText = spamProb.toFixed(1) + '%';
      document.getElementById('spamBar').style.width = spamProb + '%';

      // Structural Chips
      document.getElementById('subtypeVal').innerText = res.subtype_label;
      document.getElementById('domainVal').innerText = res.graph_metadata.sender_domain || 'None';
      document.getElementById('urlsVal').innerText = `${res.graph_metadata.detected_urls} URL(s)`;
      document.getElementById('nodesVal').innerText = `${res.graph_metadata.num_nodes} Nodes`;

      // Zero-Shot Vector Memory Match Alert
      const alertBox = document.getElementById('memoryMatchAlert');
      const matches = res.memory_matches || [];
      if (matches.length > 0) {
        let matchHtml = `
          <div class="memory-alert-header">
            <div class="memory-alert-title">
              <span>🧠</span> <strong>Zero-Shot Threat Memory Match (${matches.length} Campaign Match${matches.length > 1 ? 'es' : ''})</strong>
            </div>
            <span class="memory-match-badge ${matches[0].label === 'PHISHING' ? 'memory-match-phish' : (matches[0].label === 'HAM' ? 'memory-match-ham' : 'memory-match-spam')}">
              ${matches[0].similarity_percent}% SIMILARITY TO ${matches[0].label}
            </span>
          </div>
          <p style="font-size:12px;color:#cbd5e1;margin-bottom:8px;">
            This email's 256-dimensional multimodal embedding (<code>z_fused</code>) matches a vector previously saved in persistent memory.
          </p>
        `;
        matches.forEach(m => {
          matchHtml += `
            <div class="memory-match-item">
              <strong>Vector #${m.id} [${m.label}]</strong> (${m.similarity_percent}% Match) &bull; <em>"${m.subject}"</em> from <code>${m.sender}</code><br>
              <span style="font-size:11px;color:#94a3b8;">📅 Cataloged: ${m.timestamp} ${m.note ? ' &bull; 📝 ' + m.note : ''}</span>
            </div>
          `;
        });
        alertBox.innerHTML = matchHtml;
        alertBox.style.display = 'block';
      } else {
        alertBox.style.display = 'none';
      }

      // Reset Memory Flagging Box
      const saveStatus = document.getElementById('memorySaveStatus');
      if (saveStatus) saveStatus.style.display = 'none';
      const noteInput = document.getElementById('memoryNoteInput');
      if (noteInput) noteInput.value = '';

      // Update header memory count
      if (res.memory_count !== undefined) {
        document.getElementById('headerMemoryCount').innerText = res.memory_count;
      }

      // Render XAI Components
      const bundle = res.xai_bundle || res.xai;
      if (bundle) {
        renderXAI(bundle);
      }
    }

    function renderXAI(bundle) {
      // 1. Word Attribution Heatmap (ALBERT via Integrated Gradients)
      const wordsContainer = document.getElementById('wordHeatmapContainer');
      wordsContainer.innerHTML = '';
      const words = bundle.word_attributions || [];
      document.getElementById('wordCountLabel').innerText = `${words.length} words evaluated`;

      if (words.length === 0) {
        wordsContainer.innerHTML = '<span style="color:#94a3b8;font-size:13px;">No text tokens evaluated.</span>';
      } else {
        words.forEach(w => {
          const span = document.createElement('span');
          span.className = 'word-token';
          span.innerText = w.word;

          const nScore = w.normalized_score || 0;
          let bgColor = 'rgba(255, 255, 255, 0.04)';
          let titleText = `Word: "${w.word}" | Score: ${nScore.toFixed(4)}`;

          if (nScore > 0.05) {
            const alpha = 0.15 + (Math.min(1.0, nScore) * 0.65);
            bgColor = `rgba(239, 68, 68, ${alpha})`;
            titleText += ' (Pushes toward Predicted Class)';
          } else if (nScore < -0.05) {
            const alpha = 0.15 + (Math.min(1.0, Math.abs(nScore)) * 0.65);
            bgColor = `rgba(16, 185, 129, ${alpha})`;
            titleText += ' (Opposes Class / Supports Authentic)';
          }

          span.style.background = bgColor;
          span.title = titleText;
          wordsContainer.appendChild(span);
        });
      }

      // 2. Modality Attribution (Controlled Branch Ablation)
      const ablation = bundle.modality_attribution || {};
      const dText = ablation.delta_text_ablation !== undefined ? ablation.delta_text_ablation : 0;
      const dGraph = ablation.delta_graph_ablation !== undefined ? ablation.delta_graph_ablation : 0;
      const dJoint = ablation.delta_joint_ablation !== undefined ? ablation.delta_joint_ablation : 0;

      document.getElementById('ablationDeltaText').innerText = (dText >= 0 ? '+' : '') + dText.toFixed(4);
      document.getElementById('ablationDeltaText').className = 'ablation-delta ' + (dText > 0.01 ? 'delta-drop' : 'delta-none');
      document.getElementById('ablationProbText').innerText = `Prob: ${(ablation.text_ablated_probability || 0).toFixed(4)}`;

      document.getElementById('ablationDeltaGraph').innerText = (dGraph >= 0 ? '+' : '') + dGraph.toFixed(4);
      document.getElementById('ablationDeltaGraph').className = 'ablation-delta ' + (dGraph > 0.01 ? 'delta-drop' : 'delta-none');
      document.getElementById('ablationProbGraph').innerText = `Prob: ${(ablation.graph_ablated_probability || 0).toFixed(4)}`;

      document.getElementById('ablationDeltaJoint').innerText = (dJoint >= 0 ? '+' : '') + dJoint.toFixed(4);
      document.getElementById('ablationDeltaJoint').className = 'ablation-delta ' + (dJoint > 0.01 ? 'delta-drop' : 'delta-none');
      document.getElementById('ablationProbJoint').innerText = `Prob: ${(ablation.joint_ablated_probability || 0).toFixed(4)}`;

      document.getElementById('ablationInterpretationText').innerText = ablation.interpretation || '';
      const gateMean = ablation.gate_statistics ? ablation.gate_statistics.mean_alpha : 0.5;
      document.getElementById('gateStatBadge').innerText = `Internal Gate α: ${gateMean.toFixed(4)}`;

      // 3. Per-URL Attribution Table
      const urlWrapper = document.getElementById('urlTableWrapper');
      urlWrapper.innerHTML = '';
      const urlAttrs = bundle.url_attribution || [];

      if (urlAttrs.length === 0) {
        urlWrapper.innerHTML = '<div style="color:#94a3b8;font-size:13px;padding:8px 0;">No external hyperlinks detected in email.</div>';
      } else {
        let tableHtml = `
          <table class="url-table">
            <thead>
              <tr>
                <th>Destination Domain</th>
                <th>Text-Only ΔP</th>
                <th>Graph-Only ΔP</th>
                <th>Combined ΔP</th>
                <th>Risk Impact</th>
              </tr>
            </thead>
            <tbody>
        `;
        urlAttrs.forEach(u => {
          const isRisk = u.impact_direction === 'INCREASES_RISK';
          const badgeStyle = isRisk ? 'color:#f87171;' : 'color:#34d399;';
          tableHtml += `
            <tr>
              <td><span title="${u.url}">${u.destination_domain || u.url.substring(0, 30)}</span></td>
              <td>${(u.delta_text_only >= 0 ? '+' : '') + u.delta_text_only.toFixed(4)}</td>
              <td>${(u.delta_graph_only >= 0 ? '+' : '') + u.delta_graph_only.toFixed(4)}</td>
              <td><strong>${(u.delta_combined >= 0 ? '+' : '') + u.delta_combined.toFixed(4)}</strong></td>
              <td style="${badgeStyle}"><strong>${u.impact_direction}</strong></td>
            </tr>
          `;
        });
        tableHtml += '</tbody></table>';
        urlWrapper.innerHTML = tableHtml;
      }

      // 4. GNNExplainer Subgraph & Entity Rankings
      const gnn = bundle.gnn_explanation || {};
      const nodesList = document.getElementById('gnnNodesList');
      nodesList.innerHTML = '';
      const topNodes = (gnn.node_importance || []).slice(0, 4);

      if (topNodes.length === 0) {
        nodesList.innerHTML = '<div style="color:#94a3b8;font-size:12px;">No graph entities available.</div>';
      } else {
        topNodes.forEach(n => {
          nodesList.innerHTML += `
            <div class="gnn-node-row">
              <span style="color:#fff;">${n.entity}</span>
              <span style="font-family:'JetBrains Mono';font-weight:700;color:#c084fc;">Importance: ${n.percentage}</span>
            </div>
          `;
        });
      }

      const edgesList = document.getElementById('gnnEdgesList');
      edgesList.innerHTML = '';
      const topEdges = (gnn.edge_importance || []).slice(0, 3);
      if (topEdges.length === 0) {
        edgesList.innerHTML = '<div style="color:#94a3b8;font-size:12px;">No active relational edges.</div>';
      } else {
        topEdges.forEach(e => {
          edgesList.innerHTML += `
            <div class="gnn-node-row">
              <span style="color:#cbd5e1;">${e.source} ↔ ${e.target}</span>
              <span style="font-family:'JetBrains Mono';font-weight:700;color:#38bdf8;">Weight: ${e.percentage}</span>
            </div>
          `;
        });
      }

      // 5. Threat Indicators
      const threatList = document.getElementById('threatIndicatorsList');
      threatList.innerHTML = '';
      const indicators = bundle.threat_indicators || [];
      if (indicators.length === 0) {
        threatList.innerHTML = `
          <div class="threat-indicator-card" style="text-align:center;color:#94a3b8;font-size:13px;padding:14px;">
            ✅ No critical semantic threat signatures flagged in text content.
          </div>
        `;
      } else {
        indicators.forEach(ind => {
          let sevClass = 'sev-low';
          const lvl = (ind.level || '').toUpperCase();
          if (lvl === 'CRITICAL') sevClass = 'sev-critical';
          else if (lvl === 'HIGH') sevClass = 'sev-high';
          else if (lvl === 'MEDIUM') sevClass = 'sev-medium';

          let triggerDetails = '';
          if (ind.triggers && ind.triggers.length > 0) {
            triggerDetails = `<div style="font-size:11px;color:#a5b4fc;margin-top:2px;">Triggers: <strong>${ind.triggers.join(', ')}</strong></div>`;
          }

          threatList.innerHTML += `
            <div class="threat-indicator-card">
              <div class="threat-indicator-header">
                <span class="threat-category-title">⚠️ ${ind.title || ind.category}</span>
                <span class="severity-pill ${sevClass}">${ind.level || 'ALERT'}</span>
              </div>
              <div class="threat-evidence">${ind.description || ''}</div>
              ${triggerDetails}
            </div>
          `;
        });
      }

      // Reset Analyst Chat
      const chatHistory = document.getElementById('chatHistory');
      chatHistory.innerHTML = `
        <div class="chat-bubble bubble-agent">
          🔍 <strong>Classification Explained:</strong> This email was classified as <strong>${lastAnalysisResult.primary_label}</strong> (${(lastAnalysisResult.primary_confidence * 100).toFixed(1)}% confidence). ${ablation.interpretation || 'Multimodal features were evaluated.'} What would you like to know about this verdict?
        </div>
      `;
    }

    async function sendChatMessage(customQuestion = null) {
      const input = document.getElementById('chatInput');
      const question = customQuestion || input.value.trim();
      if (!question) return;

      if (!customQuestion) input.value = '';

      const chatHistory = document.getElementById('chatHistory');
      chatHistory.innerHTML += `
        <div class="chat-bubble bubble-user">${question}</div>
      `;
      chatHistory.scrollTop = chatHistory.scrollHeight;

      if (!lastAnalysisResult) {
        chatHistory.innerHTML += `
          <div class="chat-bubble bubble-agent">Please inspect an email above first so I can analyze its specific neural features!</div>
        `;
        return;
      }

      const tempId = 'agent-typing-' + Date.now();
      chatHistory.innerHTML += `
        <div class="chat-bubble bubble-agent" id="${tempId}">Analyzing neural features, word attributions, and graph topology...</div>
      `;
      chatHistory.scrollTop = chatHistory.scrollHeight;

      try {
        const resp = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question: question,
            email_data: lastPayload,
            prediction: lastAnalysisResult,
            xai_bundle: lastAnalysisResult.xai_bundle || lastAnalysisResult.xai
          })
        });
        const data = await resp.json();
        const typingElem = document.getElementById(tempId);
        if (typingElem) {
          typingElem.innerHTML = data.answer.replace(/\\n/g, '<br>');
        }
      } catch (err) {
        const typingElem = document.getElementById(tempId);
        if (typingElem) {
          typingElem.innerText = "Error answering question: " + err;
        }
      }
      chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function askPresetQuestion(q) {
      sendChatMessage(q);
    }

    function handleChatKey(e) {
      if (e.key === 'Enter') {
        sendChatMessage();
      }
    }

    // ==========================================
    // ZERO-SHOT VECTOR MEMORY SYSTEM FUNCTIONS
    // ==========================================
    async function flagEmailToMemory(flagLabel) {
      if (!lastAnalysisResult || !lastAnalysisResult.embedding) {
        alert('Please analyze an email first before flagging.');
        return;
      }
      const noteInput = document.getElementById('memoryNoteInput');
      const statusDiv = document.getElementById('memorySaveStatus');
      const note = noteInput ? noteInput.value.trim() : '';

      try {
        const resp = await fetch('/api/memory/add', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            embedding: lastAnalysisResult.embedding,
            sender: lastPayload.sender || 'Unknown',
            subject: lastPayload.subject || 'No Subject',
            label: flagLabel,
            subtype: lastAnalysisResult.subtype_label || 'CUSTOM',
            note: note || `Manually verified as ${flagLabel} by analyst`
          })
        });
        const data = await resp.json();
        if (data.success) {
          statusDiv.style.display = 'block';
          statusDiv.style.background = 'rgba(168, 85, 247, 0.18)';
          statusDiv.style.border = '1px solid rgba(168, 85, 247, 0.5)';
          statusDiv.style.color = '#e9d5ff';
          statusDiv.innerHTML = `✅ <strong>Vector #${data.entry.id} saved permanently to Vector Memory!</strong> Stored vectors: <strong>${data.count}</strong>. Any future incoming email with a matching 256-dim embedding will instantly trigger a Zero-Shot <strong>${flagLabel}</strong> similarity match!`;
          document.getElementById('headerMemoryCount').innerText = data.count;
        } else {
          throw new Error(data.error || 'Server error saving vector');
        }
      } catch (err) {
        statusDiv.style.display = 'block';
        statusDiv.style.background = 'rgba(239, 68, 68, 0.15)';
        statusDiv.style.border = '1px solid rgba(239, 68, 68, 0.4)';
        statusDiv.style.color = '#fca5a5';
        statusDiv.innerHTML = `❌ Failed to save vector: ${err.message}`;
      }
    }

    async function loadMemoryCount() {
      try {
        const resp = await fetch('/api/memory/count');
        const data = await resp.json();
        if (data.count !== undefined) {
          document.getElementById('headerMemoryCount').innerText = data.count;
        }
      } catch (e) {}
    }

    async function openMemoryModal() {
      const modal = document.getElementById('memoryModal');
      modal.style.display = 'flex';
      const body = document.getElementById('memoryModalBody');
      body.innerHTML = '<div style="color:#94a3b8;text-align:center;padding:24px;">Loading stored vector representations...</div>';

      try {
        const resp = await fetch('/api/memory/list');
        const data = await resp.json();
        const entries = data.entries || [];
        if (entries.length === 0) {
          body.innerHTML = `
            <div style="text-align:center;padding:40px 20px;color:#94a3b8;">
              <div style="font-size:36px;margin-bottom:10px;">🧠</div>
              <h3 style="color:#e2e8f0;margin-bottom:6px;">Vector Memory Bank is Empty</h3>
              <p style="font-size:13px;max-width:440px;margin:0 auto;line-height:1.5;">
                No emails have been cataloged yet. After analyzing an email, click <strong>Flag as PHISHING / HAM / SPAM</strong> to permanently commit its 256-dimensional embedding into on-disk storage.
              </p>
            </div>
          `;
          return;
        }

        let html = `
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
            <span style="font-size:13px;color:#cbd5e1;"><strong>${entries.length}</strong> vector(s) cataloged on disk (<code>output/memory/</code>)</span>
            <button type="button" onclick="clearMemoryBank()" style="background:rgba(239,68,68,0.2);border:1px solid rgba(239,68,68,0.4);color:#fca5a5;padding:6px 12px;border-radius:8px;font-size:11px;font-weight:700;cursor:pointer;">Clear All Memory</button>
          </div>
        `;

        entries.forEach(e => {
          let badgeClass = 'memory-match-phish';
          if (e.label === 'HAM') badgeClass = 'memory-match-ham';
          else if (e.label === 'SPAM') badgeClass = 'memory-match-spam';

          html += `
            <div class="memory-entry-card">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                <span style="font-weight:700;color:#f1f5f9;font-size:13px;">Vector #${e.id} &bull; ${e.subject || 'No Subject'}</span>
                <span class="memory-match-badge ${badgeClass}">${e.label}</span>
              </div>
              <div style="font-size:12px;color:#94a3b8;margin-bottom:4px;">
                Sender: <code style="color:#cbd5e1;">${e.sender}</code> &bull; Subtype: <code style="color:#c084fc;">${e.subtype || 'CUSTOM'}</code>
              </div>
              <div style="font-size:11px;color:#64748b;display:flex;justify-content:space-between;">
                <span>${e.note ? '📝 ' + e.note : ''}</span>
                <span>📅 ${e.timestamp}</span>
              </div>
            </div>
          `;
        });
        body.innerHTML = html;
      } catch (err) {
        body.innerHTML = `<div style="color:#f87171;padding:20px;">Error loading memory: ${err}</div>`;
      }
    }

    function closeMemoryModal() {
      document.getElementById('memoryModal').style.display = 'none';
    }

    async function clearMemoryBank() {
      if (!confirm('Are you sure you want to clear all stored vectors from memory? This cannot be undone.')) return;
      try {
        const resp = await fetch('/api/memory/clear', { method: 'POST' });
        const data = await resp.json();
        document.getElementById('headerMemoryCount').innerText = '0';
        openMemoryModal();
      } catch (err) {
        alert('Failed to clear memory: ' + err);
      }
    }

    // Load stored count on startup
    document.addEventListener('DOMContentLoaded', loadMemoryCount);
  </script>
</body>
</html>
"""


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler providing HTML GUI and REST API endpoints."""
    protocol_version = "HTTP/1.1"

    def _send_json(self, status_code, data_dict):
        payload = json.dumps(data_dict).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        global classifier
        if classifier is None:
            classifier = EmailClassifierInference()

        if self.path == "/" or self.path.startswith("/?"):
            html_bytes = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(html_bytes)
        elif self.path == "/api/presets":
            self._send_json(200, PRESETS)
        elif self.path == "/api/memory/list":
            entries = classifier.memory.list_all(100)
            self._send_json(200, {"count": classifier.memory.count(), "entries": entries})
        elif self.path == "/api/memory/count":
            self._send_json(200, {"count": classifier.memory.count()})
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()

    def do_POST(self):
        global classifier
        if classifier is None:
            classifier = EmailClassifierInference()

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        if self.path == "/api/classify":
            try:
                data = json.loads(post_data.decode("utf-8"))
                result = classifier.predict(
                    subject=data.get("subject", ""),
                    body=data.get("body", ""),
                    sender=data.get("sender", "user@example.com"),
                    receiver=data.get("receiver", "user@example.com"),
                    has_spf=bool(data.get("has_spf", False)),
                    has_dkim=bool(data.get("has_dkim", False)),
                    return_xai=True
                )
                self._send_json(200, result)
            except Exception as e:
                self._send_json(500, {"error": str(e)})

        elif self.path == "/api/chat":
            try:
                data = json.loads(post_data.decode("utf-8"))
                question = data.get("question", "")
                email_data = data.get("email_data", {})
                prediction = data.get("prediction", {})
                xai_bundle = data.get("xai_bundle", {})
                answer = conversational_xai_agent(
                    question=question,
                    xai_bundle=xai_bundle,
                    email_data=email_data,
                    prediction=prediction
                )
                self._send_json(200, {"answer": answer})
            except Exception as e:
                self._send_json(500, {"error": str(e)})

        elif self.path == "/api/memory/add":
            try:
                data = json.loads(post_data.decode("utf-8"))
                entry = classifier.flag_email_in_memory(
                    embedding=data.get("embedding", []),
                    sender=data.get("sender", "Unknown"),
                    subject=data.get("subject", "No subject"),
                    label=data.get("label", "PHISHING"),
                    subtype=data.get("subtype", "CUSTOM"),
                    note=data.get("note", "")
                )
                self._send_json(200, {"success": True, "count": classifier.memory.count(), "entry": entry})
            except Exception as e:
                self._send_json(500, {"error": str(e)})

        elif self.path == "/api/memory/clear":
            try:
                classifier.memory.clear()
                self._send_json(200, {"success": True, "count": 0})
            except Exception as e:
                self._send_json(500, {"error": str(e)})

        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()

    def log_message(self, format, *args):
        try:
            sys.stderr.write(f"[Server] {format % args}\n")
        except Exception:
            pass


def open_browser(port):
    """Automatically launches the user's default web browser reliably on Windows."""
    url = f"http://127.0.0.1:{port}"
    try:
        if sys.platform == "win32":
            import subprocess
            subprocess.Popen(f'start {url}', shell=True)
        else:
            webbrowser.open(url)
    except Exception:
        try:
            webbrowser.open(url)
        except Exception:
            pass


def main():
    global classifier
    print("=" * 70)
    print("   INITIALIZING ALBERT + GIN DASHBOARD WITH COMPLETE XAI PIPELINE")
    print("=" * 70)
    print("Loading model weights from ./output/checkpoints/best_model.pt...")
    classifier = EmailClassifierInference()
    print("XAI Suite initialized: Branch Ablation, Per-URL Perturbation, Integrated Gradients & GNNExplainer.")

    port = PORT
    server = None
    for p in range(PORT, PORT + 20):
        try:
            server = ThreadedHTTPServer(("127.0.0.1", p), DashboardRequestHandler)
            port = p
            break
        except OSError:
            continue

    if server is None:
        raise RuntimeError(f"Could not bind to any port from {PORT} to {PORT+20}")

    url = f"http://127.0.0.1:{port}"
    print(f"\n>>> Web GUI is live at: {url}")
    print("Launching default browser automatically...")
    print("Tip: You can also hold Ctrl and click the link above directly in the terminal.")
    print("Press Ctrl+C in this terminal anytime to stop the server.\n")

    threading.Timer(0.5, open_browser, args=(port,)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Email Security Dashboard server...")
        server.server_close()
        print("Server shutdown cleanly.")


if __name__ == "__main__":
    main()
