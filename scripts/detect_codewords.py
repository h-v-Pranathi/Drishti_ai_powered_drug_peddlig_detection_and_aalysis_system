"""
DRISHTI — Codeword Detection Module
Scans processed text for suspicious patterns across four signal categories:
  • Selling signals
  • Buyer signals
  • Secrecy signals
  • Emoji signals

Outputs:
  detected_codewords  — JSON list of matched patterns
  suspicion_score     — normalised float 0–1
  is_suspected        — bool (score ≥ 0.35)
  explanation         — human-readable reason string
"""

import re
import json
import os
import ast
import pandas as pd
from pathlib import Path

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE        = Path(__file__).resolve().parent.parent
CLEAN_PATH  = BASE / "data" / "processed_posts.csv"
SCORED_PATH = BASE / "data" / "scored_posts.csv"

# ─── Signal dictionaries ─────────────────────────────────────────────────────

SELL_SIGNALS = [
    r"\bplug\b",
    r"\bdealer\b",
    r"\bavailable\b",
    r"\bbulk\b",
    r"\bpacket\b",
    r"\bsupply\b",
    r"\bstock\b",
    r"\bgrade\b",
    r"\bdelivery\b",
    r"\bdrop\b",
    r"\bselling\b",
    r"\bsell\b",
    r"\bservice running\b",
    r"\bhook up\b",
    r"\bconnect\b",
]

BUY_SIGNALS = [
    r"\bprice\b",
    r"\bhow much\b",
    r"\bneed\b",
    r"\blooking for\b",
    r"\bwhere (to|can)\b",
    r"\bany one selling\b",
    r"\bwho (has|sells)\b",
    r"\basap\b",
    r"\burgently\b",
    r"\bprice check\b",
]

SECRECY_SIGNALS = [
    r"\bdm only\b",
    r"\binbox me\b",
    r"\btrusted only\b",
    r"\bno face no case\b",
    r"\bcash only\b",
    r"\bdiscrete\b",
    r"\blow key\b",
    r"\bprivate only\b",
    r"\bno cops\b",
    r"\bno drama\b",
]

DRUG_SIGNALS = [
    r"\bweed\b",
    r"\bcannabis\b",
    r"\bmdma\b",
    r"\bcocaine\b",
    r"\bcoke\b",
    r"\bheroin\b",
    r"\bmeth\b",
    r"\bpills\b",
    r"\bshrooms\b",
    r"\blsd\b",
    r"\bxan(s|ax)?\b",
    r"\bpercs?\b",
    r"\boxy\b",
]

EMOJI_SIGNALS = {
    "❄️": 0.12,   # snow — cocaine slang
    "💊": 0.12,   # pill
    "📦": 0.08,   # package
    "💰": 0.07,   # money
    "🍬": 0.08,   # candy — drug slang
    "🔥": 0.05,   # fire — quality indicator
    "🌿": 0.08,   # herb — cannabis
    "🧪": 0.10,   # lab — synthesis
    "🎲": 0.04,
    "🤫": 0.06,   # shush — secrecy
}

# ─── Score weights ────────────────────────────────────────────────────────────
W_SELL    = 0.30
W_BUY     = 0.20
W_SECRECY = 0.35
W_DRUG    = 0.35
W_EMOJI   = 0.20   # total weight for emoji bucket

THRESHOLD = 0.20   # is_suspected if score ≥ this — needs ~2 meaningful signal hits


def _match_signals(text: str, patterns: list[str]) -> list[str]:
    """Return list of matched pattern strings."""
    matched = []
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            matched.append(pat.replace(r"\b", "").strip())
    return matched


def _emoji_score(text: str) -> tuple[float, list[str]]:
    """Return (weighted_score, matched_emoji_list)."""
    found = []
    raw_score = 0.0
    for emoji, weight in EMOJI_SIGNALS.items():
        if emoji in text:
            found.append(emoji)
            raw_score += weight
    # cap at W_EMOJI
    return min(raw_score, W_EMOJI), found


def score_text(text: str) -> dict:
    """
    Compute suspicion signals for a single piece of text.
    Returns a dict with score, codewords, flag, and explanation.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "detected_codewords": [],
            "suspicion_score":    0.0,
            "is_suspected":       False,
            "explanation":        "Empty text.",
        }

    sell_hits    = _match_signals(text, SELL_SIGNALS)
    buy_hits     = _match_signals(text, BUY_SIGNALS)
    secrecy_hits = _match_signals(text, SECRECY_SIGNALS)
    drug_hits    = _match_signals(text, DRUG_SIGNALS)
    emoji_score, emoji_hits = _emoji_score(text)

    # Per-hit additive scoring — each matched signal adds a fixed weight.
    # Old formula: (hits/total_patterns)*weight diluted scores so badly that
    # "available + dm only + trusted clients" only scored 0.065, never reaching
    # any meaningful threshold.
    sell_score    = min(len(sell_hits)    * 0.12, 0.35)
    buy_score     = min(len(buy_hits)     * 0.10, 0.25)
    secrecy_score = min(len(secrecy_hits) * 0.15, 0.40)
    drug_score    = min(len(drug_hits)    * 0.20, 0.50)

    total = sell_score + buy_score + secrecy_score + drug_score + emoji_score

    all_hits = sell_hits + buy_hits + secrecy_hits + drug_hits + emoji_hits

    # Build explanation
    parts = []
    if sell_hits:    parts.append(f"selling signals: {', '.join(sell_hits)}")
    if buy_hits:     parts.append(f"buyer signals: {', '.join(buy_hits)}")
    if secrecy_hits: parts.append(f"secrecy signals: {', '.join(secrecy_hits)}")
    if drug_hits:    parts.append(f"substance references: {', '.join(drug_hits)}")
    if emoji_hits:   parts.append(f"suspicious emojis: {' '.join(emoji_hits)}")
    explanation = ("; ".join(parts)) if parts else "No suspicious patterns detected."

    return {
        "detected_codewords": all_hits,
        "suspicion_score":    round(min(total, 1.0), 4),
        "is_suspected":       total >= THRESHOLD,
        "explanation":        explanation,
    }


def run():
    print("=" * 60)
    print("DRISHTI — Codeword Detection & Suspicion Scoring")
    print("=" * 60)

    if not CLEAN_PATH.exists():
        print(f"[ERROR] Preprocessed data not found at {CLEAN_PATH}")
        print("  Run preprocess_dataset.py first.")
        return

    df = pd.read_csv(CLEAN_PATH, dtype=str)
    print(f"  Loaded {len(df)} posts.")

    # Use processed_text if available and non-empty, else fall back to post_text
    text_col = df["processed_text"].fillna("").where(
        df["processed_text"].fillna("").str.strip() != "",
        df.get("post_text", df["processed_text"]).fillna(""),
    )
    results = text_col.apply(score_text)
    df["detected_codewords"] = results.apply(lambda r: json.dumps(r["detected_codewords"]))
    df["suspicion_score"]    = results.apply(lambda r: r["suspicion_score"])
    df["is_suspected"]       = results.apply(lambda r: r["is_suspected"])
    df["explanation"]        = results.apply(lambda r: r["explanation"])

    suspected_n = df["is_suspected"].sum()
    print(f"  Suspected posts: {suspected_n} / {len(df)} "
          f"({100*suspected_n/max(len(df),1):.1f}%)")

    SCORED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SCORED_PATH, index=False)
    print(f"  Saved → {SCORED_PATH}")


if __name__ == "__main__":
    run()