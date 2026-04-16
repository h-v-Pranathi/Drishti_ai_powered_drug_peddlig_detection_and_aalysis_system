"""
DRISHTI — Text Preprocessing Module
Cleans raw post text while preserving emojis and semantic content.
Adds `processed_text` column to the dataset.
"""

import re
import os
import pandas as pd
from pathlib import Path

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE       = Path(__file__).resolve().parent.parent
RAW_PATH   = BASE / "data" / "raw_posts.csv"
CLEAN_PATH = BASE / "data" / "processed_posts.csv"

# ─── Regex patterns ──────────────────────────────────────────────────────────
URL_RE    = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE  = re.compile(r"\S+@\S+\.\S+")
SPACE_RE  = re.compile(r"\s+")
# Keep: letters, digits, common punctuation, emoji (anything outside basic ASCII is kept)
STRIP_RE  = re.compile(r"[^\w\s\U00010000-\U0010ffff\U00002600-\U000027BF.,!?'\"@#$%&*()-]", re.UNICODE)


def clean_text(text: str) -> str:
    """
    Pipeline:
    1. Lowercase
    2. Remove URLs
    3. Remove email addresses
    4. Strip special characters (preserve emojis & punctuation)
    5. Normalize whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = STRIP_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning and add `processed_text` column."""
    df = df.copy()
    df["processed_text"] = df["post_text"].apply(clean_text)
    # Drop rows where cleaning yielded empty text
    df = df[df["processed_text"].str.len() > 0].reset_index(drop=True)
    return df


def run():
    print("=" * 60)
    print("DRISHTI — Text Preprocessing")
    print("=" * 60)

    if not RAW_PATH.exists():
        print(f"[ERROR] Raw data not found at {RAW_PATH}")
        print("  Run collect_live_data.py first.")
        return

    df = pd.read_csv(RAW_PATH, dtype=str)
    print(f"  Loaded {len(df)} raw posts.")

    df = preprocess(df)
    print(f"  {len(df)} posts after cleaning.")

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"  Saved → {CLEAN_PATH}")


if __name__ == "__main__":
    run()
