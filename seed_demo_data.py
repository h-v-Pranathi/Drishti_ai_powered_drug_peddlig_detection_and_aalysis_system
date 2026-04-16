"""
DRISHTI — Demo Data Seeder
Generates a realistic synthetic dataset so the dashboard can be explored
immediately without needing Reddit API access.

Run:
    python seed_demo_data.py
"""

import sys
from pathlib import Path

# Ensure we can import from scripts/
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from dashboard import generate_demo_data, DEMO_PATH
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

RAW_PATH = DATA_DIR / "raw_posts.csv"

def seed():
    print("=" * 60)
    print("DRISHTI — Seeding demo dataset (2 000 posts)")
    print("=" * 60)
    df = generate_demo_data(n=2000)
    df.to_csv(RAW_PATH, index=False)
    df.to_csv(DEMO_PATH, index=False)
    print(f"  Raw data  → {RAW_PATH}")
    print(f"  Demo data → {DEMO_PATH}")
    print("\nYou can now either:")
    print("  1. Run the full pipeline:  python run_pipeline.py")
    print("  2. Launch dashboard directly (uses demo data):  streamlit run dashboard.py")

if __name__ == "__main__":
    seed()
