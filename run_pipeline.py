"""
DRISHTI — Full Pipeline Runner
Executes all processing steps in sequence:
  1. Preprocess raw posts
  2. Detect codewords & score
  3. Train classifier & predict
  4. Cluster users

Run this after collect_live_data.py has gathered some data,
or whenever you want to refresh the processed dataset.
"""

import sys
import os
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

def run_step(name: str, fn):
    print(f"\n{'─'*60}")
    print(f"  STEP: {name}")
    print(f"{'─'*60}")
    try:
        fn()
        print(f"  ✓ {name} complete.")
    except Exception as e:
        print(f"  ✗ {name} FAILED: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 60)
    print("  DRISHTI — Full Pipeline")
    print("=" * 60)

    import preprocess_dataset
    import detect_codewords
    import train_classification_model
    import cluster_users

    run_step("Text Preprocessing",        preprocess_dataset.run)
    run_step("Codeword Detection",        detect_codewords.run)
    run_step("ML Classification",         train_classification_model.run)
    run_step("User Behavior Clustering",  cluster_users.run)

    print("\n" + "=" * 60)
    print("  Pipeline complete. Run: streamlit run dashboard.py")
    print("=" * 60)
