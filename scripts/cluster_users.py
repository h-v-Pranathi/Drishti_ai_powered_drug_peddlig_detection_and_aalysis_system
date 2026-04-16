"""
DRISHTI — User Behavior Clustering Module
Aggregates per-user behavioral metrics, then clusters with KMeans (k=3).

Cluster meanings:
  0 → Normal
  1 → Buyer-dominant
  2 → Seller-dominant

Adds `cluster_id` and `cluster_label` to the classified posts dataset,
and saves a per-user profile CSV.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE            = Path(__file__).resolve().parent.parent
CLASSIFIED_PATH = BASE / "data" / "classified_posts.csv"
CLUSTER_PATH    = BASE / "data" / "clustered_posts.csv"
USER_PROF_PATH  = BASE / "data" / "user_profiles.csv"
CLUSTER_MODEL   = BASE / "models" / "kmeans.pkl"

N_CLUSTERS = 3


def build_user_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per user:
      post_count          — total posts
      avg_suspicion_score — mean suspicion score
      seller_ratio        — fraction classified Seller
      buyer_ratio         — fraction classified Buyer
      activity_span       — days between first and last post
      most_common_city    — modal city
      latitude / longitude — mean coords
    """
    df = df.copy()
    df["suspicion_score"] = pd.to_numeric(df["suspicion_score"], errors="coerce").fillna(0)
    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    profiles = []
    for user_id, grp in df.groupby("user_id"):
        post_count = len(grp)
        avg_score  = grp["suspicion_score"].mean()

        cls_col = "ml_predicted_class" if "ml_predicted_class" in grp.columns else "label"
        classes = grp.get(cls_col, pd.Series(dtype=str)).fillna("Normal")
        seller_ratio = (classes == "Seller").sum() / post_count
        buyer_ratio  = (classes == "Buyer").sum()  / post_count

        ts = grp["timestamp"].dropna()
        if len(ts) >= 2:
            span = (ts.max() - ts.min()).days
        else:
            span = 0

        city = grp["city"].mode().iloc[0] if not grp["city"].mode().empty else "Unknown"
        lat  = grp["latitude"].mean()
        lon  = grp["longitude"].mean()

        profiles.append({
            "user_id":            user_id,
            "post_count":         post_count,
            "avg_suspicion_score": round(avg_score, 4),
            "seller_ratio":       round(seller_ratio, 4),
            "buyer_ratio":        round(buyer_ratio, 4),
            "activity_span_days": span,
            "city":               city,
            "latitude":           round(lat, 4) if pd.notna(lat) else 0.0,
            "longitude":          round(lon, 4) if pd.notna(lon) else 0.0,
        })

    return pd.DataFrame(profiles)


CLUSTER_LABELS = {0: "Normal", 1: "Buyer", 2: "Seller"}


def assign_cluster_label(cluster_id: int, seller_ratio: float, buyer_ratio: float) -> str:
    """
    Post-hoc relabelling: map cluster index to semantic name by inspecting
    per-cluster mean seller/buyer ratios so labels remain meaningful even if
    KMeans assigns cluster indices in a different order across runs.
    """
    return CLUSTER_LABELS.get(cluster_id, "Normal")


def run():
    print("=" * 60)
    print("DRISHTI — User Behavior Clustering")
    print("=" * 60)

    if not CLASSIFIED_PATH.exists():
        print(f"[ERROR] Classified data not found at {CLASSIFIED_PATH}")
        print("  Run train_classification_model.py first.")
        return

    df = pd.read_csv(CLASSIFIED_PATH, dtype=str)
    print(f"  Loaded {len(df)} posts from {df['user_id'].nunique()} users.")

    # ── Build per-user profiles ──────────────────────────────────────────────
    profiles = build_user_profiles(df)
    print(f"  Built {len(profiles)} user profiles.")

    # ── Feature matrix for clustering ────────────────────────────────────────
    feature_cols = [
        "post_count", "avg_suspicion_score",
        "seller_ratio", "buyer_ratio", "activity_span_days",
    ]
    X = profiles[feature_cols].fillna(0).values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── KMeans ───────────────────────────────────────────────────────────────
    n = min(N_CLUSTERS, len(profiles))
    km = KMeans(n_clusters=n, random_state=42, n_init=10)
    profiles["cluster_id"] = km.fit_predict(X_scaled)

    # ── Semantic relabelling ─────────────────────────────────────────────────
    # Find which cluster has highest seller_ratio → "Seller"
    # Find which cluster has highest buyer_ratio  → "Buyer"
    # Remainder                                   → "Normal"
    cluster_stats = (
        profiles.groupby("cluster_id")[["seller_ratio","buyer_ratio"]]
        .mean()
    )
    seller_cluster = int(cluster_stats["seller_ratio"].idxmax())
    buyer_cluster  = int(cluster_stats["buyer_ratio"].idxmax())
    normal_cluster = [i for i in range(n) if i not in (seller_cluster, buyer_cluster)]
    normal_cluster = normal_cluster[0] if normal_cluster else seller_cluster

    label_map = {
        seller_cluster: "Seller",
        buyer_cluster:  "Buyer",
        normal_cluster: "Normal",
    }
    profiles["cluster_label"] = profiles["cluster_id"].map(label_map).fillna("Normal")

    # ── Save user profiles ────────────────────────────────────────────────────
    USER_PROF_PATH.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(USER_PROF_PATH, index=False)
    print(f"  User profiles saved → {USER_PROF_PATH}")

    # ── Merge cluster info back to posts ─────────────────────────────────────
    # Drop existing cluster columns if present (e.g. from demo data)
    for col in ["cluster_id", "cluster_label"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    cluster_map = profiles.set_index("user_id")[["cluster_id","cluster_label"]]
    df = df.merge(cluster_map, on="user_id", how="left")
    df["cluster_id"]    = df["cluster_id"].fillna(0).astype(int)
    df["cluster_label"] = df["cluster_label"].fillna("Normal")

    CLUSTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLUSTER_PATH, index=False)

    dist = profiles["cluster_label"].value_counts().to_dict()
    print(f"  Cluster distribution (users): {dist}")
    print(f"  Clustered posts saved → {CLUSTER_PATH}")

    # ── Persist KMeans model ──────────────────────────────────────────────────
    CLUSTER_MODEL.parent.mkdir(parents=True, exist_ok=True)
    with open(CLUSTER_MODEL, "wb") as f:
        pickle.dump({"km": km, "scaler": scaler, "label_map": label_map}, f)
    print(f"  KMeans model saved → {CLUSTER_MODEL}")


if __name__ == "__main__":
    run()
