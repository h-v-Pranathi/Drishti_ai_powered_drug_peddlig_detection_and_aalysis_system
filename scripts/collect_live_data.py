"""
DRISHTI — Data Ingestion Module
Collects live public posts from Reddit using the JSON API.
Appends new posts to the dataset, deduplicates, and caps at 10,000 rows.
"""

import requests
import pandas as pd
import random
import time
import hashlib
import os
from datetime import datetime

# ─── Output path ────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw_posts.csv")
MAX_ROWS   = 10_000
SLEEP_SECS = 180  # collect every 3 minutes

# ─── Predefined Indian geographic locations ─────────────────────────────────
LOCATIONS = [
    ("Karnataka",    "Bengaluru",   "Whitefield",         12.9699,  77.7490),
    ("Karnataka",    "Bengaluru",   "Koramangala",         12.9352,  77.6245),
    ("Maharashtra",  "Mumbai",      "Bandra",              19.0553,  72.8400),
    ("Maharashtra",  "Mumbai",      "Andheri",             19.1197,  72.8464),
    ("Maharashtra",  "Pune",        "Kothrud",             18.5074,  73.8077),
    ("Delhi",        "New Delhi",   "Connaught Place",     28.6315,  77.2167),
    ("Delhi",        "New Delhi",   "Lajpat Nagar",        28.5673,  77.2373),
    ("Tamil Nadu",   "Chennai",     "T Nagar",             13.0422,  80.2337),
    ("Tamil Nadu",   "Chennai",     "Anna Nagar",          13.0850,  80.2101),
    ("West Bengal",  "Kolkata",     "Park Street",         22.5535,  88.3522),
    ("Telangana",    "Hyderabad",   "Banjara Hills",       17.4126,  78.4477),
    ("Telangana",    "Hyderabad",   "Hitech City",         17.4435,  78.3772),
    ("Gujarat",      "Ahmedabad",   "Navrangpura",         23.0395,  72.5610),
    ("Rajasthan",    "Jaipur",      "C-Scheme",            26.9124,  75.7873),
    ("Uttar Pradesh","Lucknow",     "Hazratganj",          26.8467,  80.9462),
    ("Punjab",       "Chandigarh",  "Sector 17",           30.7414,  76.7682),
    ("Madhya Pradesh","Bhopal",     "MP Nagar",            23.2332,  77.4316),
    ("Kerala",       "Kochi",       "MG Road",             9.9858,   76.2789),
    ("Assam",        "Guwahati",    "Paltan Bazaar",       26.1847,  91.7462),
    ("Bihar",        "Patna",       "Fraser Road",         25.6093,  85.1376),
]

# ─── Reddit search queries ───────────────────────────────────────────────────
SEARCH_TERMS = [
    "plug available dm only trusted",
    "supply available bulk order cash only",
    "grade available no face no case inbox",
    "looking for plug price check dm",
    "dealer active area delivery possible",
    "stock available packet ready trusted clients",
]

REDDIT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (DRISHTI-Research-Bot/1.0)"
}


def fetch_reddit_posts(query: str, limit: int = 25) -> list[dict]:
    """Fetch public Reddit posts matching query via JSON API."""
    url = f"https://www.reddit.com/search.json?q={requests.utils.quote(query)}&limit={limit}&sort=new"
    try:
        resp = requests.get(url, headers=REDDIT_HEADERS, timeout=15)
        resp.raise_for_status()
        children = resp.json()["data"]["children"]
        posts = []
        for child in children:
            d = child["data"]
            text = (d.get("selftext") or d.get("title") or "").strip()
            if not text or len(text) < 10:
                continue
            # deterministic post_id from Reddit fullname
            post_id = d.get("name", hashlib.md5(text.encode()).hexdigest()[:12])
            user_id  = "u_" + hashlib.md5(
                d.get("author", "anon").encode()
            ).hexdigest()[:8]
            # Use collection time (now) not Reddit's original post time.
            # Reddit's created_utc causes all posts in a batch to have similar
            # old timestamps, making time-window filters show identical counts.
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            loc = random.choice(LOCATIONS)
            posts.append({
                "post_id":    post_id,
                "user_id":    user_id,
                "post_text":  text,
                "timestamp":  ts,
                "platform":   "Reddit",
                "data_source": "reddit_json_api",
                "language":   "en",
                "state":      loc[0],
                "city":       loc[1],
                "area":       loc[2],
                "latitude":   loc[3],
                "longitude":  loc[4],
            })
        return posts
    except Exception as exc:
        print(f"[WARN] Reddit fetch failed: {exc}")
        return []


def load_existing(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path, dtype=str)
    cols = [
        "post_id","user_id","post_text","timestamp",
        "platform","data_source","language",
        "state","city","area","latitude","longitude",
    ]
    return pd.DataFrame(columns=cols)


def save_dataset(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def collect_once() -> int:
    """Run a single collection pass. Returns number of new rows added."""
    existing = load_existing(DATA_PATH)
    known_ids = set(existing["post_id"].tolist()) if not existing.empty else set()

    new_rows = []
    for query in SEARCH_TERMS:
        posts = fetch_reddit_posts(query, limit=25)
        for p in posts:
            if p["post_id"] not in known_ids:
                known_ids.add(p["post_id"])
                new_rows.append(p)
        time.sleep(2)  # be polite to Reddit

    if not new_rows:
        print(f"[{datetime.now():%H:%M:%S}] No new posts found.")
        return 0

    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    # Keep latest MAX_ROWS rows
    if len(combined) > MAX_ROWS:
        combined = combined.tail(MAX_ROWS).reset_index(drop=True)

    save_dataset(combined, DATA_PATH)
    print(f"[{datetime.now():%H:%M:%S}] Added {len(new_rows)} posts. Total: {len(combined)}")
    return len(new_rows)


def run_continuous():
    """Collect indefinitely every SLEEP_SECS seconds."""
    print("=" * 60)
    print("DRISHTI — Live Data Ingestion")
    print(f"Collecting every {SLEEP_SECS}s → {DATA_PATH}")
    print("=" * 60)
    while True:
        collect_once()
        print(f"  Next collection in {SLEEP_SECS}s …")
        time.sleep(SLEEP_SECS)


if __name__ == "__main__":
    run_continuous()