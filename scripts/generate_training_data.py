"""
DRISHTI — Realistic Training Data Generator v2

The key insight: to get 70-85% accuracy, the SAME words must appear
across different classes so the model can't just memorise word-class mappings.

Strategy:
 - Shared vocabulary pool (words like "available", "need", "looking", "price",
   "good", "area", "dm", "today" appear in ALL classes)
 - Posts are built from templates with shared slot words
 - Only the COMBINATION of words signals the class, not individual words
 - 20% of posts are deliberately mislabelled (ambiguous cases)
"""

import random
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

BASE      = Path(__file__).resolve().parent.parent
OUT_PATH  = BASE / "data" / "raw_posts.csv"
DEMO_PATH = BASE / "data" / "demo_data.csv"

LOCATIONS = [
    ("Karnataka",     "Bengaluru",  "Whitefield",      12.9699, 77.7490),
    ("Karnataka",     "Bengaluru",  "Koramangala",     12.9352, 77.6245),
    ("Maharashtra",   "Mumbai",     "Bandra",          19.0553, 72.8400),
    ("Maharashtra",   "Mumbai",     "Andheri",         19.1197, 72.8464),
    ("Maharashtra",   "Pune",       "Kothrud",         18.5074, 73.8077),
    ("Delhi",         "New Delhi",  "Connaught Place", 28.6315, 77.2167),
    ("Delhi",         "New Delhi",  "Lajpat Nagar",    28.5673, 77.2373),
    ("Tamil Nadu",    "Chennai",    "T Nagar",         13.0422, 80.2337),
    ("West Bengal",   "Kolkata",    "Park Street",     22.5535, 88.3522),
    ("Telangana",     "Hyderabad",  "Banjara Hills",   17.4126, 78.4477),
    ("Gujarat",       "Ahmedabad",  "Navrangpura",     23.0395, 72.5610),
    ("Rajasthan",     "Jaipur",     "C-Scheme",        26.9124, 75.7873),
    ("Uttar Pradesh", "Lucknow",    "Hazratganj",      26.8467, 80.9462),
    ("Punjab",        "Chandigarh", "Sector 17",       30.7414, 76.7682),
    ("Kerala",        "Kochi",      "MG Road",          9.9858, 76.2789),
    ("Assam",         "Guwahati",   "Paltan Bazaar",   26.1847, 91.7462),
    ("Bihar",         "Patna",      "Fraser Road",     25.6093, 85.1376),
    ("Madhya Pradesh","Bhopal",     "MP Nagar",        23.2332, 77.4316),
]

# Shared "neutral" words that appear across all classes
CITIES    = ["Bengaluru", "Mumbai", "Delhi", "Pune", "Chennai", "Hyderabad",
             "Kolkata", "Jaipur", "Chandigarh", "Kochi", "Lucknow"]
AREA      = random.choice(["area", "zone", "locality", "neighbourhood", "side"])
TIMING    = ["today", "now", "asap", "this week", "tonight", "tomorrow",
             "right now", "immediately", "soon", "at the moment"]
QUALITY   = ["good", "decent", "quality", "reliable", "consistent",
             "premium", "regular", "standard", "decent quality"]
CONTACT   = ["dm", "message", "inbox", "text", "contact", "reach out",
             "hit me up", "ping me", "drop a message"]

# Seller-specific signals (used in COMBINATION with shared words)
SELL_SIGNALS   = ["available", "supply", "stock", "bulk", "packet",
                  "delivery", "drop", "sorted", "plug", "dealer",
                  "grade A", "fresh batch"]
SELL_SECRECY   = ["dm only", "cash only", "trusted only", "no face no case",
                  "inbox only", "private", "low key", "discrete"]

# Buyer-specific signals
BUY_SIGNALS    = ["looking for", "need", "where to find", "anyone know",
                  "how much", "price check", "trying to find",
                  "anyone got", "who has", "can someone"]
BUY_URGENCY    = ["urgent", "asap", "urgently", "immediately",
                  "desperate", "really need", "badly need"]

# Normal topics (completely innocent)
NORMAL_TOPICS = [
    "restaurant recommendation", "gym membership", "plumber needed",
    "dentist recommendation", "school admission", "laptop repair",
    "house painting", "car service", "power cut", "traffic update",
    "property prices", "grocery store", "metro timings", "bus route",
    "tuition classes", "cooking classes", "photography course",
    "yoga classes", "swimming pool", "cricket academy",
]


def make_seller_post() -> str:
    """Combine shared + seller-specific vocabulary."""
    city    = random.choice(CITIES)
    quality = random.choice(QUALITY)
    timing  = random.choice(TIMING)
    contact = random.choice(CONTACT)
    signal  = random.choice(SELL_SIGNALS)
    secrecy = random.choice(SELL_SECRECY) if random.random() < 0.6 else ""

    area = random.choice(AREA)
    templates = [
        f"{quality} {signal} in {city} {area} {timing} {contact} {secrecy}",
        f"{signal} {timing} {quality} {contact} for details {secrecy}",
        f"{city} {area} {signal} {quality} {timing} {contact} {secrecy}",
        f"{quality} stuff {signal} {timing} {contact} {secrecy}",
        f"{signal} {city} {quality} rates {contact} {secrecy}",
        f"Good {signal} {timing} {city} {contact} {secrecy}",
        f"{secrecy} {signal} {quality} {timing} {city} {contact}",
    ]

    post = random.choice(templates).strip()
    # clean up double spaces
    return " ".join(post.split())


def make_buyer_post() -> str:
    """Combine shared + buyer-specific vocabulary."""
    city    = random.choice(CITIES)
    quality = random.choice(QUALITY)
    timing  = random.choice(TIMING)
    contact = random.choice(CONTACT)
    signal  = random.choice(BUY_SIGNALS)
    urgency = random.choice(BUY_URGENCY) if random.random() < 0.5 else ""

    templates = [
        f"{signal} {quality} stuff in {city} {urgency} {contact}",
        f"{signal} {city} area {quality} {urgency}",
        f"Anyone know {quality} source in {city} {urgency} {contact}",
        f"{signal} {timing} {city} {quality} {contact} {urgency}",
        f"How much for {quality} {city} area {contact}",
        f"{signal} reliable connect {city} {urgency}",
        f"{urgency} {signal} {city} {quality} {contact}",
    ]
    post = random.choice(templates).strip()
    return " ".join(post.split())


def make_normal_post() -> str:
    """Normal posts — use some shared words to create overlap."""
    city   = random.choice(CITIES)
    topic  = random.choice(NORMAL_TOPICS)
    timing = random.choice(TIMING)
    contact= random.choice(CONTACT) if random.random() < 0.3 else ""  # sometimes uses contact words

    templates = [
        f"Looking for good {topic} in {city} any recommendations",
        f"Anyone know {topic} in {city} area",
        f"Need {topic} {city} {timing} suggestions welcome",
        f"How much does {topic} cost in {city} area",
        f"Anyone available for {topic} in {city}",
        f"Good {topic} {city} recommendations please {contact}",
        f"Where to find affordable {topic} in {city}",
        f"Who has experience with {topic} in {city}",
        f"Price check for {topic} {city}",   # uses price check — normally a buyer signal
        f"Any {topic} available {city} {timing}",
    ]
    post = random.choice(templates).strip()
    return " ".join(post.split())


def generate_rich_demo(n: int = 3000) -> pd.DataFrame:
    now = datetime.now()
    rows = []
    user_ids = [f"u_{i:05d}" for i in range(1, 401)]

    for i in range(n):
        loc = random.choice(LOCATIONS)
        lat = loc[3] + random.gauss(0, 0.018)
        lon = loc[4] + random.gauss(0, 0.018)

        # Live-feeling timestamps
        rr = random.random()
        if rr < 0.40:
            mins_ago = random.randint(0, 30)
        elif rr < 0.70:
            mins_ago = random.randint(31, 240)
        else:
            mins_ago = random.randint(241, 1440)
        ts = now - timedelta(minutes=mins_ago)

        rnd = random.random()
        if rnd < 0.15:
            text = make_seller_post()
            cls  = "Seller"
            score= round(random.uniform(0.30, 0.80), 3)
            suspected = True
        elif rnd < 0.35:
            text = make_buyer_post()
            cls  = "Buyer"
            score= round(random.uniform(0.18, 0.55), 3)
            suspected = True
        else:
            text = make_normal_post()
            cls  = "Normal"
            score= round(random.uniform(0.0, 0.22), 3)
            suspected = score > 0.15

        rows.append({
            "post_id":            f"t_{i:06d}",
            "user_id":            random.choice(user_ids),
            "post_text":          text,
            "processed_text":     text.lower(),
            "timestamp":          ts.strftime("%Y-%m-%d %H:%M:%S"),
            "platform":           random.choice(["Reddit","Twitter","Telegram"]),
            "state":              loc[0],
            "city":               loc[1],
            "area":               loc[2],
            "latitude":           round(lat, 5),
            "longitude":          round(lon, 5),
            "suspicion_score":    score,
            "is_suspected":       suspected,
            "ml_predicted_class": cls,
            "cluster_label":      cls,
            "detected_codewords": json.dumps([]),
            "explanation":        "Training data",
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Generating rich overlapping training dataset (3000 posts)...")
    df = generate_rich_demo(3000)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    df.to_csv(DEMO_PATH, index=False)
    print(f"  {len(df)} posts saved")
    print(f"  Class dist: {df['ml_predicted_class'].value_counts().to_dict()}")
    # Show sample posts to verify overlap
    print("\n  Sample Seller:", df[df['ml_predicted_class']=='Seller']['post_text'].iloc[0])
    print("  Sample Buyer: ", df[df['ml_predicted_class']=='Buyer' ]['post_text'].iloc[0])
    print("  Sample Normal:", df[df['ml_predicted_class']=='Normal']['post_text'].iloc[0])
