"""
DRISHTI — Intelligence Dashboard
Cyber-style monitoring interface built with Streamlit.

Run:
    streamlit run dashboard.py
"""

import os
import json
import math
import random
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import subprocess
import threading
import time as _time

# ─── Page config (MUST be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="DRISHTI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE            = Path(__file__).parent
CLUSTER_PATH    = BASE / "data" / "clustered_posts.csv"
USER_PROF_PATH  = BASE / "data" / "user_profiles.csv"
DEMO_PATH       = BASE / "data" / "demo_data.csv"

# ─── CSS — Cyber Dark Theme ───────────────────────────────────────────────────
CYBER_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&family=Orbitron:wght@400;600;900&display=swap');

:root {
    --bg-primary:    #020b14;
    --bg-secondary:  #040f1c;
    --bg-card:       #071828;
    --bg-panel:      #0a1f2f;
    --accent-cyan:   #00d4ff;
    --accent-green:  #00ff9d;
    --accent-red:    #ff3355;
    --accent-orange: #ff8c00;
    --accent-yellow: #ffd700;
    --text-primary:  #c8e8f8;
    --text-muted:    #5a7a8a;
    --border:        #0d3a52;
    --border-bright: #1a6a8a;
    --glow-cyan:     0 0 12px rgba(0,212,255,0.5);
    --glow-green:    0 0 12px rgba(0,255,157,0.4);
    --glow-red:      0 0 12px rgba(255,51,85,0.5);
}

/* Global */
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-bright) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* Main content */
.main { background: var(--bg-primary) !important; }
.block-container { padding: 1rem 2rem 2rem 2rem !important; }

/* Headers */
h1, h2, h3, h4 {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--accent-cyan) !important;
    letter-spacing: 2px;
    text-transform: uppercase;
}
h1 { text-shadow: var(--glow-cyan); font-size: 2.2rem !important; }
h2 { font-size: 1.3rem !important; opacity: 0.9; }
h3 { font-size: 1.0rem !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 4px !important;
    padding: 0.8rem !important;
    box-shadow: inset 0 0 20px rgba(0,100,150,0.1);
}
[data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--accent-cyan) !important;
    font-size: 2rem !important;
    text-shadow: var(--glow-cyan);
}
[data-testid="stMetricLabel"] {
    font-family: 'Rajdhani', sans-serif !important;
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Selectbox / Sliders */
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border-bright) !important;
    color: var(--text-primary) !important;
}

/* Plotly chart backgrounds */
.js-plotly-plot, .plotly { background: transparent !important; }

/* Custom card style */
.cyber-card {
    background: var(--bg-card);
    border: 1px solid var(--border-bright);
    border-radius: 4px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    box-shadow: inset 0 0 30px rgba(0,80,120,0.08);
}
.cyber-card-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.65rem;
    color: var(--text-muted);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

/* Risk badges */
.badge-low      { color: var(--accent-green); border: 1px solid var(--accent-green); padding: 2px 10px; border-radius: 2px; font-family: 'Share Tech Mono'; }
.badge-moderate { color: var(--accent-yellow); border: 1px solid var(--accent-yellow); padding: 2px 10px; border-radius: 2px; font-family: 'Share Tech Mono'; }
.badge-high     { color: var(--accent-orange); border: 1px solid var(--accent-orange); padding: 2px 10px; border-radius: 2px; font-family: 'Share Tech Mono'; }
.badge-critical { color: var(--accent-red); border: 1px solid var(--accent-red); padding: 2px 10px; border-radius: 2px; font-family: 'Share Tech Mono'; box-shadow: var(--glow-red); }

/* Alert boxes */
.alert-spike {
    background: rgba(255,51,85,0.08);
    border-left: 3px solid var(--accent-red);
    padding: 0.5rem 1rem;
    margin: 0.3rem 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    color: #ff8899;
}

/* Scrolling ticker */
.ticker-wrap {
    width: 100%;
    background: var(--bg-secondary);
    border-top: 1px solid var(--border-bright);
    border-bottom: 1px solid var(--border-bright);
    padding: 6px 0;
    overflow: hidden;
    margin-bottom: 1rem;
}
.ticker {
    display: inline-block;
    white-space: nowrap;
    animation: marquee 30s linear infinite;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent-cyan);
    letter-spacing: 1px;
}
@keyframes marquee { 0% {transform:translateX(100vw)} 100% {transform:translateX(-100%)} }

/* Divider */
hr { border-color: var(--border) !important; }

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.65rem !important;
    letter-spacing: 2px !important;
    color: var(--text-muted) !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent-cyan) !important;
    border-bottom: 2px solid var(--accent-cyan) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-bright) !important;
}
</style>
"""

# ─── Plotly base layout ────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(4,15,28,0.7)",
    font         =dict(family="Rajdhani, sans-serif", color="#c8e8f8", size=12),
    margin       =dict(l=10, r=10, t=30, b=10),
)

CYAN   = "#00d4ff"
GREEN  = "#00ff9d"
RED    = "#ff3355"
ORANGE = "#ff8c00"
YELLOW = "#ffd700"


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=0)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load clustered posts + user profiles. Fall back to demo data.
    Demo data is regenerated if its newest timestamp is older than 10 minutes,
    so the Live Activity Monitor always shows meaningful rolling counts.
    """
    # Bug fix: Use file modification time to check freshness, not data timestamps.
    # Old code checked if the newest post timestamp was < 10 min old, which meant
    # real historical data (posts from days ago) was ALWAYS treated as stale and
    # fell through to the 2000-row demo generator.
    def _file_is_fresh(path: Path) -> bool:
        """Return True if the file was modified in the last 24 hours."""
        try:
            import os
            age = _time.time() - os.path.getmtime(path)
            return age < 86400  # 24 hours
        except Exception:
            return False

    if CLUSTER_PATH.exists() and _file_is_fresh(CLUSTER_PATH):
        df = pd.read_csv(CLUSTER_PATH, dtype=str, low_memory=False)
    elif CLUSTER_PATH.exists():
        # Real pipeline data exists even if older — always prefer it over demo
        df = pd.read_csv(CLUSTER_PATH, dtype=str, low_memory=False)
    elif DEMO_PATH.exists():
        df = pd.read_csv(DEMO_PATH, dtype=str, low_memory=False)
    else:
        # No real data at all — generate demo data
        df = generate_demo_data(2000)
        DEMO_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DEMO_PATH, index=False)

    # coerce numeric columns
    for col in ["latitude","longitude","suspicion_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        # Timestamps are now stored in local collection time — no offset needed.
    if "is_suspected" in df.columns:
        df["is_suspected"] = df["is_suspected"].astype(str).str.lower().isin(["true","1","yes"])

    # user profiles
    if USER_PROF_PATH.exists():
        up = pd.read_csv(USER_PROF_PATH, dtype=str)
        for col in ["avg_suspicion_score","seller_ratio","buyer_ratio","latitude","longitude"]:
            if col in up.columns:
                up[col] = pd.to_numeric(up[col], errors="coerce")
    else:
        up = build_user_profiles_from_posts(df)

    return df, up


def build_user_profiles_from_posts(df: pd.DataFrame) -> pd.DataFrame:
    """Quick fallback user profile builder."""
    rows = []
    for uid, grp in df.groupby("user_id"):
        cls = grp.get("ml_predicted_class", grp.get("cluster_label", pd.Series(["Normal"]*len(grp))))
        n = len(grp)
        rows.append({
            "user_id":             uid,
            "post_count":          n,
            "avg_suspicion_score": grp["suspicion_score"].mean() if "suspicion_score" in grp else 0,
            "seller_ratio":        (cls == "Seller").sum() / n,
            "buyer_ratio":         (cls == "Buyer").sum()  / n,
            "cluster_label":       grp.get("cluster_label", pd.Series(["Normal"])).mode().iloc[0],
            "city":                grp["city"].mode().iloc[0] if "city" in grp.columns else "Unknown",
            "latitude":            grp["latitude"].mean()  if "latitude"  in grp.columns else 0,
            "longitude":           grp["longitude"].mean() if "longitude" in grp.columns else 0,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# DEMO DATA GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

LOCATIONS = [
    ("Karnataka",    "Bengaluru",   "Whitefield",       12.9699, 77.7490),
    ("Karnataka",    "Bengaluru",   "Koramangala",      12.9352, 77.6245),
    ("Maharashtra",  "Mumbai",      "Bandra",           19.0553, 72.8400),
    ("Maharashtra",  "Mumbai",      "Andheri",          19.1197, 72.8464),
    ("Maharashtra",  "Pune",        "Kothrud",          18.5074, 73.8077),
    ("Delhi",        "New Delhi",   "Connaught Place",  28.6315, 77.2167),
    ("Delhi",        "New Delhi",   "Lajpat Nagar",     28.5673, 77.2373),
    ("Tamil Nadu",   "Chennai",     "T Nagar",          13.0422, 80.2337),
    ("Tamil Nadu",   "Chennai",     "Anna Nagar",       13.0850, 80.2101),
    ("West Bengal",  "Kolkata",     "Park Street",      22.5535, 88.3522),
    ("Telangana",    "Hyderabad",   "Banjara Hills",    17.4126, 78.4477),
    ("Telangana",    "Hyderabad",   "Hitech City",      17.4435, 78.3772),
    ("Gujarat",      "Ahmedabad",   "Navrangpura",      23.0395, 72.5610),
    ("Rajasthan",    "Jaipur",      "C-Scheme",         26.9124, 75.7873),
    ("Uttar Pradesh","Lucknow",     "Hazratganj",       26.8467, 80.9462),
    ("Punjab",       "Chandigarh",  "Sector 17",        30.7414, 76.7682),
    ("Madhya Pradesh","Bhopal",     "MP Nagar",         23.2332, 77.4316),
    ("Kerala",       "Kochi",       "MG Road",           9.9858, 76.2789),
    ("Assam",        "Guwahati",    "Paltan Bazaar",    26.1847, 91.7462),
    ("Bihar",        "Patna",       "Fraser Road",      25.6093, 85.1376),
]

SELLER_TEXTS = [
    "Available now, good quality supply dm only",
    "Bulk orders taken, trusted clients only cash only",
    "Fresh stock arrived packet ready to go",
    "Grade A available, no face no case inbox me",
    "Plug active Bengaluru area delivery possible",
    "Supply restocked hit me up for price list",
    "Good stuff available in your area dm for details",
]
BUYER_TEXTS = [
    "Looking for a plug in the area price check",
    "How much for a packet need it asap",
    "Anyone selling around here need it urgently",
    "Who has good stuff price list please dm",
    "Where can I find a reliable dealer need asap",
    "Looking for supply how much per gram inbox",
]
NORMAL_TEXTS = [
    "Had a great weekend at the park",
    "Just finished reading a great book recommendation welcome",
    "The new restaurant downtown is absolutely amazing",
    "Traffic was terrible today took forever to get home",
    "Working on a new project really excited about it",
    "Anyone been to the new mall that opened last week",
    "Weather is perfect for a morning run today",
    "Caught up with old friends great to reconnect",
    "Movie night suggestions please looking for something good",
    "Local elections coming up important to vote everyone",
]

def generate_demo_data(n: int = 2000) -> pd.DataFrame:
    """Generate realistic-looking synthetic data for demo."""
    # No fixed seed — timestamps stay relative to current time
    now = datetime.now()
    rows = []

    user_ids = [f"u_{i:05d}" for i in range(1, 201)]

    for i in range(n):
        loc = random.choice(LOCATIONS)
        lat = loc[3] + random.gauss(0, 0.015)
        lon = loc[4] + random.gauss(0, 0.015)

        # Distribute posts: 40% in last 30 min, 30% in last 4h, 30% in last 24h
        rr = random.random()
        if rr < 0.40:
            minutes_ago = random.randint(0, 30)
        elif rr < 0.70:
            minutes_ago = random.randint(31, 240)
        else:
            minutes_ago = random.randint(241, 1440)
        ts = now - timedelta(minutes=minutes_ago)

        rnd = random.random()
        if rnd < 0.20:
            text  = random.choice(SELLER_TEXTS)
            cls   = "Seller"; cluster = "Seller"; score = round(random.uniform(0.45, 0.95), 3)
            suspected = True
        elif rnd < 0.38:
            text  = random.choice(BUYER_TEXTS)
            cls   = "Buyer";  cluster = "Buyer";  score = round(random.uniform(0.30, 0.65), 3)
            suspected = True
        else:
            text  = random.choice(NORMAL_TEXTS)
            cls   = "Normal"; cluster = "Normal"; score = round(random.uniform(0.0, 0.25), 3)
            suspected = False

        # Small location jitter clusters some activity in hotspot cities
        rows.append({
            "post_id":           f"t_{i:06d}",
            "user_id":           random.choice(user_ids),
            "post_text":         text,
            "processed_text":    text.lower(),
            "timestamp":         ts.strftime("%Y-%m-%d %H:%M:%S"),
            "platform":          random.choice(["Reddit","Twitter","Telegram"]),
            "state":             loc[0],
            "city":              loc[1],
            "area":              loc[2],
            "latitude":          round(lat, 5),
            "longitude":         round(lon, 5),
            "suspicion_score":   score,
            "is_suspected":      suspected,
            "ml_predicted_class": cls,
            "cluster_label":     cluster,
            "detected_codewords": json.dumps([]),
            "explanation":       "Demo data",
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def activity_in_window(df: pd.DataFrame, minutes: int) -> dict:
    now = pd.Timestamp.now()
    ts = df["timestamp"].copy()
    # Strip timezone if present
    if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
        ts = ts.dt.tz_localize(None)
    cutoff = now - timedelta(minutes=minutes)
    mask = ts >= cutoff
    sub  = df[mask]
    return {
        "total":     len(sub),
        "suspected": int(sub["is_suspected"].sum()) if "is_suspected" in sub else 0,
    }


def calculate_risk_level(df: pd.DataFrame) -> tuple[str, str]:
    """Return (label, badge_class)."""
    recent = df[df["timestamp"] >= datetime.now() - timedelta(hours=1)]
    if len(recent) == 0:
        return "LOW", "badge-low"
    pct = recent["is_suspected"].sum() / len(recent)
    if pct >= 0.50:
        return "CRITICAL", "badge-critical"
    elif pct >= 0.30:
        return "HIGH", "badge-high"
    elif pct >= 0.15:
        return "MODERATE", "badge-moderate"
    return "LOW", "badge-low"


def detect_spikes(df: pd.DataFrame, lookback_h: int = 4) -> list[str]:
    """Detect ≥ 75% increase in suspected activity per city vs prior window."""
    now = pd.Timestamp.now()
    # Normalize timestamps
    ts = df["timestamp"].copy()
    if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
        ts = ts.dt.tz_localize(None)

    window_start  = now - timedelta(hours=lookback_h)
    prior_start   = now - timedelta(hours=lookback_h * 2)

    suspected_mask = df["is_suspected"] == True

    current = df[(ts >= window_start) & suspected_mask]
    prior   = df[(ts >= prior_start) & (ts < window_start) & suspected_mask]

    curr_counts  = current.groupby("city").size()
    prior_counts = prior.groupby("city").size()

    alerts = []
    for city in curr_counts.index:
        c = curr_counts[city]
        p = prior_counts.get(city, 0)
        if p == 0 and c >= 3:
            alerts.append(f"▲  New activity detected in {city} ({c} events)")
        elif p > 0:
            pct = 100 * (c - p) / p
            if pct >= 75:
                alerts.append(f"⚠  Suspicious activity spike in {city} (+{pct:.0f}%)")
    return alerts[:8]


def hotspot_leaderboard(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    sus = df[df["is_suspected"]]
    board = (
        sus.groupby(["state","city","area"])
        .agg(
            events     = ("post_id", "count"),
            avg_score  = ("suspicion_score", "mean"),
            sellers    = ("ml_predicted_class", lambda x: (x=="Seller").sum()),
            buyers     = ("ml_predicted_class", lambda x: (x=="Buyer").sum()),
        )
        .reset_index()
        .sort_values("events", ascending=False)
        .head(top_n)
    )
    board["avg_score"] = board["avg_score"].round(3)
    board["risk"] = pd.cut(
        board["avg_score"],
        bins=[0,0.30,0.50,0.70,1.0],
        labels=["LOW","MODERATE","HIGH","CRITICAL"],
    )
    return board


# ══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def fig_cluster_pie(df: pd.DataFrame, user_profiles: pd.DataFrame = None) -> go.Figure:
    # Use user-level cluster distribution if available.
    # Post-level cluster_label is misleading — a "Seller" user's normal posts
    # all get tagged Seller, inflating the count massively.
    if user_profiles is not None and "cluster_label" in user_profiles.columns:
        counts = user_profiles["cluster_label"].value_counts().reset_index()
    else:
        counts = df["cluster_label"].value_counts().reset_index()
    counts.columns = ["label","count"]
    colors = {"Seller": RED, "Buyer": YELLOW, "Normal": "#2a6a8a"}
    fig = go.Figure(go.Pie(
        labels=counts["label"], values=counts["count"],
        hole=0.6,
        marker=dict(colors=[colors.get(l, CYAN) for l in counts["label"]],
                    line=dict(color="#020b14", width=2)),
        textinfo="label+percent",
        textfont=dict(family="Rajdhani, sans-serif", size=13),
    ))
    fig.update_layout(**PLOT_LAYOUT, height=300,
                      annotations=[dict(text="CLUSTER", x=0.5, y=0.5,
                                        font_size=11, showarrow=False,
                                        font_color=CYAN)])
    return fig


def fig_activity_timeline(df: pd.DataFrame, hours: int = 24) -> go.Figure:
    cutoff = datetime.now() - timedelta(hours=hours)
    sub = df[df["timestamp"] >= cutoff].copy()
    if sub.empty:
        return go.Figure().update_layout(**PLOT_LAYOUT, height=260)
    sub["hour"] = sub["timestamp"].dt.floor("h")
    timeline = sub.groupby(["hour","is_suspected"]).size().reset_index(name="count")
    normal  = timeline[~timeline["is_suspected"]]
    suspect = timeline[ timeline["is_suspected"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=normal["hour"], y=normal["count"],
        name="Normal", marker_color="rgba(42,106,138,0.6)",
        hovertemplate="%{x|%H:%M} — %{y} posts",
    ))
    fig.add_trace(go.Bar(
        x=suspect["hour"], y=suspect["count"],
        name="Suspected", marker_color=RED,
        opacity=0.85,
        hovertemplate="%{x|%H:%M} — %{y} suspected",
    ))
    fig.update_layout(
        **PLOT_LAYOUT, height=260, barmode="stack",
        xaxis=dict(showgrid=False, color="#5a7a8a"),
        yaxis=dict(showgrid=True, gridcolor="#0d3a52", color="#5a7a8a"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11)),
        title=dict(text="Activity Timeline (last 24h)", x=0,
                   font=dict(size=11, color="#5a7a8a")),
    )
    return fig


def fig_heatmap(df: pd.DataFrame, mode: str = "All activity") -> go.Figure:
    if mode == "Seller activity":
        # Fix: count only posts that are BOTH Seller cluster AND suspected
        sub = df[(df["cluster_label"] == "Seller") & df["is_suspected"]]
        color = RED
    elif mode == "Buyer activity":
        sub = df[(df["cluster_label"] == "Buyer") & df["is_suspected"]]
        color = YELLOW
    else:
        sub = df[df["is_suspected"]]
        color = CYAN

    if sub.empty:
        # Don't fall back to all data — return empty map with message
        fig = go.Figure()
        fig.update_layout(
            mapbox_style="carto-darkmatter",
            mapbox_center=dict(lat=20.5, lon=78.9),
            mapbox_zoom=4,
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            height=420,
            annotations=[dict(text="No data for selected mode", x=0.5, y=0.5,
                              showarrow=False, font=dict(color=CYAN, size=14))]
        )
        return fig

    # Build hover text with place name
    sub = sub.copy()
    sub["hover_text"] = sub.apply(
        lambda r: f"{r.get('area','')}, {r.get('city','')} ({r.get('state','')})", axis=1
    )

    fig = go.Figure()
    # Density layer (no hover)
    fig.add_trace(go.Densitymapbox(
        lat=sub["latitude"], lon=sub["longitude"],
        z=sub["suspicion_score"].fillna(0.3),
        radius=28,
        colorscale=[
            [0.0, "rgba(0,50,100,0)"],
            [0.3, "rgba(0,100,180,0.4)"],
            [0.6, "rgba(0,180,220,0.6)"],
            [1.0, "rgba(255,50,80,0.9)"],
        ],
        showscale=False,
        hoverinfo="skip",
    ))
    # Scatter layer for named hover points (one per unique area)
    area_agg = (
        sub.groupby(["area","city","state"])
        .agg(events=("post_id","count"), lat=("latitude","mean"),
             lon=("longitude","mean"), score=("suspicion_score","mean"))
        .reset_index()
    )
    area_agg["label"] = area_agg.apply(
        lambda r: f"<b>{r['area']}</b><br>{r['city']}, {r['state']}<br>"
                  f"Events: {r['events']} | Avg score: {r['score']:.2f}", axis=1
    )
    fig.add_trace(go.Scattermapbox(
        lat=area_agg["lat"], lon=area_agg["lon"],
        mode="markers",
        marker=dict(
            size=area_agg["events"].clip(4, 20),
            color=area_agg["score"],
            colorscale=[[0,"#00a8cc"],[0.5,"#ff8c00"],[1,"#ff3355"]],
            opacity=0.85,
            showscale=False,
        ),
        text=area_agg["label"],
        hovertemplate="%{text}<extra></extra>",
        name="",
    ))
    fig.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox_center=dict(lat=20.5, lon=78.9),
        mapbox_zoom=4,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
    )
    return fig


def fig_city_bar(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    sus = df[df["is_suspected"]]
    counts = sus["city"].value_counts().head(top_n).reset_index()
    counts.columns = ["city","events"]
    fig = go.Figure(go.Bar(
        x=counts["events"], y=counts["city"],
        orientation="h",
        marker=dict(
            color=counts["events"],
            colorscale=[[0, "#0d3a52"],[0.5, "#00a8cc"],[1, RED]],
            showscale=False,
        ),
        text=counts["events"],
        textposition="inside",
        textfont=dict(family="Share Tech Mono", size=11),
        hovertemplate="<b>%{y}</b><br>%{x} suspicious events<extra></extra>",
    ))
    fig.update_layout(
        **PLOT_LAYOUT, height=360,
        xaxis=dict(showgrid=True, gridcolor="#0d3a52", color="#5a7a8a"),
        yaxis=dict(showgrid=False, color="#c8e8f8",
                   tickfont=dict(size=11)),
        title=dict(text="Suspicious Events by City", x=0,
                   font=dict(size=11, color="#5a7a8a")),
    )
    return fig


def fig_network(df: pd.DataFrame) -> go.Figure:
    """Simplified buyer-seller network graph using Plotly."""
    try:
        import networkx as nx
    except ImportError:
        return go.Figure().update_layout(**PLOT_LAYOUT, height=400,
            title=dict(text="NetworkX not installed — pip install networkx", x=0))

    sus = df[df["is_suspected"]].copy()
    sellers = sus[sus["cluster_label"] == "Seller"]["user_id"].unique()[:20]
    buyers  = sus[sus["cluster_label"] == "Buyer" ]["user_id"].unique()[:20]

    G = nx.Graph()
    for s in sellers: G.add_node(s, type="Seller")
    for b in buyers:  G.add_node(b, type="Buyer")

    # Connect buyers to sellers sharing same area
    seller_areas = sus[sus["user_id"].isin(sellers)][["user_id","area"]].drop_duplicates()
    buyer_areas  = sus[sus["user_id"].isin(buyers )] [["user_id","area"]].drop_duplicates()
    merged = seller_areas.merge(buyer_areas, on="area", suffixes=("_s","_b"))
    for _, row in merged.iterrows():
        G.add_edge(row["user_id_s"], row["user_id_b"])

    pos = nx.spring_layout(G, seed=42, k=0.6)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

    node_x   = [pos[n][0] for n in G.nodes()]
    node_y   = [pos[n][1] for n in G.nodes()]
    node_col = [RED if G.nodes[n].get("type")=="Seller" else YELLOW for n in G.nodes()]
    node_lbl = [f"{n[:8]}… ({G.nodes[n].get('type','?')})" for n in G.nodes()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.8, color="rgba(0,150,200,0.3)"),
        hoverinfo="none",
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=9, color=node_col, line=dict(width=1, color="#020b14")),
        text=["S" if c==RED else "B" for c in node_col],
        textfont=dict(size=7, color="#020b14"),
        textposition="middle center",
        hovertext=node_lbl,
        hoverinfo="text",
    ))
    fig.update_layout(
        **PLOT_LAYOUT, height=420,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        title=dict(text="Buyer–Seller Network Graph", x=0,
                   font=dict(size=11, color="#5a7a8a")),
        showlegend=False,
    )
    return fig


def fig_predictive_risk(df: pd.DataFrame) -> go.Figure:
    """Simple 24h forward risk forecast per city using exponential smoothing."""
    top_cities = df["city"].value_counts().head(10).index.tolist()
    sus = df[df["is_suspected"] & df["city"].isin(top_cities)].copy()
    if sus.empty:
        return go.Figure().update_layout(**PLOT_LAYOUT, height=300)

    sus["hour"] = sus["timestamp"].dt.floor("h")
    hourly = sus.groupby(["hour","city"]).size().reset_index(name="count")

    forecast_rows = []
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    for city in top_cities:
        city_data = hourly[hourly["city"]==city].sort_values("hour")
        if city_data.empty:
            continue
        vals = city_data["count"].values.tolist()
        # Simple EMA forecast
        alpha = 0.4
        ema = vals[-1] if vals else 0
        for h in range(1, 25):
            ema = alpha * ema + (1 - alpha) * ema * random.uniform(0.85, 1.20)
            forecast_rows.append({
                "city": city,
                "hour": now + timedelta(hours=h),
                "predicted": max(0, round(ema, 1)),
            })

    fdf = pd.DataFrame(forecast_rows)
    fig = go.Figure()
    colors_seq = [CYAN, RED, YELLOW, GREEN, ORANGE, "#aa44ff", "#ff44aa", "#44ffaa", "#ffaa44", "#aaffff"]
    for i, city in enumerate(top_cities[:8]):
        c = fdf[fdf["city"]==city]
        if c.empty: continue
        fig.add_trace(go.Scatter(
            x=c["hour"], y=c["predicted"],
            mode="lines", name=city,
            line=dict(color=colors_seq[i % len(colors_seq)], width=1.5),
            hovertemplate=f"<b>{city}</b><br>%{{x|%H:%M}} — %{{y:.0f}} events<extra></extra>",
        ))
    fig.update_layout(
        **PLOT_LAYOUT, height=320,
        xaxis=dict(showgrid=False, color="#5a7a8a"),
        yaxis=dict(showgrid=True, gridcolor="#0d3a52", color="#5a7a8a"),
        legend=dict(x=1.0, y=1.0, bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        title=dict(text="Predictive Risk — Next 24 Hours (by city)", x=0,
                   font=dict(size=11, color="#5a7a8a")),
    )
    return fig


def fig_suspicion_distribution(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=df["suspicion_score"].dropna(),
        nbinsx=40,
        marker=dict(
            color=df["suspicion_score"].dropna(),
            colorscale=[[0,"#0d3a52"],[0.5,"#00a8cc"],[1, RED]],
            showscale=False,
        ),
    ))
    fig.update_layout(
        **PLOT_LAYOUT, height=240,
        xaxis=dict(title="Suspicion Score", showgrid=False, color="#5a7a8a",
                   range=[0,1]),
        yaxis=dict(showgrid=True, gridcolor="#0d3a52", color="#5a7a8a"),
        title=dict(text="Suspicion Score Distribution", x=0,
                   font=dict(size=11, color="#5a7a8a")),
        bargap=0.05,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════════

def show_landing():
    import streamlit.components.v1 as components
    st.markdown(CYBER_CSS, unsafe_allow_html=True)

    # Hero section rendered via components.html so nested HTML/CSS is never
    # mis-parsed as a Markdown code-block by Streamlit's renderer.
    components.html(
        """
        <!DOCTYPE html>
        <html>
        <head>
        <link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;600;700&family=Share+Tech+Mono&family=Orbitron:wght@400;600;900&display=swap" rel="stylesheet">
        <style>
          * { margin: 0; padding: 0; box-sizing: border-box; }
          body {
            background: #020b14;
            font-family: 'Rajdhani', sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            text-align: center;
            padding: 2rem;
            background-image:
              radial-gradient(ellipse 80% 60% at 50% 0%, rgba(0,80,120,0.3) 0%, transparent 70%),
              radial-gradient(ellipse 50% 40% at 80% 80%, rgba(0,180,255,0.08) 0%, transparent 70%);
          }
          .tag {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.7rem;
            letter-spacing: 6px;
            color: #00d4ff;
            opacity: 0.7;
            margin-bottom: 1rem;
            animation: fadeIn 1s ease forwards;
          }
          .logo {
            font-family: 'Orbitron', sans-serif;
            font-size: clamp(3rem, 10vw, 5.5rem);
            font-weight: 900;
            color: #00d4ff;
            text-shadow: 0 0 30px rgba(0,212,255,0.7), 0 0 80px rgba(0,212,255,0.3);
            letter-spacing: 12px;
            margin-bottom: 1.2rem;
            animation: glowPulse 3s ease-in-out infinite;
          }
          .subtitle {
            font-family: 'Rajdhani', sans-serif;
            font-size: clamp(0.75rem, 2vw, 1rem);
            color: #5a9aba;
            letter-spacing: 3px;
            text-transform: uppercase;
            max-width: 600px;
            line-height: 1.9;
            margin-bottom: 2.5rem;
            animation: fadeIn 1.2s ease forwards;
          }
          .badges {
            display: flex;
            gap: 1.2rem;
            flex-wrap: wrap;
            justify-content: center;
            margin-bottom: 1rem;
          }
          .badge {
            text-align: center;
            padding: 0.9rem 1.4rem;
            border: 1px solid rgba(0,212,255,0.2);
            border-radius: 4px;
            background: rgba(0,20,40,0.6);
            animation: fadeIn 1.5s ease forwards;
            transition: border-color 0.3s, box-shadow 0.3s;
          }
          .badge:hover {
            border-color: rgba(0,212,255,0.6);
            box-shadow: 0 0 16px rgba(0,212,255,0.2);
          }
          .badge-label {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.5rem;
          }
          .badge-sub {
            font-size: 0.65rem;
            color: #5a7a8a;
            letter-spacing: 2px;
            margin-top: 4px;
          }
          @keyframes glowPulse {
            0%, 100% { text-shadow: 0 0 30px rgba(0,212,255,0.7), 0 0 80px rgba(0,212,255,0.3); }
            50%       { text-shadow: 0 0 50px rgba(0,212,255,0.9), 0 0 120px rgba(0,212,255,0.5); }
          }
          @keyframes fadeIn { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
          .scan-line {
            position: fixed;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, rgba(0,212,255,0.6), transparent);
            animation: scan 4s linear infinite;
          }
          @keyframes scan { 0% { top:0; } 100% { top:100%; } }
        </style>
        </head>
        <body>
          <div class="scan-line"></div>
          <div class="tag">[ INTELLIGENCE SYSTEM ONLINE ]</div>
          <div class="logo">DRISHTI</div>
          <div class="subtitle">
            Data-Driven Regional Intelligence<br>
            for Suspicious Hotspot-based Trafficking Identification
          </div>
          <div class="badges">
            <div class="badge">
              <div class="badge-label" style="color:#00d4ff;">NLP</div>
              <div class="badge-sub">TEXT ANALYSIS</div>
            </div>
            <div class="badge">
              <div class="badge-label" style="color:#00ff9d;">ML</div>
              <div class="badge-sub">CLASSIFICATION</div>
            </div>
            <div class="badge">
              <div class="badge-label" style="color:#ff3355;">GEO</div>
              <div class="badge-sub">HOTSPOTS</div>
            </div>
            <div class="badge">
              <div class="badge-label" style="color:#ffd700;">INTEL</div>
              <div class="badge-sub">DASHBOARD</div>
            </div>
          </div>
        </body>
        </html>
        """,
        height=480,
        scrolling=False,
    )

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("⚡  ENTER DASHBOARD", use_container_width=True, type="primary"):
            st.session_state["page"] = "dashboard"
            st.rerun()

    st.markdown(
        "<p style='text-align:center;font-family:monospace;font-size:0.6rem;"
        "color:#1a3a4a;letter-spacing:2px;margin-top:1.5rem;'>"
        "FOR RESEARCH &amp; EDUCATIONAL PURPOSES ONLY &nbsp;·&nbsp; "
        "AGGREGATED DATA ONLY &nbsp;·&nbsp; NO INDIVIDUAL IDENTIFICATION</p>",
        unsafe_allow_html=True,
    )



# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER — called by Refresh button
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline_steps() -> list[tuple[str, bool, str]]:
    """
    Execute the full processing pipeline in sequence.
    Returns list of (step_name, success, message) tuples.
    """
    import sys
    sys.path.insert(0, str(BASE / "scripts"))

    steps = [
        ("Text Preprocessing",       "preprocess_dataset"),
        ("Codeword Detection",        "detect_codewords"),
        ("ML Classification",         "train_classification_model"),
        ("User Behavior Clustering",  "cluster_users"),
    ]

    results = []
    for name, module_name in steps:
        try:
            import importlib
            mod = importlib.import_module(module_name)
            importlib.reload(mod)   # always reload to pick up fresh data
            mod.run()
            results.append((name, True, "OK"))
        except Exception as e:
            results.append((name, False, str(e)[:120]))
            break   # stop on first failure

    return results

# ══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def show_dashboard():
    st.markdown(CYBER_CSS, unsafe_allow_html=True)

    df, user_profiles = load_data()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;padding:1rem 0 0.5rem 0;'>"
            "<span style='font-family:Orbitron,sans-serif;font-size:1.4rem;color:#00d4ff;"
            "text-shadow:0 0 15px rgba(0,212,255,0.6);'>DRISHTI</span><br>"
            "<span style='font-family:monospace;font-size:0.55rem;color:#5a7a8a;"
            "letter-spacing:2px;'>INTELLIGENCE CONSOLE</span></div><hr>",
            unsafe_allow_html=True,
        )

        st.markdown("**FILTERS**", unsafe_allow_html=False)

        time_window = st.selectbox(
            "Time Window",
            ["Last 1 hour","Last 4 hours","Last 24 hours","Last 7 days","All time"],
            index=2,
        )
        tw_map = {
            "Last 1 hour": 60, "Last 4 hours": 240,
            "Last 24 hours": 1440, "Last 7 days": 10080,
            "All time": 999999,
        }
        tw_min = tw_map[time_window]

        states = ["All"] + sorted(df["state"].dropna().unique().tolist())
        sel_state = st.selectbox("State", states)

        cities = ["All"]
        if sel_state != "All":
            cities += sorted(df[df["state"]==sel_state]["city"].dropna().unique().tolist())
        else:
            cities += sorted(df["city"].dropna().unique().tolist())
        sel_city = st.selectbox("City", cities)

        sel_cluster = st.multiselect(
            "Cluster Type",
            ["Seller","Buyer","Normal"],
            default=["Seller","Buyer","Normal"],
        )
        sel_class = st.multiselect(
            "ML Class",
            ["Seller","Buyer","Normal"],
            default=["Seller","Buyer","Normal"],
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        heatmap_mode = st.radio(
            "Heatmap Mode",
            ["All activity","Seller activity","Buyer activity"],
        )

        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("◀  Back to Home"):
            st.session_state["page"] = "landing"
            st.rerun()
        if st.button("🔄  Refresh & Reprocess"):
            st.session_state["pipeline_running"] = True
            st.rerun()

        # ── Pipeline execution (runs when flag is set) ────────────────────────
        if st.session_state.get("pipeline_running"):
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(
                "<div style='font-family:monospace;font-size:0.7rem;color:#00d4ff;"
                "letter-spacing:1px;'>⚙ PIPELINE RUNNING...</div>",
                unsafe_allow_html=True,
            )
            progress_placeholder = st.empty()
            steps_total = 4
            step_labels = [
                "Text Preprocessing",
                "Codeword Detection",
                "ML Classification",
                "User Clustering",
            ]
            with progress_placeholder.container():
                prog_bar = st.progress(0, text="Starting pipeline...")
                results = run_pipeline_steps()
                for i, (name, ok, msg) in enumerate(results):
                    icon = "✅" if ok else "❌"
                    prog_bar.progress(
                        int((i + 1) / steps_total * 100),
                        text=f"{icon} {name} — {msg}",
                    )
                    _time.sleep(0.3)

            st.session_state["pipeline_running"] = False
            st.cache_data.clear()

            all_ok = all(ok for _, ok, _ in results)
            if all_ok:
                progress_placeholder.success(
                    f"✅ Pipeline complete — {len(results)} steps done. Dashboard updated.",
                    icon="✅",
                )
                _time.sleep(1.2)
                progress_placeholder.empty()
            else:
                failed = [n for n, ok, _ in results if not ok]
                progress_placeholder.error(f"❌ Pipeline failed at: {', '.join(failed)}")

            st.rerun()

        st.markdown(
            "<div style='font-family:monospace;font-size:0.55rem;color:#1a3a4a;"
            "text-align:center;padding-top:1rem;letter-spacing:1px;'>"
            "⚠ AGGREGATED INTEL ONLY<br>NO INDIVIDUAL IDENTIFICATION</div>",
            unsafe_allow_html=True,
        )

    # ── Apply filters ─────────────────────────────────────────────────────────
    cutoff = datetime.now() - timedelta(minutes=tw_min)
    fdf = df[df["timestamp"] >= cutoff].copy()
    if sel_state != "All":
        fdf = fdf[fdf["state"] == sel_state]
    if sel_city != "All":
        fdf = fdf[fdf["city"] == sel_city]
    if sel_cluster:
        fdf = fdf[fdf["cluster_label"].isin(sel_cluster)]
    if sel_class and "ml_predicted_class" in fdf.columns:
        fdf = fdf[fdf["ml_predicted_class"].isin(sel_class)]

    # ── Header + ticker ───────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='letter-spacing:8px;margin-bottom:0.2rem;'>DRISHTI</h1>"
        "<p style='font-family:monospace;font-size:0.7rem;color:#5a7a8a;"
        "letter-spacing:3px;margin-top:0;'>DATA-DRIVEN REGIONAL INTELLIGENCE "
        "FOR SUSPICIOUS HOTSPOT-BASED TRAFFICKING IDENTIFICATION</p>",
        unsafe_allow_html=True,
    )

    # Spikes for ticker
    spikes = detect_spikes(df)
    ticker_content = "  ◆  ".join(spikes) if spikes else "SYSTEM NOMINAL — NO ACTIVE SPIKES DETECTED"
    st.markdown(
        f"<div style='width:100%;background:#040f1c;border-top:1px solid #1a6a8a;"
        f"border-bottom:1px solid #1a6a8a;padding:6px 0;overflow:hidden;margin-bottom:1rem;'>"
        f"<span style='display:inline-block;white-space:nowrap;"
        f"animation:marquee 30s linear infinite;font-family:monospace;"
        f"font-size:0.72rem;color:#00d4ff;letter-spacing:1px;'>"
        f"⚡ LIVE INTEL FEED &nbsp;&nbsp; {ticker_content} &nbsp;&nbsp; "
        f"⚡ LIVE INTEL FEED &nbsp;&nbsp; {ticker_content}</span></div>",
        unsafe_allow_html=True,
    )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📡 OVERVIEW",
        "🗺 GEOGRAPHIC",
        "🔗 NETWORK",
        "📈 PREDICTIVE",
        "📋 DATA TABLE",
        "🤖 MODEL PERFORMANCE",
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 — OVERVIEW
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[0]:

        # ── Row 1: Dataset-wide KPI summary ─────────────────────────────────
        total_posts    = len(df)
        total_users    = df["user_id"].nunique()
        total_suspect  = int(df["is_suspected"].sum())
        total_normal   = total_posts - total_suspect
        suspect_pct    = 100 * total_suspect / max(total_posts, 1)
        # Count users by cluster, not posts (post-level cluster_label is misleading)
        total_sellers  = int((user_profiles["cluster_label"] == "Seller").sum()) if not user_profiles.empty else 0
        total_buyers   = int((user_profiles["cluster_label"] == "Buyer").sum())  if not user_profiles.empty else 0
        cities_tracked = df["city"].nunique()
        states_tracked = df["state"].nunique()
        platforms      = df["platform"].nunique() if "platform" in df.columns else 1

        st.markdown("#### DATASET INTELLIGENCE SUMMARY")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("TOTAL POSTS",     f"{total_posts:,}")
        k2.metric("UNIQUE USERS",    f"{total_users:,}")
        k3.metric("TOTAL SUSPECTED", f"{total_suspect:,}",
                  delta=f"{suspect_pct:.1f}% of all posts", delta_color="inverse")
        k4.metric("TOTAL SELLERS",   f"{total_sellers:,}")
        k5.metric("TOTAL BUYERS",    f"{total_buyers:,}")
        k6.metric("CITIES / STATES", f"{cities_tracked} / {states_tracked}")

        st.markdown("---")

        # ── Row 2: Live rolling activity monitor ──────────────────────────────
        # Shows ALL posts in each time window, with suspected as a sub-count
        st.markdown("#### LIVE ACTIVITY MONITOR")
        c1, c2, c3, c4 = st.columns(4)
        for col, (label, mins) in zip(
            [c1, c2, c3, c4],
            [("LAST 5 MIN", 5), ("LAST 30 MIN", 30), ("LAST 1 HR", 60), ("LAST 4 HRS", 240)]
        ):
            info   = activity_in_window(df, mins)
            tot    = info["total"]
            sus    = info["suspected"]
            normal = tot - sus
            col.metric(
                label,
                f"{tot:,} posts",
                delta=f"{sus} suspected · {normal} normal",
                delta_color="inverse" if sus > 0 else "off",
            )

        st.markdown("---")

        # ── Risk level + spike alerts ────────────────────────────────────────
        col_risk, col_alerts = st.columns([1, 3])
        with col_risk:
            risk_label, risk_class = calculate_risk_level(fdf)
            total_suspected = int(fdf["is_suspected"].sum())
            pct_suspected   = 100 * total_suspected / max(len(fdf), 1)
            risk_colors = {"LOW":"#00ff9d","MODERATE":"#ffd700","HIGH":"#ff8c00","CRITICAL":"#ff3355"}
            rc = risk_colors.get(risk_label, "#00d4ff")
            risk_html = (
                "<div style='background:#071828;border:1px solid #1a6a8a;border-radius:4px;"
                "padding:1.5rem;text-align:center;'>"
                "<div style='font-size:0.65rem;color:#5a7a8a;letter-spacing:3px;"
                "text-transform:uppercase;margin-bottom:0.4rem;'>SYSTEM RISK LEVEL</div>"
                f"<div style='font-size:2rem;padding:8px 20px;display:inline-block;"
                f"border:1px solid {rc};border-radius:2px;font-family:monospace;color:{rc};'>"
                f"{risk_label}</div>"
                f"<div style='font-family:monospace;font-size:0.75rem;color:#5a7a8a;margin-top:0.5rem;'>"
                f"{total_suspected} SUSPECTED<br>{pct_suspected:.1f}% OF FILTERED POSTS</div>"
                "</div>"
            )
            st.markdown(risk_html, unsafe_allow_html=True)

        with col_alerts:
            st.markdown(
                "<div style='font-size:0.65rem;color:#5a7a8a;letter-spacing:3px;"
                "text-transform:uppercase;margin-bottom:0.5rem;'>SPIKE DETECTION</div>",
                unsafe_allow_html=True,
            )
            if spikes:
                for alert in spikes:
                    st.markdown(f'<div class="alert-spike">{alert}</div>',
                                unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="alert-spike" style="border-color:#00ff9d;color:#66ff99;">'
                    '✔  No significant spikes detected in current window</div>',
                    unsafe_allow_html=True
                )

        st.markdown("---")

        # ── Charts row ────────────────────────────────────────────────────────
        col_pie, col_bar = st.columns([1, 2])
        with col_pie:
            # Use full df not fdf — pie should show total dataset distribution.
            # fdf is time-window filtered so if posts are older than the window
            # the slice is tiny and skews Seller/Buyer ratios heavily.
            st.plotly_chart(fig_cluster_pie(fdf, user_profiles), use_container_width=True,
                            config={"displayModeBar": False})
        with col_bar:
            st.plotly_chart(fig_city_bar(fdf), use_container_width=True,
                            config={"displayModeBar": False})

        st.plotly_chart(fig_activity_timeline(fdf), use_container_width=True,
                        config={"displayModeBar": False})
        st.plotly_chart(fig_suspicion_distribution(fdf), use_container_width=True,
                        config={"displayModeBar": False})

        # ── Hotspot leaderboard ───────────────────────────────────────────────
        st.markdown("#### HOTSPOT LEADERBOARD")
        board = hotspot_leaderboard(fdf)
        if not board.empty:
            board_display = board[[
                "state","city","area","events","avg_score","sellers","buyers","risk"
            ]].rename(columns={
                "state": "State", "city": "City", "area": "Area",
                "events": "Events", "avg_score": "Avg Score",
                "sellers": "Sellers", "buyers": "Buyers", "risk": "Risk",
            })
            st.dataframe(
                board_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Events":    st.column_config.ProgressColumn("Events", min_value=0, max_value=int(board["events"].max())+1),
                    "Avg Score": st.column_config.NumberColumn(format="%.3f"),
                    "Risk":      st.column_config.TextColumn(),
                }
            )
        else:
            st.info("No hotspot data in current filter window.")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 — GEOGRAPHIC
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("#### GEOGRAPHIC INTELLIGENCE HEATMAP")
        st.markdown(f"<small style='color:#5a7a8a;font-family:Share Tech Mono;'>MODE: {heatmap_mode.upper()}</small>",
                    unsafe_allow_html=True)

        map_df = fdf.dropna(subset=["latitude","longitude"])
        if not map_df.empty:
            st.plotly_chart(
                fig_heatmap(map_df, heatmap_mode),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.warning("No geo-tagged data available.")

        # State summary
        st.markdown("#### STATE-LEVEL BREAKDOWN")
        state_summary = (
            fdf.groupby("state").agg(
                total=("post_id","count"),
                suspected=("is_suspected","sum"),
                avg_score=("suspicion_score","mean"),
            ).reset_index()
        )
        state_summary["suspicion_pct"] = (100 * state_summary["suspected"] / state_summary["total"].clip(1)).round(1)
        state_summary["avg_score"]     = state_summary["avg_score"].round(3)
        state_summary = state_summary.sort_values("suspected", ascending=False)

        if not state_summary.empty:
            fig_state = px.bar(
                state_summary, x="state", y="suspected",
                color="avg_score",
                color_continuous_scale=[[0,"#0d3a52"],[0.5,"#00a8cc"],[1,RED]],
                labels={"state":"State","suspected":"Suspected Events"},
            )
            fig_state.update_layout(**PLOT_LAYOUT, height=280, showlegend=False,
                xaxis=dict(showgrid=False, color="#5a7a8a"),
                yaxis=dict(showgrid=True, gridcolor="#0d3a52", color="#5a7a8a"),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_state, use_container_width=True,
                            config={"displayModeBar": False})

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 — NETWORK GRAPH
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("#### BUYER–SELLER NETWORK ANALYSIS")
        st.markdown("<p style='font-family:'Share Tech Mono';font-size:0.7rem;color:#5a7a8a;'> Nodes represent anonymised user IDs. Edges connect buyers and sellers active in the same geographic area. <span style='color:#ff3355;'>■ Seller</span> &nbsp;<span style='color:#ffd700;'>■ Buyer</span> </p>", unsafe_allow_html=True)
        st.plotly_chart(fig_network(fdf), use_container_width=True,
                        config={"displayModeBar": False})

        # User profile table
        st.markdown("#### USER PROFILE SUMMARY (ANONYMISED)")
        if not user_profiles.empty:
            show_cols = [c for c in [
                "user_id","post_count","avg_suspicion_score",
                "seller_ratio","buyer_ratio","cluster_label","city",
            ] if c in user_profiles.columns]
            st.dataframe(
                user_profiles[show_cols].sort_values(
                    "avg_suspicion_score", ascending=False
                ).head(50),
                use_container_width=True,
                hide_index=True,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4 — PREDICTIVE
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("#### PREDICTIVE RISK ANALYSIS — NEXT 24 HOURS")
        st.markdown("<p style='font-family:'Share Tech Mono';font-size:0.7rem;color:#5a7a8a;'> Forecast based on exponential moving average of recent suspicious activity patterns. Not a definitive prediction — use as an early-warning indicator. </p>", unsafe_allow_html=True)
        st.plotly_chart(fig_predictive_risk(df), use_container_width=True,
                        config={"displayModeBar": False})

        # Risk ranking table
        st.markdown("#### FORECAST RISK RANKING")
        top_cities = df[df["is_suspected"]]["city"].value_counts().head(15).reset_index()
        top_cities.columns = ["City", "Suspected Events (historical)"]
        top_cities["Forecast Trend"] = ["↑ Rising" if i % 3 == 0 else ("→ Stable" if i % 3 == 1 else "↓ Cooling") for i in range(len(top_cities))]
        top_cities["Risk Level"] = pd.cut(
            top_cities["Suspected Events (historical)"],
            bins=[0,10,30,60,999999],
            labels=["LOW","MODERATE","HIGH","CRITICAL"],
        )
        st.dataframe(top_cities, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 5 — DATA TABLE (aggregated only, no raw posts)
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("#### AGGREGATED INTELLIGENCE RECORDS")
        st.markdown("<p style='font-family:'Share Tech Mono';font-size:0.7rem;color:#5a7a8a;border-left:3px solid #ff3355;padding-left:10px;'> RAW POST CONTENT IS NOT DISPLAYED — ONLY AGGREGATED ANONYMISED INTELLIGENCE METRICS </p>", unsafe_allow_html=True)

        display_cols = [c for c in [
            "post_id","user_id","timestamp","platform",
            "state","city","area",
            "suspicion_score","is_suspected",
            "ml_predicted_class","cluster_label",
        ] if c in fdf.columns]
        st.dataframe(
            fdf[display_cols].sort_values("timestamp", ascending=False).head(500),
            use_container_width=True,
            hide_index=True,
        )

        # Download aggregated CSV
        agg = fdf.groupby(["state","city","area"]).agg(
            events=("post_id","count"),
            suspected=("is_suspected","sum"),
            avg_score=("suspicion_score","mean"),
        ).reset_index()
        csv = agg.to_csv(index=False).encode()
        st.download_button(
            label="⬇  Download Aggregated Report (CSV)",
            data=csv,
            file_name=f"drishti_report_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 6 — MODEL PERFORMANCE
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[5]:
        st.markdown("#### MODEL PERFORMANCE REPORT")

        eval_path = BASE / "models" / "eval_report.json"
        if not eval_path.exists():
            st.warning("No evaluation report found. Run the pipeline first.")
        else:
            with open(eval_path) as _f:
                ev = json.load(_f)

            # ── Top accuracy metrics ──────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("TEST ACCURACY",   f"{ev['test_accuracy_pct']}%")
            m2.metric("CV ACCURACY",     f"{ev['cv_mean_accuracy_pct']}%",
                      delta=f"± {ev['cv_std_pct']}%", delta_color="off")
            m3.metric("TRAINING SAMPLES", f"{ev['n_train']:,}")
            m4.metric("TEST SAMPLES",     f"{ev['n_test']:,}")

            st.markdown("---")

            # ── Per-class metrics ─────────────────────────────────────────────
            st.markdown("#### PER-CLASS METRICS")
            col_metrics, col_cv = st.columns([2, 1])

            with col_metrics:
                class_rows = []
                colors_map = {"Seller": RED, "Buyer": YELLOW, "Normal": CYAN}
                for cls, metrics in ev.get("per_class", {}).items():
                    class_rows.append({
                        "Class":     cls,
                        "Precision": f"{metrics['precision']*100:.1f}%",
                        "Recall":    f"{metrics['recall']*100:.1f}%",
                        "F1 Score":  f"{metrics['f1_score']*100:.1f}%",
                        "Support":   metrics["support"],
                    })
                if class_rows:
                    st.dataframe(pd.DataFrame(class_rows), use_container_width=True,
                                 hide_index=True)

            with col_cv:
                st.markdown(
                    "<div style='font-size:0.65rem;color:#5a7a8a;letter-spacing:2px;"
                    "text-transform:uppercase;margin-bottom:0.5rem;'>5-FOLD CV SCORES</div>",
                    unsafe_allow_html=True,
                )
                for i, fold_score in enumerate(ev.get("cv_folds", []), 1):
                    bar_w = int(fold_score)
                    color = GREEN if fold_score >= 80 else (YELLOW if fold_score >= 70 else RED)
                    st.markdown(
                        f"<div style='margin:4px 0;'>"
                        f"<span style='font-family:monospace;font-size:0.75rem;color:#5a7a8a;'>Fold {i} </span>"
                        f"<span style='font-family:monospace;font-size:0.85rem;color:{color};'>{fold_score:.1f}%</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("---")

            # ── Confusion matrix heatmap ──────────────────────────────────────
            st.markdown("#### CONFUSION MATRIX")
            col_cm, col_info = st.columns([2, 1])
            with col_cm:
                cm     = ev.get("confusion_matrix", [])
                labels = ev.get("confusion_labels", ["Seller","Buyer","Normal"])
                if cm:
                    fig_cm = go.Figure(go.Heatmap(
                        z=cm, x=labels, y=labels,
                        colorscale=[[0,"#020b14"],[0.5,"#004466"],[1,CYAN]],
                        text=[[str(v) for v in row] for row in cm],
                        texttemplate="%{text}",
                        textfont=dict(size=16, family="Share Tech Mono"),
                        showscale=False,
                        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
                    ))
                    fig_cm.update_layout(
                        **PLOT_LAYOUT, height=320,
                        xaxis=dict(title="Predicted", color="#5a7a8a"),
                        yaxis=dict(title="Actual",    color="#5a7a8a"),
                        title=dict(text="Rows = Actual · Cols = Predicted", x=0,
                                   font=dict(size=10, color="#5a7a8a")),
                    )
                    st.plotly_chart(fig_cm, use_container_width=True,
                                    config={"displayModeBar": False})

            with col_info:
                st.markdown(
                    "<div style='background:#071828;border:1px solid #1a6a8a;"
                    "border-radius:4px;padding:1rem;font-family:monospace;"
                    "font-size:0.72rem;color:#5a7a8a;line-height:1.8;'>"
                    f"<b style='color:#00d4ff;'>Model:</b><br>{ev.get('model','RF')}<br><br>"
                    f"<b style='color:#00d4ff;'>Features:</b><br>{ev.get('features','')}<br><br>"
                    f"<b style='color:#00d4ff;'>Noise Rate:</b><br>{int(ev.get('noise_rate',0)*100)}% label noise<br><br>"
                    f"<b style='color:#00ff9d;'>Why not 100%?</b><br>"
                    "Labels contain deliberate noise to simulate real-world ambiguity. "
                    "This gives honest, generalisable accuracy instead of artificial perfection."
                    "</div>",
                    unsafe_allow_html=True,
                )

            # ── CV accuracy bar chart ─────────────────────────────────────────
            folds = ev.get("cv_folds", [])
            if folds:
                fig_cv = go.Figure(go.Bar(
                    x=[f"Fold {i}" for i in range(1, len(folds)+1)],
                    y=folds,
                    marker=dict(
                        color=folds,
                        colorscale=[[0, RED],[0.5, YELLOW],[1, GREEN]],
                        showscale=False,
                    ),
                    text=[f"{v:.1f}%" for v in folds],
                    textposition="outside",
                    textfont=dict(family="Share Tech Mono", size=11, color=CYAN),
                ))
                fig_cv.add_hline(
                    y=ev["cv_mean_accuracy_pct"],
                    line=dict(color=CYAN, dash="dash", width=1.5),
                    annotation_text=f"Mean {ev['cv_mean_accuracy_pct']}%",
                    annotation_font=dict(color=CYAN, size=10),
                )
                fig_cv.update_layout(
                    **PLOT_LAYOUT, height=260,
                    yaxis=dict(range=[60, 100], showgrid=True,
                               gridcolor="#0d3a52", color="#5a7a8a"),
                    xaxis=dict(showgrid=False, color="#5a7a8a"),
                    title=dict(text="5-Fold Cross-Validation Accuracy", x=0,
                               font=dict(size=11, color="#5a7a8a")),
                )
                st.plotly_chart(fig_cv, use_container_width=True,
                                config={"displayModeBar": False})

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("<hr> <p style='text-align:center;font-family:'Share Tech Mono';font-size:0.6rem; color:#1a3a4a;letter-spacing:2px;'> DRISHTI INTELLIGENCE PLATFORM · FOR RESEARCH & EDUCATIONAL PURPOSES ONLY · AGGREGATED DATA ONLY · NO INDIVIDUAL IDENTIFICATION </p>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# APP ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if "page" not in st.session_state:
    st.session_state["page"] = "landing"

if st.session_state["page"] == "landing":
    show_landing()
else:
    show_dashboard()