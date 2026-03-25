🛰️ DRISHTI — AI-Powered Drug Peddling Detection and Analysis System

Data-Driven Regional Intelligence for Suspicious Hotspot-based Trafficking Identification


📌 About
DRISHTI is a modular AI-powered surveillance system that monitors public social media data to detect, classify, and map potential drug trafficking activity. It collects live posts from Reddit, processes them through a multi-stage NLP and machine learning pipeline, and visualises the intelligence on an interactive cyber-style dashboard mapped to Indian cities.
Built for research and educational purposes only.

🧠 What It Does

🔴 Collects live public Reddit posts every 3 minutes using the JSON API
🧹 Preprocesses text while preserving emojis as meaningful signals
🔍 Detects suspicious codewords across 4 signal categories — Selling, Buying, Secrecy, and Drug References — with a weighted suspicion score
🤖 Classifies users as Seller / Buyer / Normal using a Random Forest model trained on TF-IDF + 12 numeric features
👥 Clusters users behaviourally using KMeans to reveal potential distribution networks
🗺️ Visualises everything on a real-time Streamlit dashboard with geographic heatmaps, network graphs, spike detection, and predictive risk forecasting


⚙️ Tech Stack
LayerTechnologyLanguagePython 3.10+Data CollectionReddit JSON API + requestsData Processingpandas, numpyNLP & MLscikit-learn — TF-IDF, Random Forest, KMeansSparse MatrixscipyDashboardstreamlitChartsplotlyNetwork Graphnetworkx

🗂️ Project Structure
drishti/
├── dashboard.py                        # Streamlit dashboard
├── run_pipeline.py                     # Runs full pipeline
├── seed_demo_data.py                   # Generates synthetic demo data
├── requirements.txt
│
├── scripts/
│   ├── collect_live_data.py            # Stage 1 — Reddit data collection
│   ├── preprocess_dataset.py           # Stage 2 — Text cleaning
│   ├── detect_codewords.py             # Stage 3 — Suspicion scoring
│   ├── train_classification_model.py   # Stage 4 — ML classification
│   └── cluster_users.py               # Stage 5 — User clustering
│
├── data/                               # Auto-generated CSV files
└── models/                            # Saved ML model artifacts

🚀 Quick Start
Option A — Instant Demo (no Reddit needed)
bashpip install -r requirements.txt
streamlit run dashboard.py
Option B — Full Live Pipeline
bashpip install -r requirements.txt

# Terminal 1 — collect live data continuously
python scripts/collect_live_data.py

# Terminal 2 — run pipeline and launch dashboard
python run_pipeline.py
streamlit run dashboard.py
Option C — Seed + Pipeline
bashpip install -r requirements.txt
python seed_demo_data.py
python run_pipeline.py
streamlit run dashboard.py
```

---

## 📊 Pipeline Flow
```
Reddit API (every 3 mins)
        ↓
collect_live_data.py   →   raw_posts.csv
        ↓
preprocess_dataset.py  →   processed_posts.csv
        ↓
detect_codewords.py    →   scored_posts.csv
        ↓
train_classification_model.py  →  classified_posts.csv
        ↓
cluster_users.py       →   clustered_posts.csv
        ↓
streamlit dashboard    →   Live Intelligence Console

🔒 Privacy

Usernames are never stored — replaced with MD5-hashed anonymous IDs at point of collection
Raw post content is never displayed on the dashboard
Only aggregated regional metrics and anonymised user IDs are visualised
City-level granularity only — no precise geolocation


⚠️ Limitations

English language only — no Hindi or regional language support currently
Location data is randomly assigned from a predefined Indian city list
Trained on heuristically labelled data — no expert-annotated ground truth
Reddit only — private platforms like Telegram and WhatsApp are out of scope


🔭 Future Scope

Multi-platform support — Telegram, Instagram, X
Transformer-based NLP — BERT / RoBERTa for contextual understanding
Real geolocation via Named Entity Recognition
Multilingual detection — Hindi, Kannada, Tamil, Telugu
Automated pipeline scheduling
Real-time investigator alerts


📄 License
This project is built for academic and research purposes only.
Not intended for commercial use or actual law enforcement deployment.
