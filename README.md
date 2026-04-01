# Social Media Advertisement Performance Dashboard

An interactive Streamlit dashboard for analyzing social media ad campaign performance. Explore KPIs, geographic breakdowns, campaign economics, ad targeting effectiveness, and predictive models — all from a single browser tab.

## Data Source

The dataset used in this project is **simulated** and sourced from Kaggle:

> **[Social Media Advertisement Performance](https://www.kaggle.com/datasets/alperenmyung/social-media-advertisement-performance/data?select=users.csv)**
> by alperenmyung — License: **CC0: Public Domain**

The raw CSVs live in `archive/` and are loaded into a SQLite database (`ad_campaign_db.sqlite`) for fast querying:

| File | Rows | Description |
|------|------|-------------|
| `users.csv` | 10,000 | User demographics — gender, age group, country, location, interests |
| `campaigns.csv` | 50 | Campaign metadata — name, dates, duration, total budget |
| `ads.csv` | 200 | Ad configuration — platform, creative type, targeting criteria |
| `ad_events.csv` | 400,000 | Event log — impressions, clicks, purchases, likes, comments, shares |

## Dashboard Tabs

| Tab | What it shows |
|-----|---------------|
| **Main** | Global KPIs (impressions, CTR, purchases, CPA), interactive choropleth map, daily spend velocity with rolling average |
| **Country Location Data** | Drill-down into top/bottom 10 locations by purchase rate for a selected country; locations with zero impressions |
| **Campaigns** | Per-campaign table sorted by CPA, quarterly spend aggregates, best-performing campaigns highlighted |
| **Ad Targeting** | Performance by platform x ad type x gender x age group slices; CPA roll-ups by platform and creative type |
| **Predictive Models** | Linear regression (purchases from budget + duration) and gradient boosting (purchase rate from targeting mix) with feature importances |

## Getting Started

### Prerequisites

- Python 3.10+

### Installation

```bash
pip install -r requirements.txt
```

### Running the Dashboard

```bash
streamlit run dashboard_app.py
```

The app reads from `ad_campaign_db.sqlite` by default. To point to a different database, set the environment variable:

```bash
STREAMLIT_AD_DB_PATH=/path/to/your.db streamlit run dashboard_app.py
```

### Schema Verification

A helper script validates that the SQLite schema matches the expected CSV column layout:

```bash
python verify_schema.py
```

## Project Structure

```
.
├── README.md
├── requirements.txt          # Python dependencies
├── dashboard_app.py          # Streamlit application (all tabs, queries, models)
├── verify_schema.py          # DB schema validation utility
├── ad_campaign_db.sqlite     # Pre-built SQLite database
└── archive/                  # Raw CSV data files
    ├── users.csv
    ├── campaigns.csv
    ├── ads.csv
    └── ad_events.csv
```

## Key Dependencies

- **Streamlit** — web UI and interactive widgets
- **Pandas** — data manipulation
- **Plotly** — interactive charts and choropleth maps
- **scikit-learn** — predictive models (linear regression, gradient boosting)
