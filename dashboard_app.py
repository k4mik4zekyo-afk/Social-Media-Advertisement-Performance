"""Ad campaign dashboard: tabs for Main (KPIs, map, $ velocity), locations, campaigns.

Run: streamlit run dashboard_app.py
DB: ad_campaign_db.sqlite or STREAMLIT_AD_DB_PATH
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from verify_schema import validate_schema

DEFAULT_DB = Path(__file__).resolve().parent / "ad_campaign_db.sqlite"
MAP_WIDGET_KEY = "country_map"
PENDING_MAP_KEY = "pending_territory_from_map"

BASE_JOIN = """
FROM ad_events e
JOIN users u ON e.user_id = u.user_id
JOIN ads a ON e.ad_id = a.ad_id
JOIN campaigns c ON a.campaign_id = c.campaign_id
WHERE e.timestamp >= ? AND e.timestamp <= ?
"""


def resolve_db_path() -> Path:
    return Path(os.environ.get("STREAMLIT_AD_DB_PATH", str(DEFAULT_DB)))


def connect_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)


def _selection_points(widget_state) -> list:
    if widget_state is None:
        return []
    try:
        sel = widget_state["selection"] if isinstance(widget_state, dict) else widget_state.selection
    except (KeyError, AttributeError):
        return []
    if sel is None:
        return []
    pts = sel["points"] if isinstance(sel, dict) else getattr(sel, "points", None)
    if not pts:
        return []
    return list(pts)


def country_from_point(pt: dict) -> str | None:
    cd = pt.get("customdata")
    if cd is not None:
        if isinstance(cd, (list, tuple)) and len(cd) > 0:
            v = cd[0]
            if v is not None and str(v).strip():
                return str(v).strip()
        elif isinstance(cd, str) and cd.strip():
            return cd
    loc = pt.get("location")
    if isinstance(loc, str) and loc.strip():
        return loc
    return None


@st.cache_data(ttl=300, show_spinner="Loading date range…")
def fetch_date_bounds(path_str: str) -> tuple[str, str]:
    path = Path(path_str)
    conn = connect_ro(path)
    try:
        row = conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM ad_events").fetchone()
    finally:
        conn.close()
    if not row or row[0] is None:
        return "2025-01-01 00:00:00", "2025-12-31 23:59:59"
    return str(row[0]), str(row[1])


@st.cache_data(ttl=300, show_spinner="Loading country metrics…")
def fetch_country_metrics(path_str: str, start: str, end: str) -> pd.DataFrame:
    path = Path(path_str)
    sql = f"""
    SELECT
      u.country AS country,
      SUM(CASE WHEN e.event_type = 'Impression' THEN 1 ELSE 0 END) AS impressions,
      SUM(CASE WHEN e.event_type = 'Click' THEN 1 ELSE 0 END) AS clicks,
      SUM(CASE WHEN e.event_type = 'Purchase' THEN 1 ELSE 0 END) AS purchases,
      SUM(CASE WHEN e.event_type = 'Like' THEN 1 ELSE 0 END) AS likes,
      SUM(CASE WHEN e.event_type = 'Comment' THEN 1 ELSE 0 END) AS comments,
      SUM(CASE WHEN e.event_type = 'Share' THEN 1 ELSE 0 END) AS shares,
      COUNT(DISTINCT c.campaign_id) AS campaigns_reached
    {BASE_JOIN}
    GROUP BY u.country
    """
    conn = connect_ro(path)
    try:
        df = pd.read_sql_query(sql, conn, params=[start, end])
    finally:
        conn.close()
    imp = df["impressions"].replace(0, pd.NA)
    clk = df["clicks"].replace(0, pd.NA)
    df["ctr_pct"] = (df["clicks"] / imp * 100).fillna(0.0)
    df["purchase_rate_imp_pct"] = (df["purchases"] / imp * 100).fillna(0.0)
    df["purchase_rate_click_pct"] = (df["purchases"] / clk * 100).fillna(0.0)
    return df


@st.cache_data(ttl=300, show_spinner="Loading attributed budget…")
def fetch_country_budget(path_str: str, start: str, end: str) -> pd.DataFrame:
    path = Path(path_str)
    sql = f"""
    SELECT u.country AS country, c.campaign_id, MAX(c.total_budget) AS budget
    {BASE_JOIN}
    GROUP BY u.country, c.campaign_id
    """
    conn = connect_ro(path)
    try:
        df = pd.read_sql_query(sql, conn, params=[start, end])
    finally:
        conn.close()
    if df.empty:
        return pd.DataFrame(columns=["country", "budget_sum"])
    out = df.groupby("country", as_index=False)["budget"].sum()
    out.rename(columns={"budget": "budget_sum"}, inplace=True)
    return out


@st.cache_data(ttl=300, show_spinner="Loading global campaign spend…")
def fetch_global_campaign_spend(path_str: str, start: str, end: str) -> float:
    """Sum each campaign's budget once for campaigns active in the event filter window."""
    path = Path(path_str)
    sql = """
    SELECT SUM(sub.budget) AS total_spend
    FROM (
      SELECT c.campaign_id AS campaign_id, MAX(c.total_budget) AS budget
      FROM ad_events e
      JOIN ads a ON e.ad_id = a.ad_id
      JOIN campaigns c ON a.campaign_id = c.campaign_id
      WHERE e.timestamp >= ? AND e.timestamp <= ?
      GROUP BY c.campaign_id
    ) sub
    """
    conn = connect_ro(path)
    try:
        row = conn.execute(sql, [start, end]).fetchone()
    finally:
        conn.close()
    if not row or row[0] is None:
        return 0.0
    return float(row[0])


@st.cache_data(ttl=300, show_spinner="Loading campaigns…")
def fetch_campaigns_meta(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    conn = connect_ro(path)
    try:
        df = pd.read_sql_query(
            """
            SELECT campaign_id, name, start_date, end_date, duration_days, total_budget
            FROM campaigns
            """,
            conn,
        )
    finally:
        conn.close()
    return df


@st.cache_data(ttl=300, show_spinner="Building daily spend curve…")
def fetch_daily_prorated_spend(path_str: str, start: str, end: str) -> pd.DataFrame:
    """Linearly prorate each campaign's total_budget across its duration; sum per calendar day."""
    camps = fetch_campaigns_meta(path_str)
    filter_start = pd.Timestamp(start).normalize()
    filter_end = pd.Timestamp(end).normalize()
    if filter_end < filter_start:
        return pd.DataFrame(columns=["day", "spend"])
    days_idx = pd.date_range(filter_start, filter_end, freq="D")
    spend = pd.Series(0.0, index=days_idx, dtype=float)
    for _, row in camps.iterrows():
        try:
            c_start = pd.Timestamp(row["start_date"]).normalize()
            c_end = pd.Timestamp(row["end_date"]).normalize()
        except Exception:
            continue
        dur = int(row["duration_days"]) if pd.notna(row["duration_days"]) and int(row["duration_days"]) > 0 else max(
            (c_end - c_start).days + 1, 1
        )
        budget = float(row["total_budget"]) if pd.notna(row["total_budget"]) else 0.0
        daily_amt = budget / float(dur)
        for d in days_idx:
            if c_start <= d <= c_end:
                spend.loc[d] += daily_amt
    out = spend.reset_index()
    out.columns = ["day", "spend"]
    out["day"] = out["day"].dt.strftime("%Y-%m-%d")
    return out


@st.cache_data(ttl=300, show_spinner="Loading funnel…")
def fetch_funnel(path_str: str, start: str, end: str, country: str | None) -> pd.Series:
    path = Path(path_str)
    filt = ""
    params: list = [start, end]
    if country:
        filt = " AND u.country = ?"
        params.append(country)
    sql = f"""
    SELECT
      SUM(CASE WHEN e.event_type = 'Impression' THEN 1 ELSE 0 END) AS impressions,
      SUM(CASE WHEN e.event_type = 'Click' THEN 1 ELSE 0 END) AS clicks,
      SUM(CASE WHEN e.event_type = 'Purchase' THEN 1 ELSE 0 END) AS purchases,
      SUM(CASE WHEN e.event_type = 'Like' THEN 1 ELSE 0 END) AS likes,
      SUM(CASE WHEN e.event_type = 'Comment' THEN 1 ELSE 0 END) AS comments,
      SUM(CASE WHEN e.event_type = 'Share' THEN 1 ELSE 0 END) AS shares
    {BASE_JOIN}
    {filt}
    """
    conn = connect_ro(path)
    try:
        df = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()
    return df.iloc[0]


@st.cache_data(ttl=300, show_spinner="Loading daily events…")
def fetch_daily_metrics_global(path_str: str, start: str, end: str) -> pd.DataFrame:
    path = Path(path_str)
    sql = f"""
    SELECT date(e.timestamp) AS day,
      SUM(CASE WHEN e.event_type = 'Purchase' THEN 1 ELSE 0 END) AS purchases,
      SUM(CASE WHEN e.event_type = 'Click' THEN 1 ELSE 0 END) AS clicks,
      SUM(CASE WHEN e.event_type = 'Impression' THEN 1 ELSE 0 END) AS impressions
    {BASE_JOIN}
    GROUP BY date(e.timestamp)
    ORDER BY day
    """
    conn = connect_ro(path)
    try:
        df = pd.read_sql_query(sql, conn, params=[start, end])
    finally:
        conn.close()
    return df


@st.cache_data(ttl=300, show_spinner="Loading journey times…")
def fetch_user_journey_days(path_str: str, start: str, end: str, country: str | None) -> pd.DataFrame:
    path = Path(path_str)
    country_f = ""
    params = [start, end]
    if country:
        country_f = " AND u.country = ?"
        params.append(country)
    sql = f"""
    WITH ev AS (
      SELECT e.user_id AS user_id, e.event_type AS event_type, e.timestamp AS ts
      FROM ad_events e
      JOIN users u ON e.user_id = u.user_id
      WHERE e.timestamp >= ? AND e.timestamp <= ? {country_f}
    ),
    per_user AS (
      SELECT user_id,
        MIN(CASE WHEN event_type = 'Impression' THEN ts END) AS first_imp,
        MIN(CASE WHEN event_type = 'Purchase' THEN ts END) AS first_pur
      FROM ev
      GROUP BY user_id
      HAVING first_imp IS NOT NULL AND first_pur IS NOT NULL AND first_pur >= first_imp
    )
    SELECT u.country AS country, u.location AS location,
      (julianday(p.first_pur) - julianday(p.first_imp)) AS days_to_purchase
    FROM per_user p
    JOIN users u ON u.user_id = p.user_id
    """
    conn = connect_ro(path)
    try:
        df = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()
    return df


@st.cache_data(ttl=300, show_spinner="Loading location performance…")
def fetch_location_stats(
    path_str: str, start: str, end: str, country: str, min_impressions: int
) -> pd.DataFrame:
    path = Path(path_str)
    sql = f"""
    SELECT
      u.location AS location,
      SUM(CASE WHEN e.event_type = 'Impression' THEN 1 ELSE 0 END) AS impressions,
      SUM(CASE WHEN e.event_type = 'Click' THEN 1 ELSE 0 END) AS clicks,
      SUM(CASE WHEN e.event_type = 'Purchase' THEN 1 ELSE 0 END) AS purchases
    {BASE_JOIN}
    AND u.country = ?
    GROUP BY u.location
    HAVING impressions >= ?
    """
    conn = connect_ro(path)
    try:
        df = pd.read_sql_query(sql, conn, params=[start, end, country, min_impressions])
    finally:
        conn.close()
    imp = df["impressions"].replace(0, pd.NA)
    clk = df["clicks"].replace(0, pd.NA)
    df["purchase_rate_imp_pct"] = (df["purchases"] / imp * 100).fillna(0.0)
    df["purchase_rate_click_pct"] = (df["purchases"] / clk * 100).fillna(0.0)
    return df


@st.cache_data(ttl=300, show_spinner="Loading zero-impression locations…")
def fetch_locations_zero_impressions(
    path_str: str, start: str, end: str, country: str, max_rows: int = 50
) -> pd.DataFrame:
    path = Path(path_str)
    sql = """
    WITH locs AS (
      SELECT DISTINCT location AS location FROM users WHERE country = ?
    ),
    imp AS (
      SELECT u.location AS location,
        SUM(CASE WHEN e.event_type = 'Impression' THEN 1 ELSE 0 END) AS impressions
      FROM ad_events e
      JOIN users u ON e.user_id = u.user_id
      JOIN ads a ON e.ad_id = a.ad_id
      JOIN campaigns c ON a.campaign_id = c.campaign_id
      WHERE e.timestamp >= ? AND e.timestamp <= ? AND u.country = ?
      GROUP BY u.location
    )
    SELECT l.location AS location,
      COUNT(DISTINCT u.user_id) AS users_in_user_table
    FROM locs l
    LEFT JOIN imp i ON l.location = i.location
    JOIN users u ON u.country = ? AND u.location = l.location
    WHERE COALESCE(i.impressions, 0) = 0
    GROUP BY l.location
    ORDER BY users_in_user_table DESC, l.location
    LIMIT ?
    """
    conn = connect_ro(path)
    try:
        df = pd.read_sql_query(
            sql, conn, params=[country, start, end, country, country, max_rows]
        )
    finally:
        conn.close()
    return df


@st.cache_data(ttl=300, show_spinner="Loading campaign metrics…")
def fetch_campaign_metrics_table(path_str: str, start: str, end: str) -> pd.DataFrame:
    path = Path(path_str)
    sql = f"""
    SELECT
      c.campaign_id AS campaign_id,
      MAX(c.name) AS name,
      MAX(c.total_budget) AS total_budget,
      SUM(CASE WHEN e.event_type = 'Impression' THEN 1 ELSE 0 END) AS impressions,
      SUM(CASE WHEN e.event_type = 'Click' THEN 1 ELSE 0 END) AS clicks,
      SUM(CASE WHEN e.event_type = 'Purchase' THEN 1 ELSE 0 END) AS purchases
    {BASE_JOIN}
    GROUP BY c.campaign_id
    ORDER BY c.campaign_id
    """
    conn = connect_ro(path)
    try:
        df = pd.read_sql_query(sql, conn, params=[start, end])
    finally:
        conn.close()
    if df.empty:
        return df
    imp = df["impressions"].replace(0, np.nan)
    clk = df["clicks"].replace(0, np.nan)
    pur = df["purchases"].replace(0, np.nan)
    df["ctr_pct"] = (df["clicks"] / imp * 100).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["conversion_rate_pct"] = (df["purchases"] / clk * 100).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["cpa"] = df.apply(
        lambda r: (float(r["total_budget"]) / float(r["purchases"])) if r["purchases"] and r["purchases"] > 0 else np.nan,
        axis=1,
    )
    return df


@st.cache_data(ttl=300, show_spinner="Loading ad / targeting rollups…")
def fetch_ad_strategy_granular(path_str: str, start: str, end: str) -> pd.DataFrame:
    """One row per (targeting slice × campaign) with events and campaign budget."""
    path = Path(path_str)
    sql = """
    SELECT
      a.ad_platform AS ad_platform,
      a.ad_type AS ad_type,
      a.target_gender AS target_gender,
      a.target_age_group AS target_age_group,
      a.campaign_id AS campaign_id,
      MAX(c.total_budget) AS campaign_budget,
      SUM(CASE WHEN e.event_type = 'Impression' THEN 1 ELSE 0 END) AS impressions,
      SUM(CASE WHEN e.event_type = 'Click' THEN 1 ELSE 0 END) AS clicks,
      SUM(CASE WHEN e.event_type = 'Purchase' THEN 1 ELSE 0 END) AS purchases
    FROM ad_events e
    JOIN ads a ON e.ad_id = a.ad_id
    JOIN campaigns c ON a.campaign_id = c.campaign_id
    WHERE e.timestamp >= ? AND e.timestamp <= ?
    GROUP BY a.ad_platform, a.ad_type, a.target_gender, a.target_age_group, a.campaign_id
    """
    conn = connect_ro(path)
    try:
        df = pd.read_sql_query(sql, conn, params=[start, end])
    finally:
        conn.close()
    return df


def rollup_ad_strategy(granular: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to targeting dimensions; sum budget once per campaign per slice."""
    if granular.empty:
        return pd.DataFrame(
            columns=[
                "ad_platform",
                "ad_type",
                "target_gender",
                "target_age_group",
                "impressions",
                "clicks",
                "purchases",
                "attributed_spend",
                "ctr_pct",
                "conversion_rate_pct",
                "purchase_rate_imp_pct",
                "cpa",
            ]
        )
    gcols = ["ad_platform", "ad_type", "target_gender", "target_age_group"]
    per = granular.groupby(gcols + ["campaign_id"], as_index=False).agg(
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        purchases=("purchases", "sum"),
        campaign_budget=("campaign_budget", "max"),
    )
    out = per.groupby(gcols, as_index=False).agg(
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        purchases=("purchases", "sum"),
        attributed_spend=("campaign_budget", "sum"),
    )
    imp = out["impressions"].replace(0, np.nan)
    clk = out["clicks"].replace(0, np.nan)
    out["ctr_pct"] = (out["clicks"] / imp * 100).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["purchase_rate_imp_pct"] = (out["purchases"] / imp * 100).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["conversion_rate_pct"] = (out["purchases"] / clk * 100).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pur = out["purchases"].replace(0, np.nan)
    out["cpa"] = (out["attributed_spend"] / pur).replace([np.inf, -np.inf], np.nan)
    return out


@st.cache_data(ttl=300, show_spinner="Loading campaign training features…")
def fetch_campaign_training_frame(path_str: str, start: str, end: str) -> pd.DataFrame:
    cdf = fetch_campaign_metrics_table(path_str, start, end)
    if cdf.empty:
        return cdf
    meta = fetch_campaigns_meta(path_str)
    m = cdf.merge(
        meta[["campaign_id", "duration_days", "name"]],
        on="campaign_id",
        how="left",
    )
    m["duration_days"] = m["duration_days"].fillna(0).astype(float)
    m["purchase_rate_imp"] = np.where(
        m["impressions"] > 0, m["purchases"] / m["impressions"], 0.0
    )
    return m


def build_quarterly_rollup(
    daily_spend: pd.DataFrame, daily_events: pd.DataFrame
) -> pd.DataFrame:
    """Merge prorated spend and daily events; aggregate by calendar quarter."""
    if daily_spend.empty and daily_events.empty:
        return pd.DataFrame(
            columns=["quarter", "spend", "impressions", "clicks", "purchases", "ctr_pct", "conversion_rate_pct", "cpa"]
        )
    if daily_spend.empty:
        m = daily_events.copy()
        m["day"] = pd.to_datetime(m["day"])
        m["spend"] = 0.0
    elif daily_events.empty:
        m = daily_spend.copy()
        m["day"] = pd.to_datetime(m["day"])
        m["impressions"] = 0
        m["clicks"] = 0
        m["purchases"] = 0
    else:
        ds = daily_spend.copy()
        ds["day"] = pd.to_datetime(ds["day"])
        de = daily_events.copy()
        de["day"] = pd.to_datetime(de["day"])
        m = pd.merge(ds, de, on="day", how="outer").sort_values("day")
        m["spend"] = m["spend"].fillna(0.0)
    for col in ("impressions", "clicks", "purchases"):
        if col not in m.columns:
            m[col] = 0
        m[col] = m[col].fillna(0).astype(int)
    m["quarter"] = m["day"].dt.to_period("Q").astype(str)
    agg = m.groupby("quarter", as_index=False).agg(
        spend=("spend", "sum"),
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        purchases=("purchases", "sum"),
    )
    agg["ctr_pct"] = np.where(agg["impressions"] > 0, agg["clicks"] / agg["impressions"] * 100, 0.0)
    agg["conversion_rate_pct"] = np.where(agg["clicks"] > 0, agg["purchases"] / agg["clicks"] * 100, 0.0)
    agg["cpa"] = np.where(agg["purchases"] > 0, agg["spend"] / agg["purchases"], np.nan)
    return agg


def add_country_cpa_columns(cmetrics: pd.DataFrame) -> pd.DataFrame:
    df = cmetrics.copy()
    pur = df["purchases"].replace(0, np.nan)
    df["cpa"] = (df["budget_sum"] / pur).replace([np.inf, -np.inf], np.nan)
    df["cpa_display"] = df["cpa"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "—")
    return df


def sort_countries_by_cpa(map_df: pd.DataFrame, cap: int = 15) -> pd.DataFrame:
    d = map_df.copy()
    d["_sort_cpa"] = d["cpa"].replace([np.inf, -np.inf], np.nan)
    d = d.sort_values("_sort_cpa", ascending=True, na_position="last")
    return d.drop(columns=["_sort_cpa"]).head(cap)


def render_sidebar(
    path: Path, dmin, dmax
) -> tuple[str, str, str, int, int, str, pd.DataFrame, list[str]]:
    with st.sidebar:
        st.header("Filters")
        dr = st.date_input("Event date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
        if isinstance(dr, tuple) and len(dr) == 2:
            start_d, end_d = dr
        else:
            start_d, end_d = dmin, dmax
        start_s = pd.Timestamp(start_d).strftime("%Y-%m-%d 00:00:00")
        end_s = pd.Timestamp(end_d).strftime("%Y-%m-%d 23:59:59")

        st.subheader("Main map")
        main_map_metric = st.selectbox(
            "Choropleth color",
            options=[
                ("cpa", "Cost per acquisition (CPA)"),
                ("purchase_rate_imp_pct", "Purchase rate (% of impressions)"),
                ("ctr_pct", "CTR (% of impressions)"),
                ("impressions", "Impressions"),
                ("purchases", "Purchases"),
            ],
            format_func=lambda x: x[1],
            key="main_map_color",
        )[0]

        roll_days = st.number_input("Rolling avg window (days, Main tab)", min_value=1, value=7, step=1)

        st.subheader("Locations tab")
        min_imp = st.number_input("Min impressions (location tables)", min_value=0, value=1, step=1)
        rank_mode = st.radio(
            "Location rank by",
            options=[("imp", "Purchases ÷ impressions"), ("click", "Purchases ÷ clicks")],
            format_func=lambda x: x[1],
        )[0]

        countries_df = fetch_country_metrics(str(path), start_s, end_s)
        sorted_countries = sorted(countries_df["country"].dropna().unique().tolist())
        labels = ["All countries"] + sorted_countries
        st.selectbox("Territory (sidebar)", labels, key="territory_select")
        st.divider()
        st.caption("Map selection updates territory for the Locations tab.")
        return start_s, end_s, main_map_metric, int(roll_days), min_imp, rank_mode, countries_df, labels


def render_tab_main(
    path: Path,
    start_s: str,
    end_s: str,
    choice: str,
    main_map_metric: str,
    roll_days: int,
) -> None:
    funnel_g = fetch_funnel(str(path), start_s, end_s, None)
    spend_g = fetch_global_campaign_spend(str(path), start_s, end_s)
    fj_g = fetch_user_journey_days(str(path), start_s, end_s, None)

    impv = int(funnel_g["impressions"])
    clv = int(funnel_g["clicks"])
    puv = int(funnel_g["purchases"])
    ctr = (clv / impv * 100) if impv else 0.0
    cpa_g = (spend_g / puv) if puv else None
    med_txt = f"{float(fj_g['days_to_purchase'].median()):.1f} d" if len(fj_g) else "—"
    start_dt = pd.Timestamp(start_s)
    end_dt = pd.Timestamp(end_s)
    days_span = max((end_dt.normalize() - start_dt.normalize()).days + 1, 1)
    weeks_span = max(days_span / 7.0, 1e-9)
    avg_weekly_purchases = puv / weeks_span
    mean_cycle_days = float(fj_g["days_to_purchase"].mean()) if len(fj_g) else None
    pace_ratio = (
        (avg_weekly_purchases / mean_cycle_days)
        if mean_cycle_days is not None and mean_cycle_days > 0
        else None
    )
    pace_display = f"{pace_ratio:.4f}" if pace_ratio is not None else "—"

    st.subheader("Overview KPIs (all territories in date range)")
    st.caption(
        "CPA = attributed spend ÷ purchases. Spend sums each campaign’s budget once if it had events in this window."
    )
    r1c = st.columns(4)
    r1c[0].metric("Impressions", f"{impv:,}")
    r1c[1].metric(
        "Click-through rate (CTR)",
        f"{clv:,} ({ctr:.2f}%)",
    )
    r1c[2].metric("Purchases", f"{puv:,}")
    r1c[3].metric("Attributed spend", f"${spend_g:,.2f}")
    r2c = st.columns(4)
    r2c[0].metric("Cost per acquisition (CPA)", f"${cpa_g:,.2f}" if cpa_g is not None else "—")
    r2c[1].metric("Purchase rate (% imp.)", f"{(puv / impv * 100) if impv else 0.0:.3f}%")
    r2c[2].metric("Median days (imp→purchase)", med_txt)
    r2c[3].metric("Avg weekly purchases ÷ avg cycle (days)", pace_display)
    st.caption(
        "Last KPI: total purchases ÷ weeks in range ÷ mean impression→purchase days (users with both events). "
        "Interprets weekly purchase pace relative to average funnel length."
    )

    countries_df = fetch_country_metrics(str(path), start_s, end_s)
    bud = fetch_country_budget(str(path), start_s, end_s)
    cmetrics = countries_df.merge(bud, on="country", how="left") if not bud.empty else countries_df.copy()
    if "budget_sum" not in cmetrics.columns:
        cmetrics["budget_sum"] = 0.0
    cmetrics["budget_sum"] = cmetrics["budget_sum"].fillna(0.0)
    map_df = add_country_cpa_columns(cmetrics)

    st.subheader("Top countries by CPA (lowest first, max 15)")
    ranked_src = sort_countries_by_cpa(map_df, cap=15).copy()
    ranked_src["CTR"] = ranked_src["ctr_pct"].map(lambda x: f"{x:.2f}%")
    ranked_src["Purchase rate"] = ranked_src["purchase_rate_imp_pct"].map(lambda x: f"{x:.3f}%")
    ranked = ranked_src[
        ["country", "impressions", "clicks", "purchases", "CTR", "Purchase rate", "cpa_display"]
    ].rename(columns={"cpa_display": "CPA"})
    st.dataframe(ranked, width="stretch", hide_index=True)
    st.caption(
        "Countries with no purchases sort last. CTR and purchase rate are share of impressions; CPA uses attributed spend per country."
    )

    color_col = main_map_metric if main_map_metric in map_df.columns else "cpa"
    if color_col == "cpa":
        map_plot = map_df.copy()
        s = map_plot["cpa"]
        if s.notna().any():
            cap = float(s.quantile(0.95))
            map_plot["cpa_plot"] = s.clip(lower=0, upper=cap)
        else:
            map_plot["cpa_plot"] = s
        color_col_use = "cpa_plot"
    else:
        map_plot = map_df
        color_col_use = color_col

    hover_cols = {
        "impressions": True,
        "clicks": True,
        "purchases": True,
        "purchase_rate_imp_pct": ":.3f",
        "ctr_pct": ":.2f",
        "budget_sum": ":,.0f",
        "cpa_display": True,
    }
    cd_cols = [
        "country",
        "impressions",
        "clicks",
        "purchases",
        "purchase_rate_imp_pct",
        "ctr_pct",
        "budget_sum",
        "cpa_display",
    ]

    fig_map = px.choropleth(
        map_plot,
        locations="country",
        locationmode="country names",
        color=color_col_use,
        hover_name="country",
        hover_data=hover_cols,
        custom_data=cd_cols,
        color_continuous_scale="Blues",
        projection="natural earth",
        title="Territory (select a country to drill down on the Locations tab)",
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=520)

    event = st.plotly_chart(
        fig_map,
        width="stretch",
        key=f"{MAP_WIDGET_KEY}_{choice}_main",
        on_select="rerun",
        selection_mode="points",
    )
    pts = _selection_points(event)
    c_map = country_from_point(pts[0]) if pts else None
    if c_map and choice != c_map:
        st.session_state[PENDING_MAP_KEY] = c_map
        st.rerun()

    st.subheader("Sales velocity (prorated spend, USD)")
    st.caption("Each campaign’s budget is spread evenly over its duration; chart shows daily total and rolling average.")
    daily_spend = fetch_daily_prorated_spend(str(path), start_s, end_s)
    if daily_spend.empty:
        st.info("No days in range.")
    else:
        daily_spend = daily_spend.copy()
        daily_spend["day_dt"] = pd.to_datetime(daily_spend["day"])
        daily_spend["spend_rolling"] = daily_spend["spend"].rolling(window=roll_days, min_periods=1).mean()
        fig_v = go.Figure()
        fig_v.add_trace(
            go.Scatter(
                x=daily_spend["day_dt"],
                y=daily_spend["spend"],
                name="Daily spend ($)",
                mode="lines",
                opacity=0.35,
            )
        )
        fig_v.add_trace(
            go.Scatter(
                x=daily_spend["day_dt"],
                y=daily_spend["spend_rolling"],
                name=f"{roll_days}-day rolling avg ($)",
                mode="lines",
            )
        )
        fig_v.update_layout(
            height=400,
            yaxis=dict(title="USD"),
            legend=dict(orientation="h"),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_v, width="stretch")


def render_tab_locations(
    path: Path, start_s: str, end_s: str, active_country: str | None, min_imp: int, rank_mode: str
) -> None:
    st.subheader("Country / location drill-down")
    if active_country is None:
        st.info("Choose a **territory** in the sidebar or select a country on the **Main** map.")
        return
    locdf = fetch_location_stats(str(path), start_s, end_s, active_country, int(min_imp))
    if locdf.empty:
        st.warning("No locations meet the minimum impression threshold for this country.")
    else:
        sort_col = "purchase_rate_click_pct" if rank_mode == "click" else "purchase_rate_imp_pct"
        top = locdf.nlargest(10, sort_col)[
            ["location", "impressions", "clicks", "purchases", "purchase_rate_imp_pct", "purchase_rate_click_pct"]
        ]
        bot = locdf.nsmallest(10, sort_col)[
            ["location", "impressions", "clicks", "purchases", "purchase_rate_imp_pct", "purchase_rate_click_pct"]
        ]
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Top 10 locations**")
            st.dataframe(top, width="stretch", hide_index=True)
            st.plotly_chart(px.bar(top, x="location", y=sort_col, title="Top 10"), width="stretch")
        with t2:
            st.markdown("**Bottom 10 locations**")
            st.dataframe(bot, width="stretch", hide_index=True)
            st.plotly_chart(px.bar(bot, x="location", y=sort_col, title="Bottom 10"), width="stretch")

    zero_loc = fetch_locations_zero_impressions(str(path), start_s, end_s, active_country)
    st.markdown("**Locations with zero impressions** (in selected date range; up to 50 rows)")
    if zero_loc.empty:
        st.caption("No such locations, or all catalog locations received at least one impression.")
    else:
        st.dataframe(zero_loc, width="stretch", hide_index=True)


def _campaign_table_styler(cdf_sorted: pd.DataFrame, top_ids: set):
    disp = cdf_sorted.copy()

    def _highlight(row: pd.Series) -> list[str]:
        if row["campaign_id"] in top_ids:
            return ["background-color: rgba(34, 139, 34, 0.45)"] * len(row)
        return [""] * len(row)

    disp["ctr_pct"] = disp["ctr_pct"].map(lambda x: f"{x:.2f}%")
    disp["conversion_rate_pct"] = disp["conversion_rate_pct"].map(lambda x: f"{x:.2f}%")
    disp["cpa"] = disp["cpa"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "—")
    disp["total_budget"] = disp["total_budget"].map(lambda x: f"${float(x):,.2f}")
    show = disp[
        ["campaign_id", "name", "total_budget", "impressions", "clicks", "purchases", "ctr_pct", "conversion_rate_pct", "cpa"]
    ].rename(
        columns={
            "ctr_pct": "CTR",
            "conversion_rate_pct": "Conversion (clicks→purchase)",
            "total_budget": "Budget",
        }
    )
    return show.style.apply(_highlight, axis=1)


def render_tab_campaigns(path: Path, start_s: str, end_s: str) -> None:
    st.subheader("Campaign performance")
    st.caption(
        "Sorted by **CPA (lowest first)** — best economics first. "
        "Green rows: lowest CPA among campaigns with at least one purchase. "
        "Conversion rate = purchases ÷ clicks; CPA = budget ÷ purchases."
    )
    cdf = fetch_campaign_metrics_table(str(path), start_s, end_s)
    if cdf.empty:
        st.info("No campaign activity in this date range.")
        return
    cdf_sorted = cdf.sort_values("cpa", ascending=True, na_position="last").reset_index(drop=True)
    has_cpa = cdf_sorted["cpa"].notna() & (cdf_sorted["purchases"] > 0)
    n_best = min(5, int(has_cpa.sum()))
    best_rows = cdf_sorted[has_cpa].head(n_best)
    top_ids = set(best_rows["campaign_id"].tolist())
    if top_ids:
        names = best_rows["name"].tolist()
        st.success(f"**Top {len(names)} by CPA (lowest):** " + " · ".join(names))

    try:
        styled = _campaign_table_styler(cdf_sorted, top_ids)
        st.dataframe(styled, width="stretch", hide_index=True)
    except Exception:
        show = cdf_sorted.copy()
        show["ctr_pct"] = show["ctr_pct"].map(lambda x: f"{x:.2f}%")
        show["conversion_rate_pct"] = show["conversion_rate_pct"].map(lambda x: f"{x:.2f}%")
        show["cpa"] = show["cpa"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "—")
        show["total_budget"] = show["total_budget"].map(lambda x: f"${float(x):,.2f}")
        st.dataframe(
            show[
                [
                    "campaign_id",
                    "name",
                    "total_budget",
                    "impressions",
                    "clicks",
                    "purchases",
                    "ctr_pct",
                    "conversion_rate_pct",
                    "cpa",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

    st.subheader("Quarterly aggregates (all campaigns)")
    daily_spend = fetch_daily_prorated_spend(str(path), start_s, end_s)
    daily_events = fetch_daily_metrics_global(str(path), start_s, end_s)
    qdf = build_quarterly_rollup(daily_spend, daily_events)
    if qdf.empty:
        st.info("No quarterly data in range.")
    else:
        qshow = qdf.copy()
        qshow["ctr_pct"] = qshow["ctr_pct"].map(lambda x: f"{x:.2f}%")
        qshow["conversion_rate_pct"] = qshow["conversion_rate_pct"].map(lambda x: f"{x:.2f}%")
        qshow["cpa"] = qshow["cpa"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "—")
        qshow["spend"] = qshow["spend"].map(lambda x: f"${x:,.2f}")
        st.dataframe(
            qshow.rename(columns={"spend": "Attributed spend ($)", "conversion_rate_pct": "Conversion (clk→pur)"}),
            width="stretch",
            hide_index=True,
        )
        fig_q = px.bar(qdf, x="quarter", y="spend", title="Quarterly attributed spend ($)")
        st.plotly_chart(fig_q, width="stretch")


def _strategy_row_key(row: pd.Series) -> tuple:
    return (
        str(row["ad_platform"]),
        str(row["ad_type"]),
        str(row["target_gender"]),
        str(row["target_age_group"]),
    )


def render_tab_targeting(path: Path, start_s: str, end_s: str) -> None:
    st.subheader("Ad type and targeting")
    st.caption(
        "Each row is a **platform × ad type × gender target × age target** slice. "
        "Attributed spend sums campaign budgets (once per campaign per slice). "
        "Table sorted by **CPA (lowest first)**; green highlights the **5 best** slices with purchases."
    )
    min_imp = st.number_input(
        "Minimum impressions per slice",
        min_value=0,
        value=200,
        step=50,
        key="targeting_min_imp",
    )
    gran = fetch_ad_strategy_granular(str(path), start_s, end_s)
    strat = rollup_ad_strategy(gran)
    if strat.empty:
        st.info("No ad-level activity in this date range.")
        return
    strat = strat[strat["impressions"] >= int(min_imp)].copy()
    if strat.empty:
        st.warning("No targeting slices meet the impression floor.")
        return
    strat_sorted = strat.sort_values("cpa", ascending=True, na_position="last").reset_index(drop=True)
    valid = strat_sorted["cpa"].notna() & (strat_sorted["purchases"] > 0)
    n_best = min(5, int(valid.sum()))
    best_slices = strat_sorted[valid].head(n_best)
    top_keys = {_strategy_row_key(r) for _, r in best_slices.iterrows()}

    disp = strat_sorted.copy()

    def _hl(row: pd.Series) -> list[str]:
        return (
            ["background-color: rgba(34, 139, 34, 0.45)"] * len(row)
            if _strategy_row_key(row) in top_keys
            else [""] * len(row)
        )

    disp["ctr_pct"] = disp["ctr_pct"].map(lambda x: f"{x:.2f}%")
    disp["conversion_rate_pct"] = disp["conversion_rate_pct"].map(lambda x: f"{x:.2f}%")
    disp["purchase_rate_imp_pct"] = disp["purchase_rate_imp_pct"].map(lambda x: f"{x:.3f}%")
    disp["cpa"] = disp["cpa"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "—")
    disp["attributed_spend"] = disp["attributed_spend"].map(lambda x: f"${float(x):,.2f}")
    show_cols = [
        "ad_platform",
        "ad_type",
        "target_gender",
        "target_age_group",
        "impressions",
        "clicks",
        "purchases",
        "ctr_pct",
        "conversion_rate_pct",
        "purchase_rate_imp_pct",
        "attributed_spend",
        "cpa",
    ]
    try:
        st.dataframe(disp[show_cols].style.apply(_hl, axis=1), width="stretch", hide_index=True)
    except Exception:
        st.dataframe(disp[show_cols], width="stretch", hide_index=True)

    st.subheader("CPA by ad platform (rolled up)")
    plat = (
        strat_sorted.groupby("ad_platform", as_index=False)
        .agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            purchases=("purchases", "sum"),
            attributed_spend=("attributed_spend", "sum"),
        )
    )
    plat["cpa"] = np.where(plat["purchases"] > 0, plat["attributed_spend"] / plat["purchases"], np.nan)
    plat = plat.sort_values("cpa", ascending=True, na_position="last")
    plat_v = plat.dropna(subset=["cpa"])
    if not plat_v.empty:
        fig_p = px.bar(plat_v, x="ad_platform", y="cpa", title="Lower is better — CPA by platform")
        fig_p.update_layout(yaxis_title="CPA ($)")
        st.plotly_chart(fig_p, width="stretch")
    else:
        st.caption("No platform-level CPA (no purchases in roll-up).")

    st.subheader("CPA by ad type (rolled up)")
    typ = (
        strat_sorted.groupby("ad_type", as_index=False)
        .agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            purchases=("purchases", "sum"),
            attributed_spend=("attributed_spend", "sum"),
        )
    )
    typ["cpa"] = np.where(typ["purchases"] > 0, typ["attributed_spend"] / typ["purchases"], np.nan)
    typ = typ.sort_values("cpa", ascending=True, na_position="last")
    typ_v = typ.dropna(subset=["cpa"])
    if not typ_v.empty:
        fig_t = px.bar(typ_v, x="ad_type", y="cpa", title="Lower is better — CPA by creative type")
        fig_t.update_layout(yaxis_title="CPA ($)")
        st.plotly_chart(fig_t, width="stretch")
    else:
        st.caption("No ad-type CPA (no purchases in roll-up).")


def render_tab_models(path: Path, start_s: str, end_s: str) -> None:
    st.subheader("Predictive models (exploratory)")
    st.caption(
        "Simple fits on the **selected date range** only. Use for directional insight, not production forecasting."
    )

    train = fetch_campaign_training_frame(str(path), start_s, end_s)
    if train.empty or len(train) < 10:
        st.info("Need at least ~10 campaigns with activity in this window to train reliably.")
        return

    st.markdown("### Model 1 — Purchases from budget and duration")
    st.caption("Linear regression: numeric features only.")
    X1 = train[["total_budget", "duration_days"]].astype(float).values
    y1 = train["purchases"].astype(float).values
    if np.unique(y1).size < 2 or len(train) < 12:
        st.warning("Not enough variation or rows for a held-out test; showing in-sample fit only.")
        m1 = LinearRegression().fit(X1, y1)
        pred_eval = m1.predict(X1)
        y_a, y_b = y1, pred_eval
        st.write(f"In-sample R²: {r2_score(y1, pred_eval):.3f}, MAE: {mean_absolute_error(y1, pred_eval):.2f}")
    else:
        Xtr, Xte, ytr, yte = train_test_split(X1, y1, test_size=0.25, random_state=42)
        m1 = LinearRegression().fit(Xtr, ytr)
        pred_eval = m1.predict(Xte)
        y_a, y_b = yte, pred_eval
        st.write(f"Hold-out R²: {r2_score(yte, pred_eval):.3f}, MAE: {mean_absolute_error(yte, pred_eval):.2f}")
    st.write(
        f"**Coefficients** (budget, duration days): `{m1.coef_[0]:.6f}`, `{m1.coef_[1]:.6f}` — intercept `{m1.intercept_:.2f}`"
    )
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=y_a, y=y_b, mode="markers", name="Predicted vs actual"))
    lo = float(min(y_a.min(), y_b.min()))
    hi = float(max(y_a.max(), y_b.max()))
    fig1.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="Perfect fit",
            line=dict(dash="dash"),
        )
    )
    fig1.update_layout(
        title="Model 1: actual vs predicted purchases",
        xaxis_title="Actual purchases",
        yaxis_title="Predicted purchases",
        height=360,
    )
    st.plotly_chart(fig1, width="stretch")

    st.markdown("### Model 2 — Purchase rate from targeting mix")
    st.caption(
        "Gradient boosting on one-hot platform, ad type, gender target, age target (slice-level rows)."
    )
    gran = fetch_ad_strategy_granular(str(path), start_s, end_s)
    strat = rollup_ad_strategy(gran)
    strat = strat[strat["impressions"] >= 100].copy()
    if len(strat) < 12:
        st.warning("Not enough targeting slices (≥100 impressions each) for Model 2 in this window.")
        return
    cat_cols = ["ad_platform", "ad_type", "target_gender", "target_age_group"]
    X2 = strat[cat_cols].astype(str)
    y2 = strat["purchase_rate_imp_pct"].astype(float).values
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", max_categories=30, sparse_output=False), cat_cols)],
        remainder="drop",
    )
    pipe = Pipeline(
        [
            ("pre", pre),
            (
                "gb",
                GradientBoostingRegressor(
                    random_state=42,
                    max_depth=4,
                    n_estimators=120,
                    learning_rate=0.08,
                ),
            ),
        ]
    )
    if len(strat) < 20 or np.unique(y2).size < 3:
        pipe.fit(X2, y2)
        pred2 = pipe.predict(X2)
        st.write(f"In-sample R²: {r2_score(y2, pred2):.3f}, MAE: {mean_absolute_error(y2, pred2):.3f} (pp)")
    else:
        Xtr2, Xte2, ytr2, yte2 = train_test_split(X2, y2, test_size=0.25, random_state=42)
        pipe.fit(Xtr2, ytr2)
        pred2 = pipe.predict(Xte2)
        st.write(f"Hold-out R²: {r2_score(yte2, pred2):.3f}, MAE: {mean_absolute_error(yte2, pred2):.3f} (pp)")

    imp = pipe.named_steps["gb"].feature_importances_
    enc = pipe.named_steps["pre"].named_transformers_["cat"]
    names = enc.get_feature_names_out(cat_cols)
    top_idx = np.argsort(imp)[::-1][:15]
    imp_df = pd.DataFrame({"feature": names[top_idx], "importance": imp[top_idx]})
    st.markdown("**Top feature importances (one-hot segments)**")
    st.dataframe(imp_df, width="stretch", hide_index=True)
    fig2 = px.bar(imp_df.iloc[::-1], x="importance", y="feature", orientation="h", title="Model 2 — importance")
    st.plotly_chart(fig2, width="stretch")


def main() -> None:
    st.set_page_config(page_title="Ad performance", layout="wide")
    if "territory_select" not in st.session_state:
        st.session_state.territory_select = "All countries"
    pending = st.session_state.pop(PENDING_MAP_KEY, None)
    if pending is not None:
        if pending == "__ALL__":
            st.session_state.territory_select = "All countries"
        else:
            st.session_state.territory_select = pending

    db = resolve_db_path()
    if not db.is_file():
        st.error(f"Database not found: {db}")
        st.stop()
    schema_errors = validate_schema(db)
    if schema_errors:
        st.error("Database schema does not match expected CSV layout:")
        for line in schema_errors:
            st.code(line, language=None)
        st.stop()

    tmin, tmax = fetch_date_bounds(str(db))
    dmin = pd.to_datetime(tmin).date()
    dmax = pd.to_datetime(tmax).date()

    st.title("Social ad performance")
    st.caption("Territory analysis, funnel-style metrics, campaign economics, and spend velocity.")

    start_s, end_s, main_map_metric, roll_days, min_imp, rank_mode, _countries_df, _labels = render_sidebar(
        db, dmin, dmax
    )
    choice = st.session_state.territory_select
    active_country = None if choice == "All countries" else choice

    tab_main, tab_loc, tab_camp, tab_tgt, tab_mdl = st.tabs(
        ["Main", "Country location data", "Campaigns", "Ad targeting", "Predictive models"]
    )
    with tab_main:
        render_tab_main(db, start_s, end_s, choice, main_map_metric, roll_days)
    with tab_loc:
        render_tab_locations(db, start_s, end_s, active_country, min_imp, rank_mode)
    with tab_camp:
        render_tab_campaigns(db, start_s, end_s)
    with tab_tgt:
        render_tab_targeting(db, start_s, end_s)
    with tab_mdl:
        render_tab_models(db, start_s, end_s)

    st.divider()
    st.caption(
        "Per-country attributed spend allocates full campaign budget to each geo that saw activity; global totals de-duplicate by campaign."
    )


if __name__ == "__main__":
    main()
