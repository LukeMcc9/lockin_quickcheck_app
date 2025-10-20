#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lock-In Quick Check — Streamlit (CSV-backed, fast)
- Loads league-wide game logs from a prebuilt CSV (past 2 Regular Seasons)
- Player select with typeahead
- Beeswarm plot locating the input score within the player's distribution
- Stoplight colors for the recommendation (shaded by bucket)
- No live NBA API calls
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

CSV_PATH = Path(__file__).parent / "data" / "lockin_baseline_2023-24_2024-25.csv"

st.set_page_config(page_title="Lock-In Quick Check", page_icon="✅", layout="centered")
st.title("Lock-In Quick Check")
st.caption("Sample = prior two Regular Seasons + prior current-season games")

# ---------- Data loaders ----------
@st.cache_data(show_spinner=True)
def load_baseline(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df["FP"] = pd.to_numeric(df["FP"], errors="coerce")
    df = df.dropna(subset=["PLAYER_ID","PLAYER_NAME","FP"])
    return df

@st.cache_data(show_spinner=False)
def get_player_list(df: pd.DataFrame) -> List[str]:
    names = (
        df.groupby(["PLAYER_ID","PLAYER_NAME"], as_index=False)
          .size()
          .sort_values("PLAYER_NAME")["PLAYER_NAME"]
          .tolist()
    )
    return names

@st.cache_data(show_spinner=False)
def player_distribution(df: pd.DataFrame, player_name: str) -> pd.Series:
    s = df.loc[df["PLAYER_NAME"] == player_name, "FP"].dropna().astype(float)
    return s.reset_index(drop=True)

# ---------- Metrics ----------
@st.cache_data(show_spinner=False)
def percentile_of(x: float, dist: pd.Series) -> float:
    if dist.empty or pd.isna(x):
        return float("nan")
    return float((dist <= x).sum() * 100.0 / dist.size)

# Stoplight palette (shades per bucket)
PALETTE = {
    "LOCK IT IN":          ("#0E8542", "#E7F6ED"),  # deep green / very light green bg
    "Likely lock":         ("#38A169", "#F0FBF4"),  # medium green
    "Borderline":          ("#D69E2E", "#FFF8E1"),  # yellow
    "Usually pass":        ("#F59E0B", "#FFF3D9"),  # amber (toward red)
    "Do not lock":         ("#C53030", "#FDE8E8"),  # red
    "Insufficient data":   ("#4A5568", "#EDF2F7"),  # gray
}

def recommendation_from_percentile(p: float) -> str:
    if pd.isna(p):
        return "Insufficient data"
    if p >= 90:
        return "LOCK IT IN"
    if p >= 75:
        return "Likely lock"
    if p >= 60:
        return "Borderline"
    if p >= 40:
        return "Usually pass"
    return "Do not lock"

def rec_colors(label: str) -> Tuple[str, str]:
    if label in PALETTE:
        return PALETTE[label]
    # map close labels to keys
    if label.startswith("Usually pass"):
        return PALETTE["Usually pass"]
    return PALETTE.get("Insufficient data")

def colored_badge(text: str, fg: str, bg: str) -> str:
    return f"""
    <span style="
        background:{bg};
        color:{fg};
        padding:6px 10px;
        border-radius:10px;
        font-weight:700;
        border:1px solid rgba(0,0,0,0.05);
        ">
        {text}
    </span>
    """

# ---------- Load data ----------
if not CSV_PATH.exists():
    st.error(f"Baseline CSV not found at: {CSV_PATH}")
    st.stop()

df = load_baseline(CSV_PATH)
player_names = get_player_list(df)

# ---------- UI ----------
col1, col2 = st.columns([2, 1])
player_sel = col1.selectbox("Player", options=player_names, index=None, placeholder="Start typing a name…")
score = col2.number_input("Fantasy Score", min_value=0.0, step=0.1, format="%.2f")

go = st.button("Check")

if go:
    if not player_sel:
        st.warning("Select a player from the dropdown.")
        st.stop()

    dist = player_distribution(df, player_sel)
    pctl = round(percentile_of(score, dist), 1)
    rec = recommendation_from_percentile(pctl)

    # ---------- Result ----------
    st.subheader("Result")

    # Percentile line
    sample_note = f" (small sample, n={dist.size})" if dist.size < 8 else ""
    st.markdown(f"**Percentile:** {('NA' if pd.isna(pctl) else pctl)}{sample_note}")

    # Stoplight badge
    fg, bg = rec_colors(rec)
    st.markdown(colored_badge(rec, fg, bg), unsafe_allow_html=True)

    # ---------- Beeswarm ----------
    st.subheader("Where this score lands")
    if dist.empty:
        st.info("No distribution available.")
    else:
        x = dist.values.astype(float)
        rng = np.random.default_rng(42)
        y = rng.normal(loc=0.0, scale=0.04, size=len(x))

        fig, ax = plt.subplots(figsize=(8, 2.2))
        ax.scatter(x, y, alpha=0.5, s=14)
        ax.axvline(score, linestyle='--')
        ax.set_yticks([])
        ax.set_xlabel("Fantasy Points")

        try:
            q10, q25, q50, q75, q90 = np.percentile(x, [10, 25, 50, 75, 90])
            for v in [q10, q25, q50, q75, q90]:
                ax.axvline(v, alpha=0.15)
            ax.text(score, 0.08, f"Input: {score:.1f}", ha='center')
        except Exception:
            pass

        plt.tight_layout()
        st.pyplot(fig)

    with st.expander("Distribution summary"):
        if not dist.empty:
            desc = pd.Series(dist).describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).rename("FP").to_frame()
            st.dataframe(desc)
        else:
            st.write("No data available.")
