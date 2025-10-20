#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lock-In Quick Check — Streamlit (CSV-backed, fast)
- Loads league-wide game logs from a prebuilt CSV (past 2 Regular Seasons)
- Player select with typeahead
- Distribution view: histogram (density) + KDE curve + rug of data points
- Stoplight colors for the recommendation (shaded by bucket)
- No live NBA API calls
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Path to your prebuilt CSV (two prior seasons of game logs)
CSV_PATH = Path(__file__).parent / "data" / "lockin_baseline_2023-24_2024-25.csv"

# ---- Streamlit Page Config ----
st.set_page_config(
    page_title="Lock-In Quick Check",
    page_icon="✅",
    layout="centered",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)

st.title("Lock-In Quick Check")
st.caption("Sample = prior two Regular Seasons + prior current-season games")

# ---------- Data Loaders ----------
@st.cache_data(show_spinner=True)
def load_baseline(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df["FP"] = pd.to_numeric(df["FP"], errors="coerce")
    df = df.dropna(subset=["PLAYER_ID", "PLAYER_NAME", "FP"])
    return df


@st.cache_data(show_spinner=False)
def get_player_list(df: pd.DataFrame) -> List[str]:
    names = (
        df.groupby(["PLAYER_ID", "PLAYER_NAME"], as_index=False)
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
    "LOCK IT IN": ("#0E8542", "#E7F6ED"),      # deep green / very light green
    "Likely lock": ("#38A169", "#F0FBF4"),    # medium green
    "Borderline": ("#D69E2E", "#FFF8E1"),     # yellow
    "Usually pass": ("#F59E0B", "#FFF3D9"),   # amber
    "Do not lock": ("#C53030", "#FDE8E8"),    # red
    "Insufficient data": ("#4A5568", "#EDF2F7"),  # gray
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


# ---------- Simple KDE (no extra deps) ----------
def kde_curve(x: np.ndarray, points: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gaussian KDE with Silverman's bandwidth. Returns (grid_x, density_y).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        if n == 1:
            grid = np.linspace(x.min(), x.max() + 1, points)
            dens = np.full_like(grid, 1.0 / (grid.max() - grid.min()))
            return grid, dens
        return np.array([]), np.array([])

    std = np.std(x, ddof=1)
    if std == 0:
        grid = np.linspace(x.min(), x.max() + 1, points)
        dens = np.full_like(grid, 1.0 / (grid.max() - grid.min()))
        return grid, dens

    h = 1.06 * std * (n ** (-1 / 5))  # Silverman's rule
    lo = max(0, x.min() - 1 * std)    # Floor at 0 (no negatives)
    hi = x.max() + 3 * std
    grid = np.linspace(lo, hi, points)

    diffs = (grid[:, None] - x[None, :]) / h
    dens = np.exp(-0.5 * diffs**2).sum(axis=1) / (n * h * np.sqrt(2 * np.pi))
    return grid, dens


# ---------- Load Data ----------
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

    sample_note = f" (small sample, n={dist.size})" if dist.size < 8 else ""
    st.markdown(f"**Percentile:** {('NA' if pd.isna(pctl) else pctl)}{sample_note}")

    # Stoplight badge
    fg, bg = rec_colors(rec)
    st.markdown(colored_badge(rec, fg, bg), unsafe_allow_html=True)

    # ---------- Distribution: histogram + KDE + rug ----------
    st.subheader("Where this score lands")
    if dist.empty:
        st.info("No distribution available.")
    else:
        x = dist.values.astype(float)

        fig, ax = plt.subplots(figsize=(8, 3))

        # Histogram (density)
        ax.hist(x, bins="auto", density=True, alpha=0.3)

        # KDE
        grid, dens = kde_curve(x)
        if grid.size > 0:
            ax.plot(grid, dens, linewidth=2)

        # Vertical line for input score
        ax.axvline(score, linestyle="--")

        # Rug plot (sample to limit overdraw for very large n)
        if x.size > 0:
            max_d = dens.max() if grid.size > 0 else 0.02
            rug_h = max(0.02, 0.04 * max_d)
            rng = np.random.default_rng(123)
            if x.size > 400:
                rug_x = rng.choice(x, size=400, replace=False)
            else:
                rug_x = x
            for xi in rug_x:
                ax.vlines(xi, 0, rug_h, alpha=0.25)

        ax.set_xlim(left=0)  # <-- Floor at 0
        ax.set_xlabel("Fantasy Points")
        ax.set_ylabel("Density")
        plt.tight_layout()
        st.pyplot(fig)

    with st.expander("Distribution summary"):
        if not dist.empty:
            desc = pd.Series(dist).describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).rename("FP").to_frame()
            st.dataframe(desc)
        else:
            st.write("No data available.")
