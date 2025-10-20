#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Builds a baseline CSV of fantasy points for the previous two Regular Seasons (league-wide).

Usage:
  python build_lockin_baseline_csv.py --season 2025-26 --out data/lockin_baseline_2023-24_2024-25.csv
"""

import argparse, sys, time
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.library.http import NBAStatsHTTP

NBAStatsHTTP.timeout = 10

NUMERIC_COLS = [
    "MIN",
    "PTS",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FTM",
    "FTA",
    "OREB",
    "DREB",
]

WEIGHTS = {
    "PTS": 1.0,
    "REB": 1.0,
    "AST": 1.4,
    "STL": 2.4,
    "BLK": 2.0,
    "TOV": -2.0,
    "DD2": 2.0,
    "TD3": 4.0,
    "FGM": 1.0,
    "MISS_FG": -0.5,
    "FTM": 0.5,
    "MISS_FT": -0.1,
    "FG3M": 0.5,
    "MISS_3": -0.1,
    "OREB": 0.5,
}


def prev_two(season_str: str):
    start = int(season_str[:4])
    s1 = f"{start-2}-{str((start-1) % 100).zfill(2)}"
    s2 = f"{start-1}-{str(start % 100).zfill(2)}"
    return s1, s2


def safe_league_log(
    season: str, stype: str = "Regular Season", pause: float = 0.7, retries: int = 3
) -> pd.DataFrame:
    last_err = None
    for i in range(retries):
        try:
            time.sleep(pause)
            gl = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=stype,
                player_or_team_abbreviation="P",
            )
            df = gl.get_data_frames()[0].copy()
            df["SEASON"] = season
            return df
        except Exception as e:
            last_err = e
            time.sleep(pause * (1.6**i))
    raise last_err


def compute_dd2_td3(df: pd.DataFrame) -> pd.DataFrame:
    """Manually add DD2 and TD3 columns based on core box stats."""
    df = df.copy()
    core = ["PTS", "REB", "AST", "STL", "BLK"]
    for c in core:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    hit = df[core].ge(10).sum(axis=1)
    df["DD2"] = (hit >= 2).astype(int)
    df["TD3"] = (hit >= 3).astype(int)
    return df


def compute_fp(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    for c in NUMERIC_COLS:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    miss_fg = df["FGA"] - df["FGM"]
    miss_3 = df["FG3A"] - df["FG3M"]
    miss_ft = df["FTA"] - df["FTM"]
    fp = (
        df["PTS"] * WEIGHTS["PTS"]
        + df["REB"] * WEIGHTS["REB"]
        + df["AST"] * WEIGHTS["AST"]
        + df["STL"] * WEIGHTS["STL"]
        + df["BLK"] * WEIGHTS["BLK"]
        + df["TOV"] * WEIGHTS["TOV"]
        + df["DD2"] * WEIGHTS["DD2"]
        + df["TD3"] * WEIGHTS["TD3"]
        + df["FGM"] * WEIGHTS["FGM"]
        + miss_fg * WEIGHTS["MISS_FG"]
        + df["FTM"] * WEIGHTS["FTM"]
        + miss_ft * WEIGHTS["MISS_FT"]
        + df["FG3M"] * WEIGHTS["FG3M"]
        + miss_3 * WEIGHTS["MISS_3"]
        + df["OREB"] * WEIGHTS["OREB"]
    )
    return fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--season",
        required=True,
        help="Target season like 2025-26 (we will pull the prior two seasons).",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output CSV path, e.g., data/lockin_baseline_2023-24_2024-25.csv",
    )
    ap.add_argument("--pause", type=float, default=0.7)
    args = ap.parse_args()

    s1, s2 = prev_two(args.season)
    print(f"Pulling league logs for {s1} and {s2} (Regular Season)…")

    df1 = safe_league_log(s1, pause=args.pause)
    df2 = safe_league_log(s2, pause=args.pause)

    # Add DD2 and TD3 manually
    df1 = compute_dd2_td3(df1)
    df2 = compute_dd2_td3(df2)

    # Compute FP
    for d in (df1, df2):
        d["FP"] = compute_fp(d)

    keep = [
        "SEASON",
        "GAME_DATE",
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ABBREVIATION",
        "PTS",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "FGM",
        "FGA",
        "FG3M",
        "FG3A",
        "FTM",
        "FTA",
        "OREB",
        "DREB",
        "DD2",
        "TD3",
        "FP",
    ]
    out = pd.concat([df1[keep], df2[keep]], ignore_index=True)
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"])

    out.sort_values(["PLAYER_ID", "GAME_DATE"], inplace=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out):,} rows → {args.out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
