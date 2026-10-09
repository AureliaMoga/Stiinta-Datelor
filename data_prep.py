"""
step2_prepare_data.py
=====================
Aggregates monthly LLM scores to quarterly averages,
merges with real economic data (quarterly), creates lagged variables.

Inputs:
    outputs/part2_output_md_<SOURCE>.csv
    outputs/part2_output_ro_<SOURCE>.csv
    real_data_md.csv
    real_data_ro.csv

Output:
    outputs/regression_data_md.csv
    outputs/regression_data_ro.csv
"""

import os
import pandas as pd
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────
SOURCE_MD = "zdg"
SOURCE_RO = "hotnews"
OUT_DIR   = "./outputs"
# ─────────────────────────────────────────────────────────────

LLM_SCORE_COLS = [
    "sentiment_index",
    "economic_growth_score",
    "inflation_score",
    "unemployment_score",
    "interest_rates_score",
    "exchange_rate_score",
]

REAL_COLS = [
    "gdp_yoy_pct",
    "inflation_yoy_pct",
    "unemployment_pct",
    "interest_rate_pct",
    "exchange_rate_yoy_chg_pct",
]


def month_to_quarter(date_str):
    """Convert '2019-01' → '2019-Q1'."""
    ts = pd.Timestamp(date_str + "-01")
    return f"{ts.year}-Q{ts.quarter}"


def aggregate_to_quarterly(df_monthly):
    """Average monthly LLM scores within each quarter."""
    df = df_monthly.copy()
    df["quarter"] = df["date"].apply(month_to_quarter)
    quarterly = df.groupby("quarter")[LLM_SCORE_COLS].mean().reset_index()
    quarterly = quarterly.sort_values("quarter").reset_index(drop=True)
    return quarterly


def prepare_regression_data(tag, source):
    """Load, aggregate, merge, lag, save."""
    part2_path = os.path.join(OUT_DIR, f"part2_output_{tag}_{source}.csv")
    real_path  = f"real_data_{tag}.csv"

    if not os.path.exists(part2_path):
        print(f"  [ERROR] Missing: {part2_path}")
        return None
    if not os.path.exists(real_path):
        print(f"  [ERROR] Missing: {real_path}")
        return None

    # Load
    df_llm  = pd.read_csv(part2_path)
    df_real = pd.read_csv(real_path)

    print(f"\n{tag.upper()} — LLM months: {len(df_llm)}, Real quarters: {len(df_real)}")

    # Aggregate monthly → quarterly
    df_q = aggregate_to_quarterly(df_llm)
    print(f"  LLM quarterly records: {len(df_q)}")

    # Merge with real data on quarter key
    df_merged = pd.merge(df_real[["quarter"] + REAL_COLS],
                         df_q, on="quarter", how="inner")
    df_merged = df_merged.sort_values("quarter").reset_index(drop=True)
    print(f"  Merged records (before lag): {len(df_merged)}")

    # Create lagged LLM variables (t-1 quarter predicts real data at t)
    for col in LLM_SCORE_COLS:
        df_merged[f"{col}_lag1"] = df_merged[col].shift(1)

    # Drop rows where lag is NaN (first row) or real GDP is NaN (2026-Q2)
    df_merged = df_merged.dropna(subset=[f"{c}_lag1" for c in LLM_SCORE_COLS])

    print(f"  Final regression-ready rows: {len(df_merged)}")

    out_path = os.path.join(OUT_DIR, f"regression_data_{tag}.csv")
    df_merged.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")
    return df_merged


def main():
    for tag, source in [("md", SOURCE_MD), ("ro", SOURCE_RO)]:
        prepare_regression_data(tag, source)


if __name__ == "__main__":
    main()