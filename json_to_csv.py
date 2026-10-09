"""
step1_json_to_csv.py
====================
Reads monthly LLM sentiment JSON files from sentiment_md/ and sentiment_ro/
and produces the required part2_output CSVs.

JSON format expected (one file per month, e.g. 2019-01.json):
{
    "sentiment_index":  float (-2 to +2),
    "economic_growth":  float (-5 to +5),
    "inflation":        float (-5 to +5),
    "unemployment":     float (-5 to +5),
    "interest_rates":   float (-5 to +5),
    "exchange_rate":    float (-5 to +5),
    "sentiment_reasoning":   str,
    "indicators_reasoning":  str
}

Output:
    outputs/part2_output_md_<SOURCE>.csv
    outputs/part2_output_ro_<SOURCE>.csv
"""

import os
import json
import glob
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────
SOURCE_MD = "zdg"       # change to your actual MD source name
SOURCE_RO = "hotnews"      # change to your actual RO source name
DIR_MD    = "/home/aureliaaa/PycharmProjects/MLM_RO_MD/sentiment_md"
DIR_RO    = "/home/aureliaaa/PycharmProjects/MLM_RO_MD/sentiment_ro"
OUT_DIR   = "./outputs"
os.makedirs(OUT_DIR, exist_ok=True)
# ─────────────────────────────────────────────────────────────


def load_jsons(folder):
    """Load all .json/.JSON files from folder, return sorted DataFrame."""
    records = []
    files = sorted(glob.glob(os.path.join(folder, "*.json")) +
                   glob.glob(os.path.join(folder, "*.JSON")))

    if not files:
        print(f"  [WARN] No JSON files found in '{folder}'.")
        return pd.DataFrame()

    for fpath in files:
        stem = os.path.basename(fpath).replace(".JSON", "").replace(".json", "")
        try:
            date = pd.Timestamp(stem + "-01")
        except Exception:
            print(f"  [SKIP] Cannot parse date from filename: {fpath}")
            continue
        try:
            with open(fpath, encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            print(f"  [SKIP] Cannot read {fpath}: {e}")
            continue

        records.append({
            "date":                  date.strftime("%Y-%m"),
            "sentiment_index":       float(d.get("sentiment_index",  0)),
            "economic_growth_score": float(d.get("economic_growth",  0)),
            "inflation_score":       float(d.get("inflation",         0)),
            "unemployment_score":    float(d.get("unemployment",      0)),
            "interest_rates_score":  float(d.get("interest_rates",    0)),
            "exchange_rate_score":   float(d.get("exchange_rate",     0)),
        })

    df = pd.DataFrame(records)
    print(f"  Loaded {len(df)} records from '{folder}'")
    return df


def main():
    for folder, source, tag in [
        (DIR_MD, SOURCE_MD, "md"),
        (DIR_RO, SOURCE_RO, "ro"),
    ]:
        print(f"\nProcessing {tag.upper()} ({source}) ...")
        df = load_jsons(folder)
        if df.empty:
            print(f"  [SKIP] No data for {tag.upper()}.")
            continue

        out_path = os.path.join(OUT_DIR, f"part2_output_{tag}_{source}.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved → {out_path}  ({len(df)} rows)")
        print(df[["date","sentiment_index","economic_growth_score",
                   "inflation_score","unemployment_score"]].head(3).to_string(index=False))


if __name__ == "__main__":
    main()