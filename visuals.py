"""
step5_graphs.py
===============
Generates all publication-ready graphs for the assignment report:

  1. Sentiment Index over time (MD + RO on same plot)
  2. LLM scores vs real indicator over time (one plot per indicator per country)
  3. Predicted vs Actual — Model A & B (one plot per indicator per country)
  4. Scatter plots — LLM score vs real indicator value
  5. R² comparison bar chart — Model A vs Model B

Inputs:
    outputs/regression_data_md.csv
    outputs/regression_data_ro.csv
    outputs/regression_results_md.csv
    outputs/regression_results_ro.csv

Output:
    outputs/graphs/  (PNG files, 150 dpi)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────
OUT_DIR    = "./outputs"
GRAPHS_DIR = os.path.join(OUT_DIR, "graphs")
os.makedirs(GRAPHS_DIR, exist_ok=True)

SOURCE_MD = "zdg"
SOURCE_RO = "hotnews"

COUNTRY_COLORS = {
    "Moldova": {"main": "#1a6faf", "light": "#a8d1f0", "train": "#1a6faf", "test": "#e05c2a"},
    "Romania": {"main": "#c0392b", "light": "#f5b7b1", "train": "#c0392b", "test": "#27ae60"},
}

INDICATOR_MAP = {
    "gdp_yoy_pct":               ("GDP Growth",    "YoY %",     "economic_growth_score"),
    "inflation_yoy_pct":         ("Inflation",     "YoY %",     "inflation_score"),
    "unemployment_pct":          ("Unemployment",  "%",         "unemployment_score"),
    "interest_rate_pct":         ("Interest Rate", "%",         "interest_rates_score"),
    "exchange_rate_yoy_chg_pct": ("Exchange Rate", "YoY chg %", "exchange_rate_score"),
}

LLM_IND_COLS_LAG = [
    "economic_growth_score_lag1",
    "inflation_score_lag1",
    "unemployment_score_lag1",
    "interest_rates_score_lag1",
    "exchange_rate_score_lag1",
]
SENT_LAG = "sentiment_index_lag1"

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "grid.linestyle":  "--",
    "figure.dpi":      150,
})


def load_data():
    dfs = {}
    res = {}
    for tag, country in [("md", "Moldova"), ("ro", "Romania")]:
        dp = os.path.join(OUT_DIR, f"regression_data_{tag}.csv")
        rp = os.path.join(OUT_DIR, f"regression_results_{tag}.csv")
        if os.path.exists(dp):
            dfs[country] = pd.read_csv(dp)
        if os.path.exists(rp):
            res[country] = pd.read_csv(rp)
    return dfs, res


def train_test_split_ts(df, train_frac=0.8):
    n = len(df)
    split = int(np.ceil(n * train_frac))
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def quarter_xticks(ax, quarters, step=4):
    """Show every `step` quarter labels on x axis."""
    idxs = range(0, len(quarters), step)
    ax.set_xticks(list(idxs))
    ax.set_xticklabels([quarters[i] for i in idxs], rotation=45, ha="right", fontsize=8)


# ══════════════════════════════════════════════════════════════
# GRAPH 1 — Sentiment Index over time (MD + RO)
# ══════════════════════════════════════════════════════════════
def graph1_sentiment(dfs):
    fig, ax = plt.subplots(figsize=(13, 4.5))

    for country, df in dfs.items():
        col  = COUNTRY_COLORS[country]
        quarters = df["quarter"].tolist()
        x    = range(len(quarters))
        sent = df["sentiment_index"].values
        ax.plot(x, sent, color=col["main"], linewidth=2,
                marker="o", markersize=3.5, label=country)
        ax.fill_between(x, 0, sent,
                        where=sent >= 0, alpha=0.12, color=col["main"])
        ax.fill_between(x, 0, sent,
                        where=sent < 0,  alpha=0.12, color="#e74c3c")

    # Use MD quarters for x-axis (same range)
    sample_df = list(dfs.values())[0]
    quarters  = sample_df["quarter"].tolist()
    quarter_xticks(ax, quarters, step=4)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.axhline( 1, color="gray", linewidth=0.5, linestyle=":")
    ax.axhline(-1, color="gray", linewidth=0.5, linestyle=":")
    ax.set_ylim(-2.3, 2.3)
    ax.set_ylabel("Sentiment Index (-2 to +2)", fontsize=10)
    ax.set_title("LLM-Derived Sentiment Index over Time — Moldova & Romania",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=10)
    # Shade COVID and energy crisis periods
    q_list = quarters
    def qi(q):
        try: return q_list.index(q)
        except: return None
    for start, end, label, color in [
        ("2020-Q1", "2020-Q4", "COVID-19", "#f39c12"),
        ("2022-Q1", "2023-Q1", "Energy Crisis", "#8e44ad"),
    ]:
        s, e = qi(start), qi(end)
        if s is not None and e is not None:
            ax.axvspan(s, e, alpha=0.07, color=color, label=label)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=9, loc="upper right")

    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "1_sentiment_index_over_time.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
# GRAPH 2 — LLM score vs real indicator over time
# ══════════════════════════════════════════════════════════════
def graph2_scores_vs_real(dfs):
    for country, df in dfs.items():
        col      = COUNTRY_COLORS[country]
        quarters = df["quarter"].tolist()
        x        = range(len(quarters))

        fig, axes = plt.subplots(5, 1, figsize=(13, 18), sharex=True)
        fig.suptitle(f"LLM Scores vs Real Indicators — {country}",
                     fontsize=13, fontweight="bold", y=1.01)

        for ax, (real_col, (label, unit, llm_col)) in zip(axes, INDICATOR_MAP.items()):
            real_vals = df[real_col].values if real_col in df.columns else np.full(len(x), np.nan)
            llm_vals  = df[llm_col].values  if llm_col  in df.columns else np.full(len(x), np.nan)

            ax2 = ax.twinx()
            ax.plot(x, real_vals, color=col["main"], linewidth=2,
                    label=f"Real {label} ({unit})", zorder=3)
            ax2.bar(x, llm_vals, color=col["light"], alpha=0.6, width=0.7,
                    label=f"LLM Score (-5 to +5)", zorder=2)
            ax2.axhline(0, color="gray", linewidth=0.5)
            ax2.set_ylim(-6, 6)
            ax2.set_ylabel("LLM Score", fontsize=8, color="gray")

            ax.set_ylabel(f"{label}\n({unit})", fontsize=9)
            ax.set_title(label, fontsize=10, fontweight="bold")

            lines1, labs1 = ax.get_legend_handles_labels()
            lines2, labs2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper right")

        quarter_xticks(axes[-1], quarters, step=4)
        plt.tight_layout()
        fname = f"2_scores_vs_real_{country.lower()}.png"
        path  = os.path.join(GRAPHS_DIR, fname)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
# GRAPH 3 — Predicted vs Actual (Model A & B)
# ══════════════════════════════════════════════════════════════
def graph3_predicted_vs_actual(dfs):
    for country, df in dfs.items():
        col   = COUNTRY_COLORS[country]
        train, test = train_test_split_ts(df)

        fig, axes = plt.subplots(5, 2, figsize=(15, 20))
        fig.suptitle(f"Predicted vs Actual — {country}\n(dashed line = train/test split)",
                     fontsize=13, fontweight="bold", y=1.01)

        for row_i, (real_col, (label, unit, _)) in enumerate(INDICATOR_MAP.items()):
            for col_i, (model_name, feat_cols) in enumerate([
                ("Model A (5 Indicator Scores)", LLM_IND_COLS_LAG),
                ("Model B (Sentiment Index)",    [SENT_LAG]),
            ]):
                ax = axes[row_i][col_i]

                tr = train.dropna(subset=[real_col])
                te = test.dropna(subset=[real_col])

                if len(tr) < 3:
                    ax.set_visible(False)
                    continue

                X_tr = tr[feat_cols].values
                y_tr = tr[real_col].values
                X_te = te[feat_cols].values if len(te) > 0 else np.zeros((0, len(feat_cols)))
                y_te = te[real_col].values  if len(te) > 0 else np.array([])

                reg = LinearRegression().fit(X_tr, y_tr)
                pred_tr = reg.predict(X_tr)
                pred_te = reg.predict(X_te) if len(X_te) > 0 else np.array([])

                r2_tr = r2_score(y_tr, pred_tr)
                r2_te = r2_score(y_te, pred_te) if len(y_te) >= 2 else np.nan

                n_tr = len(y_tr)
                x_tr = range(n_tr)
                x_te = range(n_tr, n_tr + len(y_te))

                ax.plot(x_tr, y_tr,    "o-", color=col["train"],  linewidth=1.8,
                        markersize=4, label="Actual (train)")
                ax.plot(x_tr, pred_tr, "s--", color=col["light"], linewidth=1.5,
                        markersize=3, label="Predicted (train)")
                if len(y_te) > 0:
                    ax.plot(x_te, y_te,    "o-", color=col["test"],  linewidth=1.8,
                            markersize=4, label="Actual (test)")
                    ax.plot(x_te, pred_te, "s--", color="#f0a500", linewidth=1.5,
                            markersize=3, label="Predicted (test)")
                    ax.axvline(n_tr - 0.5, color="gray", linestyle=":", linewidth=1.2)

                r2_te_str = f"{r2_te:.3f}" if not np.isnan(r2_te) else "N/A"
                ax.set_title(f"{label} | {model_name}\nR²_train={r2_tr:.3f}  R²_test={r2_te_str}",
                             fontsize=8.5, fontweight="bold")
                ax.set_ylabel(f"{label} ({unit})", fontsize=8)
                ax.legend(fontsize=7, loc="upper right")

        plt.tight_layout()
        fname = f"3_predicted_vs_actual_{country.lower()}.png"
        path  = os.path.join(GRAPHS_DIR, fname)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
# GRAPH 4 — Scatter: LLM score vs real indicator
# ══════════════════════════════════════════════════════════════
def graph4_scatter(dfs):
    for country, df in dfs.items():
        col = COUNTRY_COLORS[country]

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle(f"Scatter: LLM Scores vs Real Indicators — {country}",
                     fontsize=13, fontweight="bold")
        axes_flat = axes.flatten()

        # First subplot: sentiment vs average of all real cols
        plots = [("sentiment_index", "gdp_yoy_pct",
                  "Sentiment Index", "GDP Growth (YoY %)")]
        for real_col, (label, unit, llm_col) in INDICATOR_MAP.items():
            plots.append((llm_col, real_col, f"LLM {label} Score", f"Real {label} ({unit})"))

        for i, (x_col, y_col, x_label, y_label) in enumerate(plots[:6]):
            ax = axes_flat[i]
            if x_col not in df.columns or y_col not in df.columns:
                ax.set_visible(False)
                continue

            subset = df[[x_col, y_col]].dropna()
            x_vals = subset[x_col].values
            y_vals = subset[y_col].values

            ax.scatter(x_vals, y_vals, color=col["main"], alpha=0.65,
                       edgecolors="white", linewidth=0.5, s=55, zorder=3)

            # Trend line
            if len(x_vals) > 2:
                z    = np.polyfit(x_vals, y_vals, 1)
                p    = np.poly1d(z)
                xfit = np.linspace(x_vals.min(), x_vals.max(), 100)
                ax.plot(xfit, p(xfit), color=col["test"], linewidth=1.8,
                        linestyle="--", label=f"Trend (slope={z[0]:.2f})")
                corr = np.corrcoef(x_vals, y_vals)[0, 1]
                ax.set_title(f"{x_label}\nvs {y_label}\nr = {corr:.3f}",
                             fontsize=8.5, fontweight="bold")
                ax.legend(fontsize=7)

            ax.set_xlabel(x_label, fontsize=8)
            ax.set_ylabel(y_label, fontsize=8)

        # Hide unused subplot
        if len(plots) < 6:
            axes_flat[-1].set_visible(False)

        plt.tight_layout()
        fname = f"4_scatter_{country.lower()}.png"
        path  = os.path.join(GRAPHS_DIR, fname)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
# GRAPH 5 — R² comparison bar chart (Model A vs B)
# ══════════════════════════════════════════════════════════════
def graph5_r2_comparison(res_dict):
    for country, df_res in res_dict.items():
        col = COUNTRY_COLORS[country]

        indicators = df_res["indicator"].tolist()
        n = len(indicators)
        x = np.arange(n)
        width = 0.2

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
        fig.suptitle(f"R² Comparison — Model A vs Model B | {country}",
                     fontsize=13, fontweight="bold")

        for ax, split, title in [
            (axes[0], "train", "Training Set (80%)"),
            (axes[1], "test",  "Test Set (20%)"),
        ]:
            a_vals = pd.to_numeric(df_res[f"A_r2_{split}"], errors="coerce").fillna(0).values
            b_vals = pd.to_numeric(df_res[f"B_r2_{split}"], errors="coerce").fillna(0).values

            bars_a = ax.bar(x - width/2, a_vals, width, label="Model A (5 scores)",
                            color=col["main"], alpha=0.85, edgecolor="white")
            bars_b = ax.bar(x + width/2, b_vals, width, label="Model B (sentiment)",
                            color=col["test"], alpha=0.85, edgecolor="white")

            # Value labels on bars
            for bar in list(bars_a) + list(bars_b):
                h = bar.get_height()
                if not np.isnan(h):
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                            f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)

            ax.axhline(0, color="black", linewidth=0.8)
            ax.axhline(0.5, color="gray", linewidth=0.7, linestyle=":", alpha=0.7,
                       label="R²=0.5 threshold")
            ax.set_xticks(x)
            ax.set_xticklabels([i.replace(" (YoY %)", "").replace(" (%)", "")
                                 .replace(" (YoY chg %)", "") for i in indicators],
                               rotation=20, ha="right", fontsize=9)
            ax.set_ylabel("R²", fontsize=10)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.legend(fontsize=8)
            # Shade negative region
            ymin = min(a_vals.min(), b_vals.min(), 0) - 0.05
            ax.set_ylim(ymin, max(a_vals.max(), b_vals.max(), 0.6) + 0.12)
            if ymin < 0:
                ax.axhspan(ymin, 0, alpha=0.05, color="red")

        plt.tight_layout()
        fname = f"5_r2_comparison_{country.lower()}.png"
        path  = os.path.join(GRAPHS_DIR, fname)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("Loading data...")
    dfs, res = load_data()

    if not dfs:
        print("[ERROR] No regression data found. Run step2 and step3 first.")
        return

    print(f"\nGenerating Graph 1 — Sentiment Index over time...")
    graph1_sentiment(dfs)

    print(f"\nGenerating Graph 2 — LLM scores vs real indicators...")
    graph2_scores_vs_real(dfs)

    print(f"\nGenerating Graph 3 — Predicted vs Actual...")
    graph3_predicted_vs_actual(dfs)

    print(f"\nGenerating Graph 4 — Scatter plots...")
    graph4_scatter(dfs)

    if res:
        print(f"\nGenerating Graph 5 — R² comparison...")
        graph5_r2_comparison(res)

    print(f"\nAll graphs saved to: {GRAPHS_DIR}/")
    for f in sorted(os.listdir(GRAPHS_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()