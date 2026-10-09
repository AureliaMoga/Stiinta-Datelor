"""
step3_regression.py
===================
Runs two sets of regressions for each country and each economic indicator:

  Model A — Multiple regression:
      real_indicator(t) ~ all 5 LLM indicator scores(t-1)

  Model B — Simple regression:
      real_indicator(t) ~ sentiment_index(t-1)

Train/test split: first 80% of observations = train, last 20% = test.
Reports R² for both train and test sets.

Inputs:
    outputs/regression_data_md.csv
    outputs/regression_data_ro.csv

Outputs:
    outputs/regression_results_md.csv
    outputs/regression_results_ro.csv
    outputs/plots/  (scatter + fit plots)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

warnings.filterwarnings("ignore")

OUT_DIR   = "./outputs"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

INDICATORS = {
    "gdp_yoy_pct":               "GDP Growth (YoY %)",
    "inflation_yoy_pct":         "Inflation (YoY %)",
    "unemployment_pct":          "Unemployment (%)",
    "interest_rate_pct":         "Interest Rate (%)",
    "exchange_rate_yoy_chg_pct": "Exchange Rate (YoY chg %)",
}

LLM_IND_COLS = [
    "economic_growth_score_lag1",
    "inflation_score_lag1",
    "unemployment_score_lag1",
    "interest_rates_score_lag1",
    "exchange_rate_score_lag1",
]
SENT_COL = "sentiment_index_lag1"


def train_test_split_ts(df, train_frac=0.8):
    # chronological 80/20 split
    n     = len(df)
    split = int(np.ceil(n * train_frac))
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def fit_ols(X_train, y_train, X_test, y_test, feature_names):

    # statsmodels for coefficients / p-values
    X_tr_sm = sm.add_constant(X_train, has_constant="add")
    X_te_sm = sm.add_constant(X_test,  has_constant="add")
    try:
        ols = sm.OLS(y_train, X_tr_sm).fit()
        coef_dict = dict(zip(["const"] + feature_names, ols.params))
        pval_dict = dict(zip(["const"] + feature_names, ols.pvalues))
    except Exception:
        coef_dict = {}
        pval_dict = {}

    # sklearn for R²
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    r2_train = r2_score(y_train, reg.predict(X_train))
    if len(X_test) >= 2:
        r2_test = r2_score(y_test, reg.predict(X_test))
    else:
        r2_test = np.nan

    return {
        "model":     reg,
        "r2_train":  round(r2_train, 4),
        "r2_test":   round(r2_test, 4) if not np.isnan(r2_test) else np.nan,
        "coefs":     coef_dict,
        "pvalues":   pval_dict,
        "n_train":   len(X_train),
        "n_test":    len(X_test),
    }


def run_regressions(df, country):
    results = []
    train, test = train_test_split_ts(df)
    print(f"\n  {country} — Train: {len(train)} | Test: {len(test)} quarters")

    for real_col, label in INDICATORS.items():
        # Drop rows where real col is NaN
        tr = train.dropna(subset=[real_col])
        te = test.dropna(subset=[real_col])

        if len(tr) < 5:
            print(f"    [SKIP] {label}: not enough train data ({len(tr)} rows)")
            continue

        y_train = tr[real_col].values
        y_test  = te[real_col].values if len(te) > 0 else np.array([])

        #  Model A: 5 indicator scores
        X_tr_a = tr[LLM_IND_COLS].values
        X_te_a = te[LLM_IND_COLS].values if len(te) > 0 else np.zeros((0, len(LLM_IND_COLS)))
        res_a  = fit_ols(X_tr_a, y_train, X_te_a, y_test, LLM_IND_COLS)

        #  Model B: sentiment index only
        X_tr_b = tr[[SENT_COL]].values
        X_te_b = te[[SENT_COL]].values if len(te) > 0 else np.zeros((0, 1))
        res_b  = fit_ols(X_tr_b, y_train, X_te_b, y_test, [SENT_COL])

        print(f"    {label:30s} | A: R²_train={res_a['r2_train']:.3f}  R²_test={res_a['r2_test']}"
              f"  | B: R²_train={res_b['r2_train']:.3f}  R²_test={res_b['r2_test']}")

        row = {
            "country":            country,
            "indicator":          label,
            "real_col":           real_col,
            "n_train":            res_a["n_train"],
            "n_test":             res_a["n_test"],
            # Model A
            "A_r2_train":         res_a["r2_train"],
            "A_r2_test":          res_a["r2_test"],
            # Model B
            "B_r2_train":         res_b["r2_train"],
            "B_r2_test":          res_b["r2_test"],
            "B_coef_sentiment":   round(res_b["coefs"].get(SENT_COL, np.nan), 4),
            "B_intercept":        round(res_b["coefs"].get("const",     np.nan), 4),
            "B_pval_sentiment":   round(res_b["pvalues"].get(SENT_COL,  np.nan), 4),
        }
        # Add Model A coefficients
        for col in LLM_IND_COLS:
            short = col.replace("_score_lag1", "").replace("_", " ")
            row[f"A_coef_{col}"] = round(res_a["coefs"].get(col, np.nan), 4)
            row[f"A_pval_{col}"] = round(res_a["pvalues"].get(col, np.nan), 4)

        results.append(row)

        #  Plot: predicted vs actual
        _plot_predictions(tr, te, real_col, label, res_a, res_b,
                          LLM_IND_COLS, SENT_COL, country)

    return results


def _plot_predictions(tr, te, real_col, label, res_a, res_b,
                      feat_a, feat_b, country):

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(f"{country} — {label}", fontsize=12, fontweight="bold")

    quarters_tr = tr["quarter"].values
    quarters_te = te["quarter"].values if len(te) > 0 else []

    for ax, model_key, feat_cols, title in [
        (axes[0], res_a, feat_a, "Model A (5 Indicator Scores)"),
        (axes[1], res_b, [feat_b] if isinstance(feat_b, str) else feat_b,
         "Model B (Sentiment Index)"),
    ]:
        reg = model_key["model"]
        tr_clean = tr.dropna(subset=[real_col])
        te_clean = te.dropna(subset=[real_col]) if len(te) > 0 else te

        X_tr = tr_clean[feat_cols].values
        X_te = te_clean[feat_cols].values if len(te_clean) > 0 else np.zeros((0, len(feat_cols)))
        y_tr = tr_clean[real_col].values
        y_te = te_clean[real_col].values if len(te_clean) > 0 else np.array([])

        pred_tr = reg.predict(X_tr)
        pred_te = reg.predict(X_te) if len(X_te) > 0 else np.array([])

        x_tr = range(len(y_tr))
        x_te = range(len(y_tr), len(y_tr) + len(y_te))

        ax.plot(x_tr, y_tr,   "o-", color="#2c7bb6", label="Actual (train)",    markersize=4)
        ax.plot(x_tr, pred_tr,"s--",color="#abd9e9", label="Predicted (train)",  markersize=4)
        if len(y_te) > 0:
            ax.plot(x_te, y_te,   "o-", color="#d7191c", label="Actual (test)",     markersize=4)
            ax.plot(x_te, pred_te,"s--",color="#fdae61", label="Predicted (test)",   markersize=4)
            ax.axvline(x=len(y_tr) - 0.5, color="gray", linestyle=":", linewidth=1.2)

        r2_tr = model_key["r2_train"]
        r2_te = model_key["r2_test"]
        ax.set_title(f"{title}\nR²_train={r2_tr:.3f}  R²_test={r2_te}", fontsize=10)
        ax.set_xlabel("Observation index")
        ax.set_ylabel(label)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe = real_col.replace("_", "-")
    fname = os.path.join(PLOTS_DIR, f"{country.lower()}_{safe}.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    all_results = []

    for tag, country in [("md", "Moldova"), ("ro", "Romania")]:
        data_path = os.path.join(OUT_DIR, f"regression_data_{tag}.csv")
        if not os.path.exists(data_path):
            print(f"[ERROR] Missing {data_path} — run step2_prepare_data.py first.")
            continue

        df = pd.read_csv(data_path)
        print(f"\n{'='*60}\n  {country}: {len(df)} regression-ready quarters\n{'='*60}")

        results = run_regressions(df, country)
        all_results.extend(results)

        # Save per-country results CSV
        df_res = pd.DataFrame(results)
        out_csv = os.path.join(OUT_DIR, f"regression_results_{tag}.csv")
        df_res.to_csv(out_csv, index=False)
        print(f"\n  Saved results → {out_csv}")

    # Save combined
    if all_results:
        pd.DataFrame(all_results).to_csv(
            os.path.join(OUT_DIR, "regression_results_combined.csv"), index=False)
        print(f"\nCombined results saved.")


if __name__ == "__main__":
    main()