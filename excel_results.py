"""
step4_excel_results.py
======================
Builds the part3_results Excel files as required by the assignment.
One file per country + one combined file.

Inputs:
    outputs/regression_results_md.csv
    outputs/regression_results_ro.csv
    outputs/regression_data_md.csv
    outputs/regression_data_ro.csv

Outputs:
    outputs/part3_results_md_<SOURCE>.xlsx
    outputs/part3_results_ro_<SOURCE>.xlsx
"""

import os
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import (Font, PatternFill, Alignment, Border, Side,
                              GradientFill)
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule

# ── CONFIG ────────────────────────────────────────────────────
SOURCE_MD = "zdg"
SOURCE_RO = "hotnews"
OUT_DIR   = "./outputs"
# ─────────────────────────────────────────────────────────────

# Styles helpers
HDR_FILL  = PatternFill("solid", start_color="1F3864")
SUB_FILL  = PatternFill("solid", start_color="D6E4F0")
ALT_FILL  = PatternFill("solid", start_color="EBF3FB")
HDR_FONT  = Font(bold=True, color="FFFFFF", name="Calibri", size=10)
BOLD_FONT = Font(bold=True, name="Calibri", size=10)
NORM_FONT = Font(name="Calibri", size=10)
CENT      = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT      = Alignment(horizontal="left",   vertical="center", wrap_text=True)
THIN      = Side(style="thin", color="B0B0B0")
BDR       = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

POS_FILL  = PatternFill("solid", start_color="C6EFCE")  # green
NEG_FILL  = PatternFill("solid", start_color="FFC7CE")  # red
NEU_FILL  = PatternFill("solid", start_color="FFEB9C")  # yellow


def _hdr(ws, row, col, value, width=None):
    c = ws.cell(row=row, column=col, value=value)
    c.font   = HDR_FONT
    c.fill   = HDR_FILL
    c.alignment = CENT
    c.border = BDR
    if width:
        ws.column_dimensions[get_column_letter(col)].width = width
    return c


def _cell(ws, row, col, value, bold=False, fill=None, align=CENT, fmt=None):
    c = ws.cell(row=row, column=col, value=value)
    c.font      = BOLD_FONT if bold else NORM_FONT
    c.alignment = align
    c.border    = BDR
    if fill:
        c.fill = fill
    if fmt:
        c.number_format = fmt
    return c


def r2_fill(val):
    """Color-code R² cell."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if val >= 0.5:
        return POS_FILL
    elif val >= 0.1:
        return NEU_FILL
    else:
        return NEG_FILL


def build_excel(tag, source, country_name):
    res_path  = os.path.join(OUT_DIR, f"regression_results_{tag}.csv")
    data_path = os.path.join(OUT_DIR, f"regression_data_{tag}.csv")

    if not os.path.exists(res_path):
        print(f"  [ERROR] Missing {res_path}")
        return

    df_res  = pd.read_csv(res_path)
    df_data = pd.read_csv(data_path) if os.path.exists(data_path) else pd.DataFrame()

    wb = openpyxl.Workbook()

    # ══════════════════════════════════════════════════════════
    # Sheet 1 — R² Summary
    # ══════════════════════════════════════════════════════════
    ws1 = wb.active
    ws1.title = "R² Summary"

    # Title
    ws1.merge_cells("A1:L1")
    t = ws1["A1"]
    t.value     = f"Part 3 Results — {country_name} | Source: {source.upper()}"
    t.font      = Font(bold=True, color="FFFFFF", name="Calibri", size=13)
    t.fill      = HDR_FILL
    t.alignment = CENT
    ws1.row_dimensions[1].height = 28

    # Sub-header explanation
    ws1.merge_cells("A2:L2")
    s = ws1["A2"]
    s.value     = ("Model A: Multiple regression — 5 lagged LLM indicator scores → real indicator  |  "
                   "Model B: Simple regression — lagged sentiment index → real indicator  |  "
                   "Train: first 80% | Test: last 20%")
    s.font      = Font(italic=True, name="Calibri", size=9, color="444444")
    s.alignment = LEFT
    ws1.row_dimensions[2].height = 18

    # Column headers row 3
    headers = [
        ("Indicator", 24),
        ("N Train", 9), ("N Test", 9),
        ("Model A\nR² Train", 13), ("Model A\nR² Test", 13),
        ("Model B\nR² Train", 13), ("Model B\nR² Test", 13),
        ("Model B\nSentiment Coef", 16), ("Model B\nIntercept", 13),
        ("Model B\nP-value (Sent.)", 15),
        ("Better Model\n(Train)", 13), ("Better Model\n(Test)", 13),
    ]
    for col, (h, w) in enumerate(headers, 1):
        _hdr(ws1, 3, col, h, width=w)
    ws1.row_dimensions[3].height = 32

    # Data rows
    for i, row in df_res.iterrows():
        r = i + 4
        fill = ALT_FILL if i % 2 == 0 else None

        a_r2_tr = row.get("A_r2_train", np.nan)
        a_r2_te = row.get("A_r2_test",  np.nan)
        b_r2_tr = row.get("B_r2_train", np.nan)
        b_r2_te = row.get("B_r2_test",  np.nan)

        # Better model comparison
        better_train = "A" if (not np.isnan(a_r2_tr) and not np.isnan(b_r2_tr)
                                and a_r2_tr >= b_r2_tr) else "B"
        better_test  = "A" if (not np.isnan(a_r2_te) and not np.isnan(b_r2_te)
                                and a_r2_te >= b_r2_te) else "B"

        vals = [
            row.get("indicator", ""),
            int(row.get("n_train", 0)), int(row.get("n_test", 0)),
            a_r2_tr, a_r2_te,
            b_r2_tr, b_r2_te,
            row.get("B_coef_sentiment", np.nan),
            row.get("B_intercept",      np.nan),
            row.get("B_pval_sentiment", np.nan),
            better_train, better_test,
        ]
        for col, v in enumerate(vals, 1):
            val = None if (isinstance(v, float) and np.isnan(v)) else v
            c = _cell(ws1, r, col, val, fill=fill)
            # R² color coding (cols 4-7)
            if col in (4, 5, 6, 7) and val is not None:
                c.fill = r2_fill(val)
                c.number_format = "0.000"
            # Coef / pval formatting
            if col in (8, 9, 10) and val is not None:
                c.number_format = "0.0000"
            # Better model highlight
            if col in (11, 12) and val == "A":
                c.fill = POS_FILL
            elif col in (11, 12) and val == "B":
                c.fill = PatternFill("solid", start_color="DEEBF7")

        ws1.row_dimensions[r].height = 18

    # Legend
    legend_row = len(df_res) + 5
    ws1.merge_cells(f"A{legend_row}:D{legend_row}")
    ws1[f"A{legend_row}"].value = "Legend:"
    ws1[f"A{legend_row}"].font = BOLD_FONT
    colors = [("R² ≥ 0.50 (Good)", POS_FILL),
              ("R² 0.10–0.49 (Moderate)", NEU_FILL),
              ("R² < 0.10 / Negative (Poor)", NEG_FILL)]
    for j, (txt, fill) in enumerate(colors):
        c = ws1.cell(row=legend_row + 1 + j, column=1, value=txt)
        c.fill = fill
        c.font = NORM_FONT
        c.border = BDR

    ws1.freeze_panes = "A4"

    # ══════════════════════════════════════════════════════════
    # Sheet 2 — Model A Coefficients
    # ══════════════════════════════════════════════════════════
    ws2 = wb.create_sheet("Model A Coefficients")

    ind_score_labels = ["Econ Growth Score", "Inflation Score",
                        "Unemployment Score", "Interest Rate Score", "Exchange Rate Score"]
    col_keys = [
        "A_coef_economic_growth_score_lag1",
        "A_coef_inflation_score_lag1",
        "A_coef_unemployment_score_lag1",
        "A_coef_interest_rates_score_lag1",
        "A_coef_exchange_rate_score_lag1",
    ]
    pval_keys = [k.replace("A_coef_", "A_pval_") for k in col_keys]

    hdr2 = ["Indicator"] + [f"{l}\n(Coef)" for l in ind_score_labels] + \
           [f"{l}\n(p-val)" for l in ind_score_labels]
    ws2.merge_cells("A1:K1")
    ws2["A1"].value     = f"Model A — Regression Coefficients & P-values | {country_name}"
    ws2["A1"].font      = Font(bold=True, color="FFFFFF", name="Calibri", size=12)
    ws2["A1"].fill      = HDR_FILL
    ws2["A1"].alignment = CENT
    ws2.row_dimensions[1].height = 24

    for col, h in enumerate(hdr2, 1):
        c = _hdr(ws2, 2, col, h)
        ws2.column_dimensions[get_column_letter(col)].width = 16 if col > 1 else 26
    ws2.row_dimensions[2].height = 32

    for i, row in df_res.iterrows():
        r = i + 3
        fill = ALT_FILL if i % 2 == 0 else None
        _cell(ws2, r, 1, row.get("indicator", ""), fill=fill, align=LEFT)
        for j, key in enumerate(col_keys, 2):
            v = row.get(key, np.nan)
            c = _cell(ws2, r, j, None if np.isnan(v) else round(v, 4), fill=fill, fmt="0.0000")
        for j, key in enumerate(pval_keys, 7):
            v = row.get(key, np.nan)
            val = None if np.isnan(v) else round(v, 4)
            c = _cell(ws2, r, j, val, fill=fill, fmt="0.0000")
            if val is not None and val < 0.05:
                c.font = Font(bold=True, color="006100", name="Calibri", size=10)
        ws2.row_dimensions[r].height = 18

    ws2.freeze_panes = "A3"

    # ══════════════════════════════════════════════════════════
    # Sheet 3 — Raw Regression Data
    # ══════════════════════════════════════════════════════════
    if not df_data.empty:
        ws3 = wb.create_sheet("Regression Data")
        cols_show = ["quarter", "gdp_yoy_pct", "inflation_yoy_pct",
                     "unemployment_pct", "interest_rate_pct",
                     "exchange_rate_yoy_chg_pct",
                     "sentiment_index", "economic_growth_score",
                     "inflation_score", "unemployment_score",
                     "interest_rates_score", "exchange_rate_score",
                     "sentiment_index_lag1", "economic_growth_score_lag1",
                     "inflation_score_lag1", "unemployment_score_lag1",
                     "interest_rates_score_lag1", "exchange_rate_score_lag1"]
        cols_show = [c for c in cols_show if c in df_data.columns]

        ws3.merge_cells(f"A1:{get_column_letter(len(cols_show))}1")
        ws3["A1"].value     = f"Regression Dataset — {country_name} (merged LLM + real data with lags)"
        ws3["A1"].font      = Font(bold=True, color="FFFFFF", name="Calibri", size=11)
        ws3["A1"].fill      = HDR_FILL
        ws3["A1"].alignment = CENT
        ws3.row_dimensions[1].height = 22

        friendly = {
            "quarter": "Quarter",
            "gdp_yoy_pct": "GDP YoY%",
            "inflation_yoy_pct": "Inflation YoY%",
            "unemployment_pct": "Unemployment%",
            "interest_rate_pct": "Interest Rate%",
            "exchange_rate_yoy_chg_pct": "Ex.Rate YoY%",
            "sentiment_index": "Sentiment",
            "economic_growth_score": "Econ Growth Score",
            "inflation_score": "Inflation Score",
            "unemployment_score": "Unemp Score",
            "interest_rates_score": "Int.Rate Score",
            "exchange_rate_score": "Ex.Rate Score",
            "sentiment_index_lag1": "Sentiment (t-1)",
            "economic_growth_score_lag1": "Econ Growth (t-1)",
            "inflation_score_lag1": "Inflation (t-1)",
            "unemployment_score_lag1": "Unemp (t-1)",
            "interest_rates_score_lag1": "Int.Rate (t-1)",
            "exchange_rate_score_lag1": "Ex.Rate (t-1)",
        }
        for col, key in enumerate(cols_show, 1):
            _hdr(ws3, 2, col, friendly.get(key, key), width=15)
        ws3.row_dimensions[2].height = 28

        for i, (_, row) in enumerate(df_data[cols_show].iterrows()):
            r = i + 3
            fill = ALT_FILL if i % 2 == 0 else None
            for col, key in enumerate(cols_show, 1):
                v = row[key]
                val = None if (isinstance(v, float) and np.isnan(v)) else v
                c = _cell(ws3, r, col, val, fill=fill)
                if key != "quarter" and val is not None:
                    c.number_format = "0.00"
            ws3.row_dimensions[r].height = 16

        ws3.freeze_panes = "A3"

    # ══════════════════════════════════════════════════════════
    # Sheet 4 — Notes & Sources
    # ══════════════════════════════════════════════════════════
    ws4 = wb.create_sheet("Notes & Sources")
    notes = [
        ("Country",          country_name),
        ("Source",           source.upper()),
        ("Time Period",      "2019-Q1 to 2026-Q1 (Q2 2026 partial)"),
        ("Frequency",        "Monthly LLM scores → averaged to quarterly"),
        ("Model A",          "Multiple OLS: 5 lagged LLM indicator scores (t-1) → real indicator (t)"),
        ("Model B",          "Simple OLS: lagged sentiment index (t-1) → real indicator (t)"),
        ("Lag",              "LLM scores lagged 1 quarter (previous quarter predicts current)"),
        ("Train/Test Split", "First 80% of quarters = train; last 20% = test (chronological)"),
        ("R² interpretation","R² measures share of variance explained. Negative R² in test = worse than mean."),
        ("Sentiment scale",  "-2 (very negative) to +2 (very positive)"),
        ("Indicator scale",  "-5 (sharp decline) to +5 (sharp increase)"),
        ("GDP data source",  "Moldova: BNS / World Bank  |  Romania: INS / Eurostat"),
        ("Inflation source", "Moldova: NBM Inflation Reports  |  Romania: INS / Eurostat HICP"),
        ("Unemployment",     "Moldova: BNS Labour Force Survey  |  Romania: INS LFS"),
        ("Interest Rate",    "Moldova: NBM base rate (end-of-quarter)  |  Romania: BNR policy rate"),
        ("Exchange Rate",    "Moldova: MDL/EUR YoY%  |  Romania: RON/EUR YoY% (+ = depreciation)"),
        ("2026-Q2 note",     "Q2 2026 GDP not yet released; rows with NaN GDP excluded from GDP regressions."),
        ("Bold p-values",    "Green bold = statistically significant at 5% level (p < 0.05)"),
    ]
    ws4.column_dimensions["A"].width = 22
    ws4.column_dimensions["B"].width = 80
    _hdr(ws4, 1, 1, "Key")
    _hdr(ws4, 1, 2, "Description")
    for i, (k, v) in enumerate(notes, 2):
        ws4.cell(row=i, column=1, value=k).font  = BOLD_FONT
        ws4.cell(row=i, column=1).border = BDR
        ws4.cell(row=i, column=1).alignment = LEFT
        c = ws4.cell(row=i, column=2, value=v)
        c.font = NORM_FONT
        c.border = BDR
        c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        ws4.row_dimensions[i].height = 20

    # ── Save ──
    out_path = os.path.join(OUT_DIR, f"part3_results_{tag}_{source}.xlsx")
    wb.save(out_path)
    print(f"  Saved → {out_path}")


def main():
    for tag, source, name in [
        ("md", SOURCE_MD, "Moldova"),
        ("ro", SOURCE_RO, "Romania"),
    ]:
        print(f"\nBuilding Excel for {name} ({source}) ...")
        build_excel(tag, source, name)


if __name__ == "__main__":
    main()