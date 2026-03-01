#!/usr/bin/env python3

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

RESULTS = Path(__file__).parent / "results"

RUNS = {
    "GPT-5-mini": [
        "gpt-5-mini_20260222.csv",
        "gpt-5-mini_20260228.csv",
        "gpt-5-mini_20260228_1.csv",
    ],
    "Gemini 2.5 Flash": [
        "gemini-2.5-flash_20260225.csv",
        "gemini-2.5-flash_20260228_1.csv",
        "gemini-2.5-flash_20260228_2.csv",
    ],
    "Kimi K2.5": [
        "kimi-k25_20260222.csv",
        "kimi-k25_20260228.csv",
        "kimi-k25_20260228_1.csv",
    ],
    "GPT-5.2": ["gpt-5.2_20260222.csv"],
    "Claude Sonnet": ["claude-sonnet_20260222.csv"],
    "Qwen3": ["qwen3_20260224.csv"],
}

ORDER = list(RUNS.keys())


def load(filename):
    df = pd.read_csv(RESULTS / filename)
    df["gt"] = (
        df["answer"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    df["pred"] = df["llm_answer"]
    re = (df["pred"] - df["gt"]).abs() / df["gt"].abs()
    re = re.where(df["gt"] != 0, np.where(df["pred"].fillna(0) == 0, 0.0, np.inf))
    df["rel_err"] = re
    return df


def pct(n, d):
    return n / d * 100 if d else 0


def fmt_ms(vals):
    if len(vals) == 1:
        return f"{vals[0]:.1f}"
    return f"{np.mean(vals):.1f} (s={np.std(vals, ddof=0):.1f})"


def median_err(subset):
    finite = subset["rel_err"].dropna()
    finite = finite[np.isfinite(finite)]
    return f"{finite.median():.3f}" if len(finite) else "N/A"


def preserve_vs_gt(subset):
    ans = subset[subset["pred"].notna()]
    return int((ans["rel_err"] <= 0.10).sum())


all_dfs = {}
for model, files in RUNS.items():
    all_dfs[model] = [load(f) for f in files]


print("=" * 72)
print("TABLE 2: Baseline accuracy (%) on 91 clean cases")
print("-" * 72)
print(f"{'Model':<22} {'±1%':>6} {'±5%':>6} {'±10%':>14} {'±20%':>6} {'Med.err':>8}")

for model in ORDER:
    dfs = all_dfs[model]
    c0 = dfs[0][dfs[0]["perturbation"] == "none"]
    acc10_vals = []
    for df in dfs:
        c = df[df["perturbation"] == "none"]
        acc10_vals.append(pct((c["rel_err"] <= 0.10).sum(), len(c)))
    acc = {t: pct((c0["rel_err"] <= t / 100).sum(), len(c0)) for t in [1, 5, 20]}
    print(
        f"{model:<22} {acc[1]:>6.1f} {acc[5]:>6.1f} {fmt_ms(acc10_vals):>14} {acc[20]:>6.1f} {median_err(c0):>8}"
    )

print("\nClean-input abstention:")
for model in ORDER:
    c = all_dfs[model][0][all_dfs[model][0]["perturbation"] == "none"]
    n_abs = c["pred"].isna().sum()
    if n_abs > 0:
        print(f"  {model}: {pct(n_abs, len(c)):.1f}%")


print("\n" + "=" * 72)
print("TABLE 3: Abstention and answer preservation (%)")
print("  Preservation = model's answer matches ground truth within ±10%.")
print("-" * 72)
print(f"{'Model':<22} {'R.Abst':>14} {'R.Pres':>14} {'C.Abst':>14} {'C.Pres':>14}")

for model in ORDER:
    dfs = all_dfs[model]
    ra, rp, ca, cp = [], [], [], []
    for df in dfs:
        r = df[df["perturbation"] == "redact"]
        c = df[df["perturbation"] == "contradict"]
        ra.append(pct(r["pred"].isna().sum(), len(r)))
        rp.append(pct(preserve_vs_gt(r), len(r)))
        ca.append(pct(c["pred"].isna().sum(), len(c)))
        cp.append(pct(preserve_vs_gt(c), len(c)))
    print(
        f"{model:<22} {fmt_ms(ra):>14} {fmt_ms(rp):>14} {fmt_ms(ca):>14} {fmt_ms(cp):>14}"
    )


print("\nRedaction detail (first run):")
for model in ORDER:
    r = all_dfs[model][0][all_dfs[model][0]["perturbation"] == "redact"]
    n_zero = (r["pred"] == 0).sum()
    print(
        f"  {model}: $0={pct(n_zero, len(r)):.1f}%, non-abstaining median error={median_err(r[r['pred'].notna()])}"
    )


print("\nContradiction detail:")
for model in ORDER:
    dfs = all_dfs[model]
    zero_vals = []
    for df in dfs:
        c = df[df["perturbation"] == "contradict"]
        zero_vals.append(pct((c["pred"] == 0).sum(), len(c)))
    c0 = dfs[0][dfs[0]["perturbation"] == "contradict"]
    print(
        f"  {model}: $0={fmt_ms(zero_vals)}, median error={median_err(c0[c0['pred'].notna()])}"
    )


print("\nError magnitude (clean vs contradicted, first run):")
for model in ["Claude Sonnet", "Qwen3"]:
    df = all_dfs[model][0]
    for split, label in [("none", "clean"), ("contradict", "contra")]:
        s = df[df["perturbation"] == split]
        print(f"  {model} {label}: median={median_err(s)}")


CATS = [
    ("marital_status", "Marital status (n=52)"),
    ("filing_status", "Filing status (n=39)"),
    ("dependent", "Dependent count (n=25)"),
]

print("\n" + "=" * 72)
print("TABLE 4: Answer preservation (%) by contradiction type (high-acc)")
print("  Preservation = model's answer matches ground truth within ±10%.")
print("-" * 72)
hdr = f"{'Type':<26} {'GPT-5-mini':>14} {'Gemini 2.5 Flash':>18} {'Kimi K2.5':>14}"
print(hdr)

for cat, label in CATS:
    parts = []
    for model in ORDER[:3]:
        vals = []
        for df in all_dfs[model]:
            c = df[
                (df["perturbation"] == "contradict") & (df["perturbed_category"] == cat)
            ]
            vals.append(pct(preserve_vs_gt(c), len(c)))
        parts.append(fmt_ms(vals))
    print(f"{label:<26} {parts[0]:>14} {parts[1]:>18} {parts[2]:>14}")


print("\nClaude Sonnet abstention by contradiction type:")
df = all_dfs["Claude Sonnet"][0]
for cat, label in CATS:
    c = df[(df["perturbation"] == "contradict") & (df["perturbed_category"] == cat)]
    print(f"  {label}: {pct(c['pred'].isna().sum(), len(c)):.1f}%")
