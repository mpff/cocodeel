"""Structured report: same-sample vs. split posthoc recipe.

Reads `results/simulation_images/smoke_sample_splitting.csv` and writes
`research/smoke-sample-splitting-report.md` with:

  - Per-setting table: mean ± std of MSPE(f̂_X) across 10 sims, per method.
  - Paired delta: split vs. same-sample on identical draws.
  - Variance check (H2): std of f̂_X across sims, split vs. same-sample.
  - High-cv1 check (H3): is the gap smaller at cv1=0.8 than at cv1=0.0?

Usage: python experiments/simulation_images/3b-smoke_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CSV  = ROOT / "results/simulation_images/smoke_sample_splitting.csv"
OUT  = ROOT / "research/smoke-sample-splitting-report.md"


def fmt(x, d=4):
    if pd.isna(x):
        return "n/a"
    return f"{x:.{d}f}"


def main():
    if not CSV.exists():
        print(f"Expected {CSV} — run 3-smoke_test_sample_splitting.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CSV)
    settings = df["setting"].unique().tolist()
    methods = ["base_full", "base_half", "posthoc", "posthoc_same_sample"]

    lines = [
        "# Smoke-test report — sample-splitting vs. same-sample posthoc",
        "",
        f"Source: `{CSV.relative_to(ROOT)}`  ({len(df)} rows, "
        f"{df['sim_id'].nunique()} sims per setting, {len(settings)} settings).",
        "",
        "## Methods",
        "- `base_full`: backbone on full N observations (paper's reference).",
        "- `base_half`: backbone on first N/2 observations (new-recipe backbone).",
        "- `posthoc`: PostHocCovarNetwork(base_half).fit(half_B) — **new, split recipe**.",
        "- `posthoc_same_sample`: PostHocCovarNetwork(base_full).fit(full) — **old, same-sample recipe**.",
        "",
        "## Per-setting MSPE(f̂_X): mean ± std across sims",
        "",
        "| setting | " + " | ".join(methods) + " |",
        "|---" + "|---" * len(methods) + "|",
    ]
    for s in settings:
        sub = df[df["setting"] == s]
        cells = []
        for m in methods:
            x = sub[sub["method"] == m]["mspe_fx"]
            cells.append(f"{fmt(x.mean())} ± {fmt(x.std())}")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Paired contrast: split vs. same-sample (same sim_id)",
        "",
        "Reports mean of (MSPE_split − MSPE_same) across sims, and the fraction",
        "of sims where split achieves lower MSPE than same-sample on the same draw.",
        "Negative mean = split is better on average.",
        "",
        "| setting | ΔMSPE(split − same) | P(split < same) |",
        "|---|---|---|",
    ]
    for s in settings:
        sub = df[df["setting"] == s]
        # pivot to (sim_id, method)
        piv = sub.pivot_table(index="sim_id", columns="method", values="mspe_fx")
        if "posthoc" in piv.columns and "posthoc_same_sample" in piv.columns:
            delta = piv["posthoc"] - piv["posthoc_same_sample"]
            frac = (delta < 0).mean()
            lines.append(f"| {s} | {fmt(delta.mean())} ± {fmt(delta.std())} | {fmt(frac, 2)} |")

    lines += [
        "",
        "## Variance (H2): std of f̂_X prediction means across sims",
        "",
        "Higher variance across sims = less stable estimate. If the split recipe",
        "restores exogeneity, we expect lower between-sim variance in `posthoc`",
        "than in `posthoc_same_sample`.",
        "",
        "| setting | std(mean_fx) posthoc | std(mean_fx) same_sample |",
        "|---|---|---|",
    ]
    for s in settings:
        sub = df[df["setting"] == s]
        v_split = sub[sub["method"] == "posthoc"]["mean_fx_hat"].std()
        v_same  = sub[sub["method"] == "posthoc_same_sample"]["mean_fx_hat"].std()
        lines.append(f"| {s} | {fmt(v_split)} | {fmt(v_same)} |")

    lines += [
        "",
        "## Hypothesis verdicts",
        "",
        "Populate these by inspection once the table above is available.",
        "",
        "- **H1** (splitting leaves MSPE(f̂_X) qualitatively unchanged at paper's q=32 CNN): TBD.",
        "- **H2** (splitting reduces between-sim variance of f̂_X): TBD.",
        "- **H3** (at high cv1 the split does not rescue f̂_X — concurvity wall): compare `fig4b_cv0`, `fig4b_cv4`, `fig4b_cv8`.",
        "",
        "## Context",
        "",
        "See `research/session-D-synthesis.md` for the rationale behind these",
        "hypotheses and the UKBB evidence that motivated the split.",
        "",
        "**Bernoulli / Fig 3 concurvity setting was deferred** — at N=400 the IRLS",
        "loop does not converge stably; a separate run at paper-scale N is required.",
        "",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
