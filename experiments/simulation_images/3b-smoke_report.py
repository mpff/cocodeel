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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=str(CSV))
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()
    csv_path = Path(args.csv).resolve()
    out_path = Path(args.out).resolve()

    if not csv_path.exists():
        print(f"Expected {csv_path} — run 3-smoke_test_sample_splitting.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    methods = ["base_full", "base_half", "posthoc", "posthoc_same_sample"]
    # Group by (setting, n_total) so we can scan N-scaling within the same setting.
    df["setting_N"] = df["setting"] + "@N=" + df["n_total"].astype(str)
    settings = df["setting_N"].unique().tolist()

    lines = [
        "# Smoke-test report — sample-splitting vs. same-sample posthoc",
        "",
        f"Source: `{csv_path.relative_to(ROOT)}`  ({len(df)} rows, "
        f"{df['sim_id'].nunique()} sims per (setting, N), {len(settings)} (setting, N) pairs).",
        "",
        "## Methods",
        "- `base_full`: backbone on full N observations (paper's reference).",
        "- `base_half`: backbone on first N/2 observations (split-recipe backbone).",
        "- `posthoc`: PostHocCovarNetwork(base_half).fit(half_B) — **mathematically correct (split) recipe**.",
        "- `posthoc_same_sample`: PostHocCovarNetwork(base_full).fit(full) — **biased same-sample recipe** (kept only as a reference).",
        "",
        "Note on framing: `posthoc_same_sample` is a biased estimator (Pagan 1984",
        "generated regressors — the refit reuses the backbone's training sample,",
        "so H = phi(X; theta*) is endogenous w.r.t. y). At finite N its MSPE to",
        "ground truth can happen to be lower than the split's, not because it's",
        "closer to the truth in expectation but because both estimators are",
        "noisy and the biased one may variance-regularise toward its own sample",
        "noise. As N → infinity, only the split recipe is consistent.",
        "",
        "## Per-setting MSPE(f̂_X): mean ± std across sims",
        "",
        "| setting | " + " | ".join(methods) + " |",
        "|---" + "|---" * len(methods) + "|",
    ]
    for s in settings:
        sub = df[df["setting_N"] == s]
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
        sub = df[df["setting_N"] == s]
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
        sub = df[df["setting_N"] == s]
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    try:
        print(f"Wrote {out_path.relative_to(ROOT)}")
    except ValueError:
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
