"""Aggregate NPZ predictions from `run_full_simulation.py` into R-ready CSVs.

For each block, reads every `<run>/<block>/preds/<sweep_key>/seed=*.npz`
and emits `results/simulation_images/<block>.csv` with long-form columns:

    model, effect, metric, value, n, <sweep_var>

where:
  - model ∈ {base, posthoc, posthoc_orth, posthoc_web, posthoc_lam0,
             posthoc_orth_lam0}  (base_full is renamed to `base`;
             base_half is dropped — it's a split-recipe internal.)
  - effect ∈ {y, fx, fr, fz}
  - metric ∈ {mspe, bias2, var}
  - value  = mean_i of the per-point metric across S seeds.

Per-point decomposition (test point i, method m, effect e):
    bias_i^2 = (mean_s ŷ_{i,s} - truth_i)^2
    var_i    = var_s ŷ_{i,s}                  (population var, 1/S)
    mspe_i   = mean_s (ŷ_{i,s} - truth_i)^2 = bias_i^2 + var_i

Then report `mean_i(...)` — standard bias/variance decomposition.

Usage:
    python experiments/simulation_images/aggregate_full_simulation.py \\
        --run-dir results/simulation_images/runs/<stamp>_full_nsim50
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np


# Map runner method names → R-expected names.
METHOD_RENAME = {"base_full": "base"}
METHOD_DROP = {"base_half"}

# Parse a sweep_key like "n=12800_bz=1.5" or "n=3200_cv1=0.9" or "n=400_p=4"
# or "n=400_q=64" or "n=1600". Returns dict of sweep vars.
_SWEEP_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9_]*)=([\-0-9.]+)")


def parse_sweep_key(sweep_key: str) -> dict:
    out = {}
    for m in _SWEEP_RE.finditer(sweep_key):
        k, v = m.group(1), m.group(2)
        out[k] = float(v) if "." in v else int(v)
    return out


def aggregate_block(block_dir: Path) -> list[dict]:
    """Aggregate all NPZ files in one block → long-form rows."""
    preds_root = block_dir / "preds"
    if not preds_root.exists():
        return []

    rows = []
    for sweep_dir in sorted(preds_root.iterdir()):
        if not sweep_dir.is_dir():
            continue
        npz_paths = sorted(sweep_dir.glob("seed=*.npz"))
        if not npz_paths:
            continue

        # Load all seeds; build per-method per-effect stacked arrays.
        methods = None
        stacks = {}        # (method, effect) → np.ndarray [S, T]
        truths = {}        # effect → np.ndarray [T]
        for p in npz_paths:
            d = np.load(p, allow_pickle=False)
            if methods is None:
                methods = [str(m) for m in d["methods"]]
                for eff in ("y", "fx", "fr", "fz"):
                    truths[eff] = d[f"truth__{eff}"]
            for m in methods:
                for eff in ("y", "fx", "fr", "fz"):
                    stacks.setdefault((m, eff), []).append(d[f"{m}__{eff}"])
        stacks = {k: np.stack(v, axis=0) for k, v in stacks.items()}  # [S, T]

        sweep_vars = parse_sweep_key(sweep_dir.name)

        for (m, eff), preds in stacks.items():
            if m in METHOD_DROP:
                continue
            mm = METHOD_RENAME.get(m, m)
            truth = truths[eff]                              # [T]
            mean_pred = preds.mean(axis=0)                   # [T]
            bias2 = (mean_pred - truth) ** 2                 # [T]
            var = preds.var(axis=0, ddof=0)                  # [T]  population var
            mspe = ((preds - truth[None, :]) ** 2).mean(0)   # [T]
            # Sanity: mspe ≈ bias2 + var (up to float32 roundoff).
            base_row = dict(model=mm, effect=eff, **sweep_vars)
            rows.append(dict(**base_row, metric="mspe", value=float(mspe.mean())))
            rows.append(dict(**base_row, metric="bias2", value=float(bias2.mean())))
            rows.append(dict(**base_row, metric="var", value=float(var.mean())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=str)
    ap.add_argument("--out-dir", type=str,
                    default="results/simulation_images")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    block_dirs = [d for d in run_dir.iterdir()
                  if d.is_dir() and (d / "preds").exists()]

    for bd in sorted(block_dirs):
        block = bd.name
        rows = aggregate_block(bd)
        if not rows:
            print(f"[{block}] no rows", flush=True)
            continue

        # Union column order: fixed fields first, then any sweep vars seen.
        fixed = ["model", "effect", "metric", "value"]
        sweep_keys = sorted({k for r in rows for k in r
                             if k not in fixed})
        cols = fixed + sweep_keys

        outp = out_dir / f"{block}.csv"
        with outp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"[{block}] {len(rows)} rows → {outp}", flush=True)


if __name__ == "__main__":
    main()
