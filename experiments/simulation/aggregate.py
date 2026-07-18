"""Aggregate per-seed NPZ predictions from the study runs into R-ready CSVs.

For every sweep of every study run dir, reads preds/<sweep_key>/seed=*.npz and
writes output/<sweep>.csv in long form: model, effect, metric, value, plus the
sweep variables parsed from the sweep key. Per test point i the seed dimension
is reduced to bias_i^2 = (mean_s yhat_is - truth_i)^2, var_i = var_s(yhat_is),
mspe_i = mean_s (yhat_is - truth_i)^2 = bias_i^2 + var_i; reported values are
means over i.

Usage:  python experiments/simulation/aggregate.py
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RUN_DIRS = [
    ROOT / "experiments/simulation/output/runs/study_a",
    ROOT / "experiments/simulation/output/runs/study_b",
    ROOT / "experiments/simulation/output/runs/study_c",
]
OUT_DIR = ROOT / "experiments/simulation/output"

EFFECTS = ("y", "fx", "fr", "fz")

_SWEEP_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9_]*)=([\-0-9.]+)")


def parse_sweep_key(sweep_key):
    """'n=3200_bz=1.5' -> {'n': 3200, 'bz': 1.5}."""
    out = {}
    for m in _SWEEP_RE.finditer(sweep_key):
        k, v = m.group(1), m.group(2)
        out[k] = float(v) if "." in v else int(v)
    return out


def aggregate_sweep(sweep_dir):
    """Reduce all seeds of one sweep into long-form rows."""
    rows = []
    for setting_dir in sorted((sweep_dir / "preds").iterdir()):
        npz_paths = sorted(setting_dir.glob("seed=*.npz"))
        if not npz_paths:
            continue

        # stack seeds per (method, effect)
        methods = None
        stacks = {}
        truths = {}
        for p in npz_paths:
            d = np.load(p, allow_pickle=False)
            if methods is None:
                methods = [str(m) for m in d["methods"]]
                for eff in EFFECTS:
                    truths[eff] = d[f"truth__{eff}"]
            for m in methods:
                for eff in EFFECTS:
                    stacks.setdefault((m, eff), []).append(d[f"{m}__{eff}"])
        stacks = {k: np.stack(v, axis=0) for k, v in stacks.items()}  # [S, T]

        # bias/variance decomposition per method and effect
        sweep_vars = parse_sweep_key(setting_dir.name)
        for (m, eff), preds in stacks.items():
            truth = truths[eff]
            bias2 = (preds.mean(axis=0) - truth) ** 2
            var = preds.var(axis=0, ddof=0)
            mspe = ((preds - truth[None, :]) ** 2).mean(axis=0)
            base_row = dict(model=m, effect=eff, **sweep_vars)
            rows.append(dict(**base_row, metric="mspe", value=float(mspe.mean())))
            rows.append(dict(**base_row, metric="bias2", value=float(bias2.mean())))
            rows.append(dict(**base_row, metric="var", value=float(var.mean())))
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for run_dir in RUN_DIRS:
        if not run_dir.exists():
            print(f"[{run_dir.name}] missing — skipped", flush=True)
            continue
        for sweep_dir in sorted(d for d in run_dir.iterdir()
                                if d.is_dir() and (d / "preds").exists()):
            rows = aggregate_sweep(sweep_dir)
            if not rows:
                print(f"[{sweep_dir.name}] no rows", flush=True)
                continue
            fixed = ["model", "effect", "metric", "value"]
            sweep_keys = sorted({k for r in rows for k in r if k not in fixed})
            outp = OUT_DIR / f"{sweep_dir.name}.csv"
            with outp.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fixed + sweep_keys)
                w.writeheader()
                w.writerows(rows)
            print(f"[{sweep_dir.name}] {len(rows)} rows -> {outp}", flush=True)


if __name__ == "__main__":
    main()
