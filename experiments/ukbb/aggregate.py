#!/usr/bin/env python
"""Reduce per-fold refit records into the R-ready rexports for the UKBB application figure.

Reads every coef=*/fold=*/record.npz under a run directory (written by
refit_from_checkpoints.py) and writes the sample-split rexports and the summary
raw_results.csv with the schemas the figure scripts read.

Usage:  python experiments/ukbb/aggregate.py --run experiments/ukbb/runs/final_v2_refit
"""
import csv
import json
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METHODS = ["base_full", "base_half", "refit_age", "refit_age_sex"]
REFITS = ["refit_age", "refit_age_sex"]

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="experiments/ukbb/runs/final_v2_refit")
args = parser.parse_args()
RUN = (ROOT / args.run) if not Path(args.run).is_absolute() else Path(args.run)
REX = RUN / "rexports"
REX.mkdir(parents=True, exist_ok=True)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  → {path.relative_to(ROOT)}  ({len(rows)} rows)")


# ── load records ──────────────────────────────────────────────────────────────
records = sorted(RUN.glob("coef=*/fold=*/record.npz"),
                 key=lambda p: (float(p.parent.parent.name.split("=")[1]), int(p.parent.name.split("=")[1])))
if not records:
    raise SystemExit(f"no record.npz under {RUN}")
print(f"reducing {len(records)} fold records from {RUN.relative_to(ROOT)}")

# ── crossfit records (run_crossfit.py): concat per-fold rows into k{K}_crossfit_results.csv ──
if "rows" in np.load(records[0], allow_pickle=True).files:
    allrows, K = [], None
    for p in records:
        d = np.load(p, allow_pickle=True)
        allrows += json.loads(str(d["rows"]))
        K = int(d["K"])
    write_csv(RUN / f"k{K}_crossfit_results.csv", allrows,
              ["method", "coef", "fold", "auc", "bacc", "auc_marg", "bacc_marg",
               "corr_age", "corr_sex", "b_age", "b_sex", "lam"])
    print("done.")
    raise SystemExit(0)

# ── rexports (refit_from_checkpoints.py records) ────────────────────────────────
preds, ctrl, coefs, lambdas, summary, trainset, crossfit = [], [], [], [], [], [], []
testset_written = False
for p in records:
    d = np.load(p, allow_pickle=True)
    coef, fold = float(d["coef"]), int(d["fold"])
    if "crossfit" in d.files:
        crossfit += json.loads(str(d["crossfit"]))
    if not testset_written:
        write_csv(REX / "testset.csv",
                  [dict(obs_id=i, y=int(d["test_y"][i]), age=float(d["test_age"][i]), sex=float(d["test_sex"][i]))
                   for i in range(len(d["test_y"]))],
                  ["obs_id", "y", "age", "sex"])
        testset_written = True
    for m in METHODS:
        yv, fxv = d[f"y__{m}"], d[f"fx__{m}"]
        preds += [dict(obs_id=i, method=m, coef=coef, fold=fold, y=float(yv[i]), fx=float(fxv[i]))
                  for i in range(len(yv))]
    for m in REFITS:
        yc = d[f"yctrl__{m}"]
        ctrl += [dict(obs_id=i, method=m, coef=coef, fold=fold, y_controlled=float(yc[i]))
                 for i in range(len(yc))]
    trainset += [dict(coef=coef, fold=fold, half=2, y=int(d["train_y"][i]),
                      age=float(d["train_age"][i]), sex=float(d["train_sex"][i]))
                 for i in range(len(d["train_y"]))]
    coefs += json.loads(str(d["coefs"]))
    lambdas += json.loads(str(d["lambdas"]))
    summary += json.loads(str(d["summary"]))

write_csv(REX / "testset_predictions.csv", preds, ["obs_id", "method", "coef", "fold", "y", "fx"])
write_csv(REX / "testset_predictions_controlled.csv", ctrl, ["obs_id", "method", "coef", "fold", "y_controlled"])
write_csv(REX / "trainset_folds.csv", trainset, ["coef", "fold", "half", "y", "age", "sex"])
write_csv(REX / "fitted_coefs.csv", coefs, ["method", "coef", "fold", "intercept", "age", "sex", "lam"])
write_csv(RUN / "raw_results.csv", summary,
          ["method", "coef", "fold", "bacc", "auc", "bacc_marg", "auc_marg",
           "corr_age", "corr_sex", "b_age", "b_sex", "lam"])
# sorted for a deterministic column order (downstream R reads by name)
lam_fields = sorted({k for r in lambdas for k in r})
lam_head = ["method", "coef", "fold", "selected_lambda"] + [k for k in lam_fields
            if k not in ("method", "coef", "fold", "selected_lambda")]
write_csv(REX / "refit_lambda_paths.csv", lambdas, lam_head)
if crossfit:
    write_csv(RUN / "crossfit_results.csv", crossfit,
              ["method", "coef", "fold", "auc", "bacc", "auc_marg", "bacc_marg",
               "corr_age", "corr_sex", "b_age", "b_sex", "lam"])
print("done.")
