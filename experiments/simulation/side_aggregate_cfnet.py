"""Temporary side aggregation: concurvity CSV with source-protocol CF-Net spliced in.

Reads the main study_c concurvity NPZs, replaces every cfnet_* prediction
with the side rerun's (side_cfnet_rerun.py), and writes the same long-form
CSV schema as aggregate.py to output/side_concurvity_fixed_cfnet.csv. The
main aggregation and its CSVs are untouched.

Usage:  python experiments/simulation/side_aggregate_cfnet.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.simulation.aggregate import parse_sweep_key, EFFECTS

MAIN_DIR = ROOT / "experiments/simulation/output/runs/study_c/concurvity/preds"
SIDE_DIR = ROOT / "experiments/simulation/output/runs/study_c_side_cfnet/concurvity/preds"
OUT_PATH = ROOT / "experiments/simulation/output/side_concurvity_fixed_cfnet.csv"


def main():
    rows = []
    for setting_dir in sorted(MAIN_DIR.iterdir()):
        npz_paths = sorted(setting_dir.glob("seed=*.npz"))

        # stack seeds per (method, effect), cfnet arrays taken from the side run
        methods = None
        stacks = {}
        truths = {}
        for p in npz_paths:
            d = np.load(p, allow_pickle=False)
            s = np.load(SIDE_DIR / setting_dir.name / p.name, allow_pickle=False)
            assert np.array_equal(d["truth__fx"], s["truth__fx"]), f"truth mismatch: {p}"
            if methods is None:
                methods = [str(m) for m in d["methods"]]
                for eff in EFFECTS:
                    truths[eff] = d[f"truth__{eff}"]
            for m in methods:
                src = s if m.startswith("cfnet_") else d
                for eff in EFFECTS:
                    stacks.setdefault((m, eff), []).append(src[f"{m}__{eff}"])
        stacks = {k: np.stack(v, axis=0) for k, v in stacks.items()}

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

    fixed = ["model", "effect", "metric", "value"]
    sweep_keys = sorted({k for r in rows for k in r if k not in fixed})
    with OUT_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fixed + sweep_keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
