"""Oracle strength search for the concurvity penalty in the concurvity benchmark.

Per n of search_study_c.py, the Siems penalty on the MLP NAM sweeps its strength lam on R=5 simulation draws (the draws of
search_study_c.py, disjoint from the study's seeds) with every other HP as in
the study script. Each fit is scored by its MSPE against the true f_X and
f_X^re on a test draw that is not the study's test set; per (n, method)
the lam with the lowest mean MSPE on f_X^re wins, the estimand the penalty
targets by removing Z-dependence from f_X.
This uses the ground truth, so it gives the penalty its best strength in
hindsight. Writes search_study_c_lam.csv and chosen_hps_study_c_lam.json;
the winners are hardcoded in the study script.

Usage:  python experiments/simulation/hpsearch/search_study_c_lam.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from cocodeel.dataset import CovarDataset
from experiments.simulation.common.backbone import TrafficBackbone
from experiments.simulation.common.dgp import simulate_traffic_light_data
from experiments.simulation.common.loaders import simulate_dataloaders_split
from experiments.simulation.hpsearch import _grid_search
from experiments.simulation.hpsearch.search_study_c import SEARCH_NS, N_DRAWS, DRAW_SEED_BASE, Q
from experiments.simulation.study_c_concurvity_benchmark import (
    SIM_DEFAULTS, TEST_N, HP_PER_N, fit_nam_mlp_conc, gather_predictions,
)


# ── search config ─────────────────────────────────────────────────────────────
DEVICE = os.environ.get("COCODEEL_DEVICE", "cuda:0")
N_WORKERS = 4
SEARCH_TEST_SEED = 4321
LAMS = {
    "nam_mlp_conc": [0.001, 0.01, 0.1, 1.0, 10.0],
}

OUT_DIR = ROOT / "experiments/simulation/output/hp_search"
CHOSEN_PATH = Path(__file__).resolve().parent / "chosen_hps_study_c_lam.json"


# ── one fit ───────────────────────────────────────────────────────────────────
def fit_one(task):
    method, n, lam, draw = task["method"], task["n"], task["lam"], task["draw"]
    try:
        device = torch.device(DEVICE)
        seed = DRAW_SEED_BASE + draw
        torch.manual_seed(seed)
        sim_params = dict(SIM_DEFAULTS, n=n, outcome_type="continuous")
        full, _, _, _ = simulate_dataloaders_split(sim_params, seed=seed)
        model_params = dict(backbone=TrafficBackbone, backbone_params={"out_features": Q},
                            num_covariates=1, link="identity")
        t0 = time.time()
        model = fit_nam_mlp_conc(lam, *full, HP_PER_N[n], model_params)

        # oracle score on a separate test draw
        X_te, Z_te, y_te, fx_te, _, fr_te = simulate_traffic_light_data(
            **dict(sim_params, n=TEST_N), seed=SEARCH_TEST_SEED)
        loader = DataLoader(CovarDataset(X_te, Z_te, y_te), batch_size=200, shuffle=False)
        fx_hat = gather_predictions(model, loader, device)["fx"]
        return dict(method=method, n=n, lam=lam, draw=draw, seed=seed,
                    mspe_fx=float(((fx_hat - fx_te.view(-1).numpy()) ** 2).mean()),
                    mspe_fr=float(((fx_hat - fr_te.view(-1).numpy()) ** 2).mean()),
                    wall_s=time.time() - t0, status="ok")
    except Exception as e:
        import traceback
        return dict(method=method, n=n, lam=lam, draw=draw, error=str(e),
                    traceback=traceback.format_exc()[-1000:], status="error")


# ── selection ─────────────────────────────────────────────────────────────────
def choose_winners(rows):
    """Per (n, method): the lam with the lowest mean MSPE on f_X^re."""
    winners = {}
    for n in SEARCH_NS:
        winners[str(n)] = {}
        for method, lams in LAMS.items():
            ok = [r for r in rows if r["n"] == n and r["method"] == method and r["status"] == "ok"]
            means = {lam: float(np.mean([r["mspe_fr"] for r in ok if r["lam"] == lam])) for lam in lams}
            lam = min(means, key=means.get)
            winners[str(n)][method] = dict(lam=lam, mspe_fr=means[lam],
                                           n_draws=sum(r["lam"] == lam for r in ok))
    return winners


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    tasks = [dict(method=method, n=n, lam=lam, draw=d)
             for n in SEARCH_NS
             for method, lams in LAMS.items()
             for lam in lams
             for d in range(N_DRAWS)]
    # fits already in the CSV are kept, not rerun
    csv_path = OUT_DIR / "search_study_c_lam.csv"
    prev = [r for r in _grid_search.read_csv(csv_path) if r["status"] == "ok"]
    done = {(r["method"], r["n"], r["lam"], r["draw"]) for r in prev}
    tasks = [t for t in tasks if (t["method"], t["n"], t["lam"], t["draw"]) not in done]
    # largest fits first so the pool tail is short
    tasks.sort(key=lambda t: -t["n"])
    rows = prev + _grid_search.run_pool(
        tasks, fit_one, N_WORKERS,
        describe=lambda r: (f"{r.get('method')} n={r.get('n')} lam={r.get('lam')} "
                            f"draw={r.get('draw')} {r.get('status')} "
                            f"fx={r.get('mspe_fx', float('nan')):.4f} fr={r.get('mspe_fr', float('nan')):.4f}"))
    _grid_search.write_csv(rows, csv_path)
    winners = choose_winners(rows)
    CHOSEN_PATH.write_text(json.dumps(winners, indent=2))
    print(f"Wrote {CHOSEN_PATH}:\n" + json.dumps(winners, indent=2), flush=True)


if __name__ == "__main__":
    main()
