# Concurvity exploration — further work, not part of the paper

Reruns the paper's `concurvity` simulation block (`experiments/simulation_images/`)
with two changes: additional end-to-end baselines beyond what the paper
reports, and outputs kept entirely out of the paper's result tree. Nothing
here is referenced by `paper/paper.tex`.

## What's here

- `run_concurvity_exploration.py` — reruns the concurvity sweep with three
  methods per `(n, seed)`: `sgd` (end-to-end `CovarNetwork`, linear
  `f_Z`), `nam` (end-to-end `CovarNetworkMLPfz`, MLP `f_Z`), and
  `posthoc_xfit` (two-fold cross-fit refit, folds averaged). DGP, backbone,
  and split helper are imported from `experiments/simulation_images/`; HPs
  are read from its `chosen_hps.json`.
- `covar_mlp_fz.py` — `CovarNetworkMLPfz`: `CovarNetwork` with a small MLP
  for `f_Z` instead of a linear map, so both component shape functions are
  non-trivial (a "proper" NAM). Handles the centering caveat that arises
  because a nonlinear `f_Z` no longer commutes with input-centering of `Z`.
- `dashboard.py` — a live HTML dashboard that re-scans a run directory and
  re-renders the MSPE/bias²/variance-vs-N figure (mirroring
  `4-Figure2_concurvity.R`) as new seeds complete.

## Running it

```
python experiments/exploration/run_concurvity_exploration.py --nsim 50 --device cuda:1
python experiments/exploration/dashboard.py --check   # render once, no server
python experiments/exploration/dashboard.py            # live server, default port 8765
```

Outputs go to `results/exploration/runs/<stamp>/`, never the paper's
`results/simulation_images/` tree.

## Status

Requires `experiments/simulation_images/chosen_hps.json` to already exist
(run `hp_search.py` there first if it doesn't). `results/exploration/` is
gitignored — nothing under it is committed.
