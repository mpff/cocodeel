# Experiment design document

## Context

`experiments/` holds everything that turns the package (`DESIGN.md`) into
the paper's results: a data-generating process, a training/refit
pipeline, hyperparameter selection, and figure scripts, one tree per
paper section. Unlike dnn-shapes' `EXPERIMENT_DESIGN.md`, this is not a
proposal for a refactor that hasn't happened yet — the tree below is as
it stands, and this document records the contract it already follows.

The reader we have in mind is a reviewer reproducing a result, or a
researcher adapting one. They have the paper open and know the method.

## Layout

Two folders map to paper sections; a third, `experiments/common/`, is a
shared dependency of one of them (see "Cross-experiment dependencies").

| Folder | Paper section | Status |
|---|---|---|
| `experiments/simulation/` | Section 5 (simulation study) | Complete pipeline: DGP → sweep runner → aggregation → one R script per figure. |
| `experiments/ukbb/` | Section 6 (UK Biobank application) | Runs and figure script exist; per-figure reproduction recipes are not yet written down (see its `README.md`). |

## What each experiment must do, and where that lives

Unlike dnn-shapes' proposed one-folder-one-file-per-concern contract
(`data.py`/`run.py`/`figures.py`/`hpsearch.py`), `simulation/` —
the most complete experiment here — splits along two boundaries. By
study: one self-contained script per paper study
(`study_a_linear_consistency.py`, `study_b_misspecification.py`,
`study_c_concurvity_benchmark.py`), each owning its sweeps, method
roster, and hardcoded hyperparameters, sharing only the DGP, backbone,
loaders, and the resumable pool runner (`common/`). And by cost:
hyperparameter search is cheap and run once (`hpsearch/search_*.py`,
writing a committed `chosen_hps*.json` whose winners are hardcoded in
the studies), the sweeps are expensive and resumable (fixed run dirs,
checkpointed by `(sweep, sweep_key, seed)` triples on disk), and
aggregation is a separate, fast, deterministic step (`aggregate.py`)
so re-running it after a code fix doesn't require re-running a sweep.
Figures are one R script per figure (`figures/figure_*.R`), not one
script per experiment, because several figures share the same
aggregated CSVs but need independently tuned `ggplot` layouts.

`ukbb/` is simpler: a few runner scripts, a shared `ukbb_common.py` for
loaders and defaults, and figure scripts read from the run's exported
CSVs (`rexports/`).

Every folder's `README.md` is the actual reproduction instruction — this
document records the shape of the contract, not the commands; those go
stale independently per experiment and belong next to the code they
describe.

## Cross-experiment dependencies

`simulation/` and `ukbb/` never import each other. But `ukbb/`
*does* depend on a third, non-paper-section folder:
`experiments/common/backbones.py` (docstring-labeled "Shared CNN
backbones for ADNI and UKBB experiments") supplies `ResNet`, `Bottleneck`,
and the other 3D-ResNet building blocks that `ukbb_common.py` imports and
wires into `default_model_params()` — every UKBB training script depends
on it. The import is easy to miss by grepping: `ukbb_common.py` does
`sys.path.insert(1, str(code_root / "experiments" / "common"))` then
`from backbones import ResNet, Bottleneck` — a bare-name import after a
`sys.path` hack, not a dotted `experiments.common.backbones` string, so
`grep -rn "from experiments\."` or `grep -rln "experiments.common"` both
find nothing here. Trust `python -c "import backbones; ..."` after the
same `sys.path` insert, or read the file, not a string grep, when
checking whether this module is used.

Only `ADNISixtyFourBackbone` in that same file is actually dead code —
confirmed unused anywhere in the current tree (`grep -rln
ADNISixtyFourBackbone` finds nothing outside its own definition). Not
fixed here, just recorded, so a future consolidation doesn't have to
rediscover which half of the file is load-bearing.

## Non-goals

- No shared experiment-side library beyond `experiments/common/`'s
  ResNet building blocks (used by `ukbb/` only) — no runner base class,
  no unified config format.
- No unit tests inside `experiments/` — an experiment is checked by
  regenerating its figures and comparing against `baselines/sha256.txt`
  (at the `ovb-ddns/` level, outside this repo), not by `pytest`. See
  `TEST_DESIGN.md`'s non-goals for the same boundary from the package
  side.
- This document does not track reproduction commands — see each folder's
  `README.md`.
