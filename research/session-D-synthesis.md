# Session D — Synthesis

Sources: Sessions A, B, C + `proj-orthogonalisation/logs/UKBB_HighAlcSex_Synthetic_Study/2026-04-14_17-26-52_final/raw_results.csv` + `cocodeel/experiments/simulation_images/1-simulation.ipynb` (`simulate_and_fit`).

## 1. The causal story in one paragraph

The paper's posthoc recipe fits a backbone on `(X, y)`, then refits the last layer with FWL+ridge on the **same** `(X, y)`. On the training sample the residuals `ε̂` are mechanically orthogonal to the backbone's feature Jacobian (first-order conditions of backbone training), so `H = φ(X)` is an endogenous design matrix in the refit: `E[Hᵀε] ≠ 0` in population. FWL+ridge therefore stays biased — the UKBB symptom is unchanged `Corr(age, f̂_X)` and `β̂_age` ≈ 56% of truth. Splitting the pool **before** any resampling (pool A → backbone; pool B → refit) makes `H = φ(X_B)` a deterministic map of `X_B` once `θ*` is fixed on A, so `y_B` never entered the estimation of `θ*`. Exogeneity is restored, FWL+ridge regain unbiasedness, and the UKBB symptom disappears: `β̂_age` ≈ truth and `Corr(age, f̂_X)` flips sign. The many alternative fixes explored in March/April (profile likelihood, fast-`f_z`, DML, `nam_precond_refit`) were all solving the same problem with a harder instrument.

## 2. Evidence — UKBB new run (2026-04-14, 5 folds × 2 coefs)

Reading `raw_results.csv`, averaged across the 5 folds:

### `coef = 2.0` — confounded training

| method | bacc | corr(age) | corr(sex) | β̂_age (truth −0.298) | β̂_sex (truth 2.0) |
|---|---|---|---|---|---|
| `base_full`      (5000 obs, no Z) | 0.58 | **−0.45** | +0.39 | — | — |
| `base_half`      (2500 obs, no Z) | 0.58 | **−0.43** | +0.37 | — | — |
| `posthoc_age`    (base_half + Z = age)     | 0.52 | ≈ −0.11* | +0.57 | −0.239 | — |
| `posthoc_age_sex`(base_half + Z = age,sex) | 0.58 | **+0.31** | +0.12 | **−0.294** | **1.79** |

\*signed average masks large fold-to-fold variance (−0.22, +0.04, −0.02, −0.22, −0.15); point estimate is "roughly zero" after controlling for age alone.

### `coef = 0.0` — unconfounded training

| method | bacc | corr(age) | corr(sex) | β̂_age | β̂_sex |
|---|---|---|---|---|---|
| `base_full`       | 0.65 | +0.13 | +0.61 | — | — |
| `base_half`       | 0.64 | +0.18 | +0.59 | — | — |
| `posthoc_age`     | 0.63 | +0.24 | +0.53 | −0.048 | — |
| `posthoc_age_sex` | 0.71 | +0.24 | +0.36 | −0.029 | 1.62 |

Key readings:
- **`posthoc_age_sex` at coef=2.0 recovers the DGP almost exactly**: `β̂_age = −0.294` vs truth `−0.298`; `β̂_sex = 1.79` vs truth `2.0`. Contrast with the pre-split UKBB run (see Session B / Meeting Sonja 2026-04-13): `β̂_age = −0.168` at coef=−0.3, i.e. ~56% of truth.
- **Sign flip in `Corr(age, f̂_X)`**: `base_*` is strongly negatively correlated with age (the confounder was *negatively* encoded into the prediction); after posthoc with the split, the correlation flips to `+0.31`. This mirrors the "pretrained backbone" result (Session B, `−0.44 → +0.27`) — same mechanism.
- **`posthoc_age` is weaker than `posthoc_age_sex` at coef=2.0** (bacc 0.52 vs 0.58, `β̂_age` under-recovered). When sex is omitted from `Z`, the refit leaves the sex signal in `f̂_X`, which is partial OVB for sex. Consistent with the paper's theory (Theorem on OVB — Section 3.1).
- At coef=0.0, `posthoc_age_sex` **beats** `base_half` on bacc (0.71 vs 0.64): when there's no confounding, adding `Z` just adds signal, and `β̂_sex = 1.62` (truth 2.0) confirms the refit correctly recovers the sex effect. The method is not just "corrective" — it's additive in unconfounded settings too.

### Caveats worth flagging for Session E
- At `coef=2.0` two of five `posthoc_age_sex` folds select `λ` at the upper edge of the ridge path (`1.65e10`, `1.67e12`). The path should either be expanded or the selection re-examined — the current selected λ is probably within the asymptote, but the diagnostic output needs a look. Low-priority: the point estimates are still on-target.
- `posthoc_age` fold variance in `corr(age)` is large. Probably fine (5 folds × 2500 obs is not a lot for a high-dim CNN), but comparing Var across runs is a sanity check for the smoke test.

## 3. How the old same-sample recipe shows up in the cocodeel simulations

`experiments/simulation_images/1-simulation.ipynb` → `simulate_and_fit`:

```python
train_loader, val_loader = simulate_dataloader(simulation_params, seed=seed)

base_model = covar_trainer(
    BaseNetwork, ...,
    train_loader=train_loader, val_loader=val_loader,
).center_effects(train_loader)

for name, cfg in posthoc_configs.items():
    model = cfg["cls"](base_model, ..., **cfg.get("init_kwargs", {}))
    posthoc_models[name] = model.fit(train_loader, **cfg.get("fit_kwargs", {}))
```

The **same `train_loader`** is used to fit the backbone and to refit the posthoc. This is the same-sample recipe. So the paper's Figure 1b/Fig4 results are almost certainly affected by the same endogeneity as the old UKBB run; they look "fine" because (i) the backbone is a small 2-layer CNN with q=32 that does not overfit the way a ResNet-50 does, and (ii) the "direct effect / residual effect" framing of the plots reports MSPE not `Corr(age, f_x)`. But the method description in the paper is silent about splitting, and the code as shipped does not split.

This means **Session E must update the simulation study to use the split recipe**, and the smoke test at the end (nsim=10 per coef) must compare split vs same-sample to see how much the published figures would change. (Expectation: less change than on UKBB, because the CNN is small — but non-zero.)

## 4. How the old UKBB experiment code still interacts with cocodeel

- `proj-orthogonalisation/cocodeel/` is a snapshot of the cocodeel dev branch, used by the UKBB notebooks via `sys.path.insert`. `run_ukbb_experiment.py` imports `PostHocCovarNetwork` etc. from there. **Nothing about the splitting lives in cocodeel.**
- The new UKBB pipeline is built entirely at the *caller* level: the splitting, resampling, and disjointness assertions are all in `run_ukbb_experiment.py`. `cocodeel.PostHocCovarNetwork.fit(train, val)` stays a black box that takes whichever loaders the user passes.
- This is what Session C called **option (a): user-side (thin)**. It has proven to work — the new UKBB run reaches clean numbers without any API change. No API change needed in cocodeel to support this pattern.

## 5. Residual concerns after splitting

- **Concurvity (identifiability of the `f_x`/`f_z` decomposition)** is orthogonal to endogeneity. Even with an exogenous `H`, a flexible backbone can produce column directions aligned with `Z`, leaving the decomposition ambiguous by a concurvity component. The `f_z`-first convention and the mgcv-style concurvity diagnostic remain the proposed handles. Relevant for the simulation study where `c_1` can push image–confounder correlation close to 1.
- **MLP traffic-tabular failure** was ReLU-induced non-linear Z-contamination in `H`. Splitting does not fix this — it's not an endogeneity problem, it's a representation problem. If we keep a "traffic tabular" experiment, we should flag this as a known failure mode rather than treating it as paper-relevant evidence.
- **λ-path edge cases** at `coef=2.0, posthoc_age_sex` (folds 2, 4) selected λ at the upper boundary. Not a correctness issue but an implementation hygiene one — we should check whether our λ-path expansion in `PostHocCovarNetwork` is actually triggering.

## 6. What this means for the paper

- **Section 4 (Methodology)** currently describes the refit over the training features `Φ = φ(X_train)` *without* sample-splitting. A short subsection / note is needed explaining the two-sample design for posthoc refit. Argument: "for the post-hoc refit to estimate `β_X` consistently, `Φ` must be treated as a generated regressor (Pagan, 1984). Unbiasedness of FWL+ridge requires `E[Φᵀε] = 0`, which is violated on the training sample of `θ*` but holds on any sample disjoint from it. In practice we split the training pool in half; one half trains the backbone, the other half is used for posthoc estimation." (This is a ~6-line addition.)
- **Section 5 (Simulation)** figures were produced on the same-sample recipe. The smoke test in Session E tells us how much they change under the split. If the change is small (expected for the q=32 toy CNN), Fig 1b/Fig 4 remain interpretable; we just note the split in the method and reproduce the figures under the new recipe.
- **Section 6 (Application)** already needs figures. The 2026-04-14 UKBB results can plug directly into `Fig5_UKBB` slot: bar/heatmap of `β̂` and `corr` across folds for the four methods, at both `coef ∈ {0, 2}`.

## 7. Decisions and hypotheses carrying into Session E

**Decisions:**
1. **Caller owns the split** (option a in Session C) — no API change to `cocodeel`. Optional lightweight helper in `cocodeel.utils` for disjoint index splitting if simulation scripts can share it.
2. **Simulation notebooks must be updated** to use the split recipe before the smoke test. This is a ~10-line change to `simulate_and_fit` (draw two seeds; two loaders; base fits on loader A; posthoc fits on loader B).
3. **Smoke test config**: nsim=10 per `coef` / setting, same settings as Fig 1b + Fig 3 (concurvity) + Fig 4 (adversarial). Both Gaussian and Bernoulli outcomes. Comparison columns: `MSPE(f̂_X)`, `MSPE(f̂_X^re)`, plus `Bias²/Var` if the paper shows it. Also include the old same-sample recipe in the smoke test as a reference column ("old" vs "new"), so the report in Session E's step 5 has a clean before/after.
4. **Paper-irrelevant experiments get pruned ruthlessly** (traffic tabular, joint-training HP search, DML, nam_precond*, etc.) — but only after being committed/tagged so they aren't lost. Ruthless pruning after a safety commit is the instruction.

**Hypotheses to check in the smoke test:**
- H1: Splitting leaves `MSPE(f̂_X)` qualitatively unchanged on the small-q image simulations (both before and after pass the paper's claims).
- H2: Splitting **reduces the variance** of `f̂_X` across runs, because endogeneity-induced bias adds between-run variance when the backbone is retrained each sim.
- H3: At high `c_1` (image–confounder correlation), splitting does **not** rescue `f̂_X`: this is the concurvity wall, not endogeneity.
