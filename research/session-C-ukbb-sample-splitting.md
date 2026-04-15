# Session C — UKBB Sample-Splitting Scheme

Source: `proj-orthogonalisation/notebooks/paper/synthetic_confounding/run_ukbb_experiment.py` (531 lines) and `ukbb_common.py` (275 lines).
Purpose: capture exactly how the new UKBB experiment arranges data so the posthoc refit is **sample-disjoint** from the backbone fit — this is what repaired the endogeneity described in Session B.

## 1. The splitting scheme (one fold at one `coef`)

```
[full training pool: n = 14 617 UKBB training subjects]
        │
        │  5-fold StratifiedKFold on y  →  train_ix  (80 % ≈ 11 694)
        ▼
[train_ix]                                                   ← seed: RANDOM_STATE
        │
        │  train_test_split 50/50, stratified by y[train_ix]  ← seed: fold_seed
        ▼
  ┌────────────┐       ┌────────────┐
  │   pool_A   │       │   pool_B   │   (disjoint observation indices, asserted)
  └─────┬──────┘       └─────┬──────┘
        │                    │
   resample_synthetic   resample_synthetic       (synthetic confounding with strength `coef`)
   size = 2500          size = 2500
   seed = fold_seed     seed = fold_seed + 100
        │                    │
        ▼                    ▼
     h1_idx  ⊂ pool_A     h2_idx ⊂ pool_B      (asserted disjoint on global indices)
        │                    │
        │                    │
 (backbone training)    (posthoc refit)
        │                    │
        │  80/20 train/val   │  80/20 train/val
        │  seed = fold_seed+201  seed = fold_seed+202
        ▼                    ▼
  train / val loaders    train / val loaders     ← both built via `fast_loader(...)`
        │                    │
        ▼                    ▼
   covar_trainer(       PostHocCovarNetwork(
     BaseNetwork,         base_half,
     ...,                 num_covariates={1, 2},
     train, val)          orthogonalize=False
   → base_half          ).fit(train, val,
                               max_iters=400,
                               tol=1e-3)
                          → phm_age / phm_age_sex
```

Plus one sanity model on the combined `h1_idx ∪ h2_idx` (5000 obs, seed `fold_seed+200`): `base_full`. This lets us separate "splitting helps" from "more data helps".

**Two crucial design details** (they are what make the scheme work):
1. **Split pool *before* synthetic resampling.** `pool_A`, `pool_B` are disjoint at the level of the original observations. Synthetic resampling stays within each pool. A nearest-neighbour match from the DGP could otherwise pull the same real observation into both halves.
2. **Separate RNG seeds for every sub-step.** `fold_seed` controls the pool split, `fold_seed + 100` the h1 resampling, `fold_seed + 200 / 201 / 202` the train/val splits of the three loader bundles. No seed re-use → no accidental coupling.

## 2. Why this restores posthoc exogeneity

- `base_half.θ*` = argmin of the NLL on `(X_{h1}, y_{h1})`. Hence `θ*` is a random function of `(X_{h1}, y_{h1})`.
- The posthoc refit evaluates `Φ = backbone(X_{h2}; θ*)` on **`h2`**, and solves FWL+ridge against **`y_{h2}`**.
- Because `h1 ∩ h2 = ∅` in observation space (enforced by the *pre-resampling* pool split), `y_{h2}` did not enter the optimisation that produced `θ*`. From the refit's point of view, `Φ(X_{h2}; θ*)` is a **deterministic** map `X_{h2} ↦ ℝ^q`.
- Therefore `E[Φᵀε_{h2}] = 0` under standard exogeneity of `X_{h2}` w.r.t. the residual, and the FWL+ridge estimators in `PostHocCovarNetwork._fit_effects` regain their unbiasedness. This is exactly the condition that was violated in the same-sample recipe.

This is the same argument that was used implicitly in the "pretrained backbone" and "backbone-swap" experiments of Session B — they were sample-splitting by other means (Kinetics pretrain; cross-coef swap). The new scheme is the clean, in-design version.

## 3. Things the script does that are incidental but worth knowing

- **Balanced test set** (`idx_test = resample_synthetic(y_test, Z_full_test, NTEST, 0.0, RANDOM_STATE)`): resampled once, with `coef = 0`, shared across all `(coef, fold)` pairs. So everything is evaluated on the same, unconfounded, balanced reference set.
- **Link trick for IRLS stability**: `covar_trainer` is called with `link="identity"` + `BCEWithLogitsLoss`, then `base.link` is switched to `"logit"` post-hoc so `model()` returns probabilities. Avoids chaining `sigmoid` through the loss during training; keeps logit-space gradients stable.
- **`base_full.center_effects(full_tr_ld)` / `base_half.center_effects(h1_tr_ld)`** centre `Φ` on the *training* loader of the base. `PostHocCovarNetwork.fit` then re-centres internally on the refit loader — this is correct because the posthoc model owns its own `Center` buffers.
- **Controlled prediction**: `_collect_controlled_preds` marginalises the posthoc model over the empirical Z-distribution of `h2` (the refit training set), not over the balanced test set. Matches `def:controlled-pred-empirical` in the paper.
- **Per-obs export CSVs** (`testset_predictions.csv`, `testset_predictions_controlled.csv`, `fitted_coefs.csv`, `posthoc_lambda_paths.csv`, `trainset_folds.csv`): the R plotting script consumes these directly. This is a clean shape for our own smoke-test outputs too.
- **Checkpoints** per `(coef, fold)`: `.pt` files for `base_full`, `base_half`, both `phm_*`. Lets us re-evaluate without retraining.

## 4. Minimal pseudocode spec of the splitting logic (what needs to move into cocodeel)

```python
# Inputs: full pool (X, y, Z), coef, fold_seed
# 1. Split pool BEFORE any resampling / augmentation.
pool_A, pool_B = stratified_split_by_y(pool_indices, test_size=0.5, seed=fold_seed)
assert disjoint(pool_A, pool_B)

# 2. Independently resample each half (domain-specific step — here synthetic confounding).
h1 = resample(pool_A, size=N_h, seed=fold_seed)
h2 = resample(pool_B, size=N_h, seed=fold_seed + 100)
assert disjoint(h1, h2)

# 3. Build loaders with their own internal train/val splits.
h1_tr, h1_va = make_loaders(X[h1], y[h1], Z[h1], seed=fold_seed + 201)
h2_tr, h2_va = make_loaders(X[h2], y[h2], Z[h2], seed=fold_seed + 202)

# 4. Train backbone on h1.
base = covar_trainer(BaseNetwork, ..., train_loader=h1_tr, val_loader=h1_va).center_effects(h1_tr)

# 5. Posthoc refit on h2.
posthoc = PostHocCovarNetwork(base, num_covariates=p).fit(h2_tr, h2_va)
```

Everything that happens inside `covar_trainer` and `PostHocCovarNetwork.fit` is unchanged from the current `cocodeel` package. The only new thing is the **two-loader contract** at step (3)–(5): the loaders passed to the posthoc refit come from a sample that the base was not trained on.

## 5. Where should this live in `cocodeel`?

Two options, to decide in Session E:

**(a) User-side (thin) — caller owns the split.**
`PostHocCovarNetwork.fit(train_loader, val_loader)` stays exactly as today. It is the *user's responsibility* to give it loaders drawn from a sample disjoint from the backbone. Documentation + example notebook convey the rule.
+ No API change. Zero new code paths.
− Users can silently misuse the API (same sample for both stages).

**(b) Package-side (fat) — `cocodeel` owns the split.**
Add a helper that takes (X, y, Z, backbone_factory, backbone_trainer_params, posthoc_params) and performs the split, backbone training, and refit in one call. Returns the fitted posthoc model.
+ Makes misuse harder.
− Adds complexity; a general helper has to accommodate every dataset contract (transforms, num_workers, class weights, etc.) and every backbone. Hard to keep minimal.

**Leaning (a)**: simulation notebooks and the UKBB script already build their own loaders; asking them to build two is a 5-line change, not a restructuring. I'll make the case in Session E. A short documentation note + one helper for splitting an observation-index array into disjoint halves (lives in `cocodeel.utils` or stays in experiment scripts) is probably enough.

## 6. Open points to verify in Session D (synthesis)

- How the **old simulations** in `cocodeel/experiments/simulation_images/` currently construct their refit sample. Do they reuse the base training set (same-sample pathology), or do they use a separate draw?
- Whether the **centering** step on `h2_tr_ld` in `PostHocCovarNetwork.fit` is robust to the different means in `h2` vs `h1` (it should be — `Center` re-estimates from the refit loader — but worth verifying with a tiny test).
- Whether `posthoc_age_sex` vs `posthoc_age` results in the new UKBB run behaved consistently with the sample-splitting prediction (i.e., both got `|β̂_age|` close to the simulated −0.30, `|β̂_sex|` close to 2.0). Session D will compare the numbers.
