# API design document

## Context

This repository provides an implementation of *"Controlling for Omitted
Variable Bias in Deep Neural Networks."* 
The method refits the last layer of a pretrained DNN as an additive model with control variables using a backfitting estimator with a ridge penalty.
To estimate an unbiased model, we need to pretrain and refit on disjoint datasets.
For this we use k-fold crossfitting, to efficiently use the whole sample for pre-training and last-layer refit.
The implementation is currently under the cocodeel (src) name, but should be renamed to something more sensible. 
Maybe ovb-dnn.

The reader we have in mind already works in PyTorch, has the paper open, and wants to either use `RefitCovarNetwork` (+ crossfitting) on their own DNN or audit the estimator's correctness.

## Goals

- Expose the refitting machinery as a small, stable set of classes and functions, including k-fold crossfitting. It implements the correct centering and identifiability constraints.
- Follow PyTorch conventions, so the objects behave like any other `nn.Module` 
and code integrates into standard PyTorch training loops seamlessly.
- Do not expose methods for pre-training. This is part of the standard PyTorch loop. BUT: Need a contract for the refitting so that prefit models match. How?
- Let a user declare any encoder for the variable of interest 'X', without editing the library.
- Let a user declare any encoding for the control variables of interest 'Z', without edinting the library. B-spline encoding etc. are possible but are part of the dataloader, not the algorithm.
- Separate the method from what exists purely to benchmark it.

## Non-goals

- No fold-rotation orchestration inside the package. `CrossFitEnsemble`
  (below) combines `K` already-fitted `RefitCovarNetwork`s; splitting
  data into `K` folds and training each fold's backbone on the
  complementary union is left to the caller — the right splitter is
  dataset-specific (UKBB needs `StratifiedKFold`, ADNI needs
  subject-grouped `StratifiedGroupKFold`), so the package would either
  have to pick one or expose the choice anyway.
- No pretraining loop exposed as part of the method's required pipeline.
  `covar_trainer` stays in the package because the benchmark models need
  some way to be fit for comparison figures, but pretraining the refit
  method's own backbone is the caller's standard PyTorch loop —
  `covar_trainer` is one convenient option for that loop, not a
  requirement `RefitCovarNetwork` imposes.

## Design

### The Estimator

`_BaseCovarNetwork` (`model.py`) owns what every subclass needs: the
backbone, the three `Center` buffers (`center_x`, `center_z`,
`center_y`), the intercept, and `_extract_features_from_loader` (runs the
backbone once over a whole loader — every fitting routine needs the full
design matrix at once, not per-batch access). `BaseNetwork` is the
no-covariates case (`predict_fz` returns zeros) and the object a caller
pretrains before handing it to `RefitCovarNetwork`.

`RefitCovarNetwork.__init__` takes a fitted `BaseNetwork`, copies its
`state_dict`, and adds the covariate-side parameters (`center_z`, `fz`,
and, if `orthogonalize=True`, `orth`). `.fit(train_loader, val_loader,
...)` centers on the refit sample, runs the ridge backfit (`_fit_effects`
— a glmnet-style `λ` path, closed-form for the identity link, IRLS
otherwise), and optionally fits the orthogonalization term (`predict_fx`
subtracts `orth(z)`, `predict_fz` adds it back — total `η` unchanged).

### Fitting (sample split / k-fold cross-fitting)

`RefitCovarNetwork.fit` fits one refit sample: it centers on that
sample and runs the backfit. It has no notion of folds — cross-fitting is
built on top of it, not inside it.

**`CrossFitEnsemble`** holds `K` already-fitted `RefitCovarNetwork`s,
one per fold, and implements the paper's cross-fitted ensemble
(Definition, Cross-fitted ensemble model):

```
η̂(X, Z) = (1/K) Σ_k [ β̂₀,k + φ̂₋ₖ(X)ᵀβ̂_X,k + f̂_Z,k(Z) ]
```

with effect components `f̂_X(·) = K⁻¹ Σ_k φ̂₋ₖ(·)ᵀβ̂_X,k` and
`f̂_Z(·) = K⁻¹ Σ_k f̂_Z,k(·)`, each fold currently centered only on its
own fold.

**Recentering must happen per fold, before anything is averaged — not
after, and not left out.** Each fold's `β̂₀,k` was fit relative to that
fold's own centering reference `(c_x,k, c_z,k)`. Before the `K` folds can
be combined into one reported `f̂_X`, `f̂_Z` (interpretable as effects
relative to a single, common population), each fold has to be
re-expressed on a shared reference — the pooled sample `∪ₖ hₖ`. This is
`recenter()`: for a fixed, already-fitted model, `η = β₀ + (φ(X)-c_x)ᵀβ_X
+ (Z-c_z)ᵀβ_Z` is linear in the centered features for *every* link (the
link only maps `η → μ`; it never enters `η`'s own definition), so moving
`(c_x, c_z) → (c_x', c_z')` while holding `β_X, β_Z` fixed and setting
`β₀' = β₀ + (c_x'-c_x)ᵀβ_X + (c_z'-c_z)ᵀβ_Z` leaves `η` pointwise
unchanged for every observation, exactly, for any link — a one-line
algebraic identity, not an approximation, and it uses only the *features*
of the pooled sample, never its outcomes. `RefitCovarNetwork.recenter(loader)`
implements this (generalizing the existing UKBB-only `_recenter_ensemble`
off its `NumpyCovarDataset`-specific plumbing).

Skipping the `β₀` adjustment — recentering `f̂_X`/`f̂_Z` but leaving each
fold's intercept as originally fit — silently breaks `η`: on a real
fitted two-fold pair this produced a `0.2` error in the linear predictor,
not a rounding artefact. The only way to recenter *and* end up with a
prediction that still agrees with what the ensemble actually predicts is
to recenter every fold's `(β₀, c_x, c_z)` together, then average the
**recentered** intercepts — not the raw, as-fit ones:

```
β̂₀ := K⁻¹ Σ_k β̂₀,k'      (recentered intercepts, then averaged)
f̂_X, f̂_Z as above, using each fold's recentered components
```

This is not a design choice made for convenience — it is forced. Once
every fold is recentered onto the pooled sample, `η̂` computed as the
average of the (unmodified-in-value, now commonly-referenced) per-fold
`η_k` and `η̂` computed as `β̂₀ + f̂_X(X) + f̂_Z(Z)` from the recentered,
averaged decomposition are the *same number*, exactly — not two
independently-computed quantities that happen to agree, but one identity
viewed two ways. That is what guarantees the marginalized prediction
(paper's Definition, Marginalized Prediction — an average of `η`/`μ` over
many draws of `Z`) and the reported effect curves stay consistent with
each other and with the ensemble's actual predictions.

Ensembling itself averages the linear predictor `η`, then applies the
link once — not the per-fold predictions. `predict_fx`/`predict_fz` are
already `η`-space quantities (no link applied), so averaging them per
fold is correct as-is, for any link. Full predictions are not: naively
averaging each submodel's `forward` output applies `μ = g⁻¹(η)` *inside*
each fold before the folds are combined. For the identity link `g⁻¹` is
the identity, so this happens to coincide with averaging `η` first —
every current use of two-fold averaging in the simulation study is on an
`outcome_type="continuous"` block, so nothing breaks there today. But
`mean_k(sigmoid(η_k)) ≠ sigmoid(mean_k(η_k))` in general (Jensen), so
under the logit link, averaging post-link predictions is a different —
wrong — estimator from the paper's `η̂`. `RefitCovarNetwork` gains
`predict_eta(x, z)` (the linear predictor, before `output_func`), so
`CrossFitEnsemble` can average `η` across the (recentered) folds and
apply the shared link exactly once.

`CrossFitEnsemble` does no fitting itself — it is constructed from `K`
already-fitted `RefitCovarNetwork`s (see Non-goals), and its own job is
exactly the two operations above: recenter every fold onto the pooled
sample, then average. The caller owns the fold split and trains each
fold's backbone on the complementary `K-1` folds.

### Centering and identifiability

The additive model `β₀ + f_X(X) + f_Z(Z)` is unidentified by a constant
per term, same as any GAM. `center_x`/`center_z`/`center_y`
(`transform.py`'s `Center` — a mean buffer, not a parameter) fix this:
`f_X` and `f_Z` are evaluated on centered inputs, so each is zero-mean by
construction on whatever sample centering was fit on, and `β₀` absorbs
the level. `center_effects` fits these buffers once and is idempotent by
design (`is_centered` guards a second call).

`recenter(loader)` generalizes this: it moves an already-fitted model's
centering reference to a *different* sample without re-fitting anything,
by re-deriving `β₀` in closed form (see above). This is what
cross-fitting needs — every fold recentered onto the pooled sample before
ensembling — but it is a useful primitive on its own, any time `f_X`/`f_Z`
should be reported zero-mean over a population different from the one the
model was refit on.

### Link functions

Supported links (`links.py`) are `"identity"` and `"logit"`, each a
`Link` namedtuple of `(inverse, forward, derivative, variance)`.
`inverse` is `g⁻¹: η → μ` — the one function that separates a linear
predictor from a prediction, and the one function `CrossFitEnsemble` must
apply exactly once, after averaging `η` across folds, never once per
fold (see above). Adding a link means adding one entry to `LINKS`;
nothing else in `refit_model.py` or `crossfit.py` names a link by
anything other than these four functions.

## Public Interface

`cocodeel.model` (`BaseNetwork`), `cocodeel.refit_model`
(`RefitCovarNetwork`), `cocodeel.crossfit` (`CrossFitEnsemble`),
`cocodeel.dataset` (`CovarDataset`), `cocodeel.trainer`
(`covar_trainer`), `cocodeel.transform`, `cocodeel.links`. No
package-level re-exports — `cocodeel/__init__.py` is empty; a caller
imports the class it needs directly from its module.

`CrossFitEnsemble` lives in the core package, not `cocodeel.benchmarking`
— it is the paper's own recommended estimator (the construction that
"alleviates" the cost of sample-splitting), not a baseline to compare
against.