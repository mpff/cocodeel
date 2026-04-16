# Session A — Paper Summary (CoCoDEeL / ovbdnn)

Source: `~/Research/cocodeel-paper/sections/{1-intro, 3-theory, 4-methodology, 5-simulation, 6-application}.tex`, `paper-status.md`.
Purpose: establish what the paper currently claims, so Sessions B/C/D/E can reason about what changes when we drop in sample-splitting.

## 1. Central argument
Shortcut learning = **omitted variable bias** (OVB) from the classical regression perspective. Correcting OVB requires **additive control variables** in a GAM-style model; fairness / correlation-restriction methods (decorrelating `ŷ` from `Z`) are **not** sufficient — they estimate the residual effect `f_X^re`, not the direct effect `f_X`, and are themselves still biased by OVB.

Target venue: IEEE TNNLS. Abstract + intro + literature + theory are "publication-ready"; Section 6 (application) has prose but no figures yet; Section 5 (simulation) has draft artifacts in 5.4 / 5.5.

## 2. Model
Additive model on the linear predictor:
```
η(X, Z) = β₀ + f_X(X) + f_Z(Z),   with  E[f_X] = E[f_Z] = 0.
```
`f_X` is a DNN backbone `φ(X) ∈ ℝ^q` followed by a linear head, i.e. `f_X(X) = φ(X)ᵀβ_X`. `f_Z` is a parametric / spline smoother with smoother matrix `S_Z` (linear, P-spline, …).

Generalized case: same predictor under a link `g`, fitted by IRLS (pseudo-data + weights).

## 3. Two estimation targets
- **Direct effect** `f_X`: includes both the unique-to-X part and the `Z`-mediated part routed through `X`.
- **Residual effect** `f_X^re = f_X − E[f_X | Z]`: unique-to-X part only.

These differ when `Z` affects `X` (mediation through the image). Which one you want is application-driven: medicine often wants `f_X`; fairness often wants `f_X^re`.

Concurvity (flexible-function multi-collinearity between `f_X` and `f_Z`) makes the **full** additive model unidentifiable if you train end-to-end, but does **not** harm identifiability of `f_X^re`. This motivates the method.

## 4. Method — post-hoc ridge back-fitting
1. Pre-train a DNN `φ(X) + linear head` on the outcome **without** covariates.
2. Freeze `φ`. Extract features `Φ = φ(X_train)`, center them.
3. Refit the last layer as a semi-parametric GLM:
   ```
   β̂_X(λ) = (Φ̃ᵀ M̃_Z Φ̃ + λ I)^-1 Φ̃ᵀ M̃_Z ỹ,
   f̂_Z   = S̃_Z (ỹ − Φ̃ β̂_X),
   ```
   where `M̃_Z = I − S̃_Z` (centred residual-maker). Frisch–Waugh–Lovell form.
4. Generalized case: wrap in IRLS — weights + pseudo-responses updated each iteration.
5. Select `λ` on a log-spaced grid by validation loss; expand path if optimum is at boundary.
6. Optional **post-hoc orthogonalisation**: replace `f̂_X` with `f̂_X − S̃_Z Φ̃ β̂_X` to recover `f̂_X^re`; total `η̂` unchanged.

Marginal / "controlled" prediction: `ŷ*_ctrl = (1/n) Σ_i Ê[Y | X=X*, Z=z_i]` to make predictions invariant to the covariate distribution.

## 5. Key implementation surface in `cocodeel/`
Matches paper notation 1–to–1:
- `PostHocCovarNetwork.fit(train_loader, val_loader)` — drives the back-fit + λ path.
- `lambda_max`/`lambda_min` grid construction with edge-expansion.
- `Center` modules carry `Φ`-means as buffers so the centering travels with the state dict.
- `orthogonalize=True` flips between `f_X` and `f_X^re`.

## 6. Simulation study (Sec. 5)
- Synthetic images `X ∈ [0,1]^{20×60}` built from strip intensities `v_1, v_2, v_3`; `v_1, v_2` correlated with `Z` via strength `c_1, c_2`; `v_3` unconfounded.
- Outcome: `η = v_2ᵀβ_2 + β_3 v_3 + Zᵀβ_Z`; Gaussian or Bernoulli/logit.
- **Only `v_2, v_3` carry signal to `Y`**; `v_1` is a pure confounder-in-image.
- Backbone: 2× 2D conv → AdaptiveAvgPool 4×4 → FC → `q = 32`.
- Evaluation: MSPE (+ bias²/variance decomposition) of `f̂_X`, `f̂_X^re` over **100 independent training sets** of size `N_train`, held-out test set `|D_test| = 800`. Training set is itself split into train / val of equal size.
- Three adversarial axes studied in Fig.4: `q` ↑, number of confounders `p` ↑, image–confounder correlation `c_1` ↑. Robust to `q`, slower with `p`, fails at `c_1 → 1`.

## 7. Application (Sec. 6)
- **UKBB alcohol consumption** (synthetic confounding): resample the training set so `log-odds(y | age) = β_age · z_1`, `β_age ∈ {0, 2}`; test set is **never resampled** — gives a known reference for "unbiased image effect". Backbone: 3D-ResNet-50 (Kinetics pretrain), `q = 2048`. Control for age only (p=1) or age+sex (p=2). Five-fold stratified CV with resampling redrawn within fold.
- **ADNI Alzheimer's** (genuine confounding): age is a real confounder. Backbone: 3D CNN, `q = 256`. `f_z` is a P-spline in age, linear in sex. Ten-fold stratified group CV (group = subject, prevents subject leakage). Compared against linear `f_z`.

Both applications also plan LRP attribution maps (Zennit / EpsilonPlus) to visualise attribution shifts after control.

## 8. IMPORTANT assumption baked into the paper's posthoc recipe
The methodology section describes the post-hoc refit over the **training features `Φ = φ(X_train)`**, i.e. the same observations that were used to fit `φ`. There is **no mention of sample splitting** — Φ and y in the refit are from the sample that optimised `φ` for `y`. This is the entry point for Session B: when `φ` is a powerful overfitter, `Φ_train` and `y_train − η̂_train` are mechanically coupled, which biases the refit (endogeneity / over-optimism of β̂_X). The paper's current figures for UKBB were produced under this "same-sample" posthoc recipe; the new UKBB run (Session C) implements a split that repairs it.

## 9. Open items noted by specialists (from paper-status.md)
Not the focus here, but worth flagging because they intersect with our repo-consolidation plan:
- Section 5.4/5.5: draft artifacts must be removed.
- Section 6: **no figures yet** (Fig5_UKBB, Fig6_ADNI, LRP maps). Any figure we generate as part of this consolidation will plug directly into these slots.
- Section 7: stub conclusion.
- 7 missing `.bib` entries, 6 undefined `\cref{}`s.

These are outside the cocodeel-repo task but relevant — our smoke-test results could feed directly into the missing figures if the code is clean.
