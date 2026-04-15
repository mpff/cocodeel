# Session B — Endogeneity Notes Summary

Source: `obsidian-vault/2 - Projects/Confounder Control/` — notes dated **2026-04-02 through 2026-04-13** (most recent consolidations/meetings). The relevant notes:
- `Partial Ridge Regression.md` (2026-04-02) — bias derivation for FWL+ridge.
- `Ridge Shrinkage and OVB Bias.md` (2026-04-07) — applies the derivation to UKBB numbers.
- `Exogeneity of Features in Post-hoc Models.md` (2026-04-08) — theoretical framework.
- `Pretrained backbone confirms exogeneity hypothesis.md` (2026-04-08) — UKBB swap evidence.
- `Approaches for joint fx fz training.md` (2026-04-08) — proposed fixes.
- `Concurvity attribution rule.md` (2026-04-08).
- `Profile-gradient approach.md` (2026-04-08).
- `Consolidation 2026-04-08 PostHoc FWL and Joint Training.md` — the consolidation before the sample-split hypothesis appeared.
- `Traffic tabular DGP / MLP failure / OLS OVB limit.md` (2026-04-10) — MLP analogue.
- `2026-04-12 UKBB New Methods Session.md` — infrastructure + new methods (nam_precond, DML).
- `Meeting Sonja 2026-04-13.md` — status presentation.

> **Caveat (per researcher):** these notes were written **before** we discovered that *sample-splitting* fixes the observed posthoc failure on UKBB. The "endogeneity of generated regressors" diagnosis is **correct** and **is** the operative explanation. Sample-splitting solves it directly: on a held-out refit sample, `H = backbone(X_refit)` is a *deterministic* function of `X_refit` — not a function of `y_refit` — so `E[Hᵀε] = 0` holds, and FWL+ridge become unbiased. The elaborate fixes in these notes (joint training, profile likelihood, fast-`f_z`, DML, backfitting) were solving the same endogeneity problem with a harder instrument. Session C shows the simpler instrument works.

## 1. The observed symptom (UKBB, coef=2.0, fine-tuned ResNet-50)
After the posthoc FWL refit, `Corr(age, f̂_X)` does **not move** compared to the uncontrolled base model (`−0.43 → −0.42`), and `β̂_age` is only ~56% of the simulated truth. This is the pattern Sonja was shown on 2026-04-13.

Six failure modes were ruled out first (λ miscalibration, N/d ratio, IRLS convergence, Z standardisation, unstandardised X, script bug). None of them explain it.

## 2. The closed-form bias derivation (`Partial Ridge Regression.md` + `Ridge Shrinkage and OVB Bias.md`)
For `y = Xb + Zg + ε` with FWL+ridge on `X`:

```
Bias(b̂)  = −λ (R_xᵀR_x + λI)⁻¹ b
Bias(ĝ)  =  λ Γ (R_xᵀR_x + λI)⁻¹ b,   Γ = (ZᵀZ)⁻¹ZᵀX
Var(b̂)  = σ² (R_xᵀR_x + λI)⁻¹ R_xᵀR_x (R_xᵀR_x + λI)⁻¹
```

Interpretation: `b̂` has the usual ridge bias; `ĝ` inherits an OVB-shaped term because ridge under-shrinks the portion of `b` aligned with Z. At UKBB-scale `λ ≈ 2.5e5`, this term is ~95% of full OVB — which looked like it could explain the unchanged `Corr(age, f̂_X)`.

But a β₀×λ simulation (where β₀ sets how aligned the true signal is with the Z-correlated feature direction) **ruled λ out**: debiasing works at every λ when β₀=0 (82–95% reduction) and fails at every λ when β₀∈{0.3, 1.0} (~−4% "debiasing" — i.e. none). **λ is not the problem.**

## 3. The "endogeneity of generated regressors" framing (the dominant hypothesis on 2026-04-08)
Core claim: `H = backbone_{θ*}(X)` cannot be treated like an observed design matrix in FWL, because `θ*` was obtained by minimising a loss on `y`. Therefore `H = H(X, y)`, which implies `E[Hᵀε] ≠ 0` — the "generated regressors" problem of Pagan (1984).

Evidence interpreted as supporting this:
1. **Pretrained backbone (no fine-tuning) debiases correctly.** Kinetics-pretrained ResNet-50 on UKBB coef=2.0, refit with ridge: `Corr(age, f_x)` flips `−0.44 → +0.27`; `ĝ_age ≈ −0.35` (close to LR baseline `−0.30`); `ĝ_sex ≈ 1.7` (close to true 2.0). All these numbers are **correct**. Bacc is poor (0.53–0.58) because the backbone is not task-adapted, but the *statistical* behaviour matches the theory.
2. **Backbone swap.** Using the `coef=0.0` backbone on `coef=2.0` data: `Corr(age, f_x)` drops from `−0.42` to `+0.16`. Changing the training set of `θ*` changes the "contamination" of `H`. This was read as confirmation that `H` carries Z-dependence introduced during training.
3. **Traffic-tabular DGP, MLP backbone** (`Traffic tabular MLP failure.md`): same DGP as the paper's Fig. 1b, but MLP instead of CNN. Posthoc FWL converges to `Corr ≈ 0.56` where the linear oracle converges to `0.43`. MLP mixes `v_1, v_2, v_3` non-linearly in its first ReLU layer; linear FWL can only remove the linear Z-direction. The residual non-linear contamination persists.

**All three of these findings are consistent with sample-splitting being the real fix**, but none of them required that interpretation. The endogeneity framing was compatible with every observation, so it stuck.

## 4. Concurvity, separately
`Concurvity attribution rule.md` distinguishes the *endogeneity* problem above from a *separate* identifiability problem: even with an exogenous backbone, a flexible DNN can represent any `g(Z)`, so there exists a component that can be arbitrarily split between `f_x` and `f_z`. The "f_z-first convention" (let `f_z` capture the Z-signal faster than the backbone) is the proposed resolution. Concurvity is a second-order issue that becomes visible once endogeneity is removed — important but distinct.

## 5. Proposed fixes (the direction the notes were going before the sample-split idea)
`Approaches for joint fx fz training.md` + `Profile-gradient approach.md` + `2026-04-12 UKBB New Methods Session.md`:

| ID | Idea | Status in notes |
|---|---|---|
| 1 | Profile likelihood: solve `f_z` exactly at each backbone step, train backbone on `y − f̂_z(Z)` | unimplemented, design only |
| 2d | Formula-lr fast-`f_z`: joint training, `lr_fz = batch/diag(ZᵀZ)` | Best performer on MLP toy DGP; untested on UKBB at that point |
| 3a | Epoch-level backfitting: SGD epoch on backbone; full-data IRLS for `f_z` | design only |
| DML | Double ML with 2-fold cross-fitting | Running on UKBB; single-fold bacc=0.50, `β̂_age`=−0.05 (bad) |
| `nam_precond` | 2d applied to CNN | Single fold, bacc=0.594, `β̂_age`=−0.40 (best of 2026-04-12) |
| `nam_precond_refit` | 2d + posthoc refit | Single fold, bacc=0.53, `β̂_age`=−0.04 (bad) |

All of these were motivated by the endogeneity framing. None of them ended up needed once sample-splitting was tried.

## 6. What the 2026-04-13 meeting with Sonja decided
`Meeting Sonja 2026-04-13.md` lists "going-forward" items. The first one is decisive:

> **Sample-split test** — rule out endogeneity: train base on half 1, PostHoc refit on half 2. **Currently running.**

This is the hypothesis test that was missing from the earlier consolidation. At the time of that meeting the researcher did not yet know the result. The next session (C) shows that the split resolved the symptom and therefore reframes the diagnosis.

## 7. How sample-splitting resolves the endogeneity
The endogeneity math from these notes is correct and remains the operative explanation. Same-sample failure mechanism:

- On sample A, `θ*` is obtained by minimising `ℓ(y_A, φ(X_A; θ))`. First-order conditions imply `ε̂_A ⊥ Jacobian(φ)` *by construction* on A, so `H_A = φ(X_A; θ*)` is a random function of `(X_A, y_A)`. Hence `E[H_Aᵀε] ≠ 0` — the Pagan-84 generated-regressors condition.
- On a held-out sample B (disjoint from A), `H_B = φ(X_B; θ*)` is a **deterministic** transformation of `X_B`. `y_B` did not influence `θ*`. Therefore `E[H_Bᵀε_B] = 0` whenever `E[X_Bᵀε_B] = 0`, i.e. standard exogeneity is preserved. FWL+ridge regain unbiasedness.

In short: **sample-splitting removes exactly the coupling that makes `H` endogenous**, without touching the training procedure.

Downstream consequences for the other pieces of the framework:
- The **closed-form FWL+ridge bias derivations** (Partial Ridge, Ridge Shrinkage notes) remain valid and become the correct finite-sample description of the refit **once** the split is applied.
- The **pretrained-backbone** result is a degenerate case of splitting (A is "Kinetics"; B is "UKBB") — it's why it worked. The **backbone-swap** result is also a splitting instance (A is `coef=0` UKBB; B is `coef=2` UKBB).
- **Concurvity** remains a separate identifiability issue that splitting does **not** touch. Even with an exogenous `H`, a sufficiently flexible backbone can still produce column directions in `H_B` that are aligned with `Z` — this is a property of the function class, not of the estimation step. Handled by the `f_z`-first attribution convention.
- The **MLP traffic-tabular failure** is a concurvity/representation issue that splitting alone likely does **not** fully fix: the ReLU backbone produces non-linear Z-dependence in `H` that a linear FWL cannot remove, regardless of which sample the FWL runs on. Flag for Session D.

## 8. Items to carry into the repo-cleanup plan (Session E)
- Many notebooks / scripts in the `cocodeel` repo were spun up in service of the endogeneity-diagnosis narrative (traffic-tabular, ukbb single-confound, joint-training HP search, DML). These are loose threads — kept on feature branches, typically uncommitted or only partially committed. Cleanup must *first preserve them* (commit, tag, or archive) and *then* prune aggressively so only the paper-relevant artefacts remain.
- The code changes needed for the new posthoc recipe are **small**: add the ability to fit on a separate loader from the one used to train the backbone. This can be achieved either (a) by requiring two train-loaders as user input, or (b) by folding a split into `PostHocCovarNetwork.fit`. We'll decide in Session E; Session C will clarify exactly what the UKBB implementation does.
