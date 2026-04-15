# Smoke-test report — sample-splitting vs. same-sample posthoc

Source: `results/simulation_images/smoke_sample_splitting.csv`  (360 rows, 10 sims per (setting, N), 9 (setting, N) pairs).

## Methods
- `base_full`: backbone on full N observations (paper's reference).
- `base_half`: backbone on first N/2 observations (split-recipe backbone).
- `posthoc`: PostHocCovarNetwork(base_half).fit(half_B) — **mathematically correct (split) recipe**.
- `posthoc_same_sample`: PostHocCovarNetwork(base_full).fit(full) — **biased same-sample recipe** (kept only as a reference).

Note on framing: `posthoc_same_sample` is a biased estimator (Pagan 1984
generated regressors — the refit reuses the backbone's training sample,
so H = phi(X; theta*) is endogenous w.r.t. y). At finite N its MSPE to
ground truth can happen to be lower than the split's, not because it's
closer to the truth in expectation but because both estimators are
noisy and the biased one may variance-regularise toward its own sample
noise. As N → infinity, only the split recipe is consistent.

## Per-setting MSPE(f̂_X): mean ± std across sims

| setting | base_full | base_half | posthoc | posthoc_same_sample |
|---|---|---|---|---|
| fig1b_gauss_bz0@N=400 | 0.0218 ± 0.0073 | 0.0381 ± 0.0341 | 0.0420 ± 0.0294 | 0.0291 ± 0.0144 |
| fig1b_gauss_bz1@N=400 | 0.0919 ± 0.0407 | 0.0851 ± 0.0653 | 0.1036 ± 0.1318 | 0.0475 ± 0.0349 |
| fig4a_q4@N=400 | 0.0950 ± 0.0370 | 0.0667 ± 0.0362 | 0.0787 ± 0.1236 | 0.0429 ± 0.0371 |
| fig4a_q32@N=400 | 0.0952 ± 0.0393 | 0.0878 ± 0.0632 | 0.1068 ± 0.1295 | 0.0444 ± 0.0340 |
| fig4a_q256@N=400 | 0.0898 ± 0.0448 | 0.0755 ± 0.0430 | 0.0466 ± 0.0272 | 0.0437 ± 0.0324 |
| fig4a_q1024@N=400 | 0.0972 ± 0.0381 | 0.0828 ± 0.0618 | 0.0400 ± 0.0337 | 0.0462 ± 0.0322 |
| fig4b_cv0@N=400 | 0.0456 ± 0.0261 | 0.0309 ± 0.0141 | 0.0268 ± 0.0349 | 0.0202 ± 0.0161 |
| fig4b_cv4@N=400 | 0.0630 ± 0.0287 | 0.0486 ± 0.0254 | 0.0361 ± 0.0340 | 0.0231 ± 0.0140 |
| fig4b_cv8@N=400 | 0.0920 ± 0.0406 | 0.0851 ± 0.0653 | 0.1037 ± 0.1318 | 0.0475 ± 0.0349 |

## Paired contrast: split vs. same-sample (same sim_id)

Reports mean of (MSPE_split − MSPE_same) across sims, and the fraction
of sims where split achieves lower MSPE than same-sample on the same draw.
Negative mean = split is better on average.

| setting | ΔMSPE(split − same) | P(split < same) |
|---|---|---|
| fig1b_gauss_bz0@N=400 | 0.0129 ± 0.0345 | 0.30 |
| fig1b_gauss_bz1@N=400 | 0.0561 ± 0.1377 | 0.30 |
| fig4a_q4@N=400 | 0.0358 ± 0.1172 | 0.50 |
| fig4a_q32@N=400 | 0.0624 ± 0.1330 | 0.20 |
| fig4a_q256@N=400 | 0.0028 ± 0.0366 | 0.30 |
| fig4a_q1024@N=400 | -0.0062 ± 0.0433 | 0.50 |
| fig4b_cv0@N=400 | 0.0065 ± 0.0420 | 0.60 |
| fig4b_cv4@N=400 | 0.0130 ± 0.0401 | 0.50 |
| fig4b_cv8@N=400 | 0.0562 ± 0.1378 | 0.30 |

## Variance (H2): std of f̂_X prediction means across sims

Higher variance across sims = less stable estimate. If the split recipe
restores exogeneity, we expect lower between-sim variance in `posthoc`
than in `posthoc_same_sample`.

| setting | std(mean_fx) posthoc | std(mean_fx) same_sample |
|---|---|---|
| fig1b_gauss_bz0@N=400 | 0.0273 | 0.0355 |
| fig1b_gauss_bz1@N=400 | 0.0445 | 0.0354 |
| fig4a_q4@N=400 | 0.0471 | 0.0373 |
| fig4a_q32@N=400 | 0.0455 | 0.0352 |
| fig4a_q256@N=400 | 0.0362 | 0.0336 |
| fig4a_q1024@N=400 | 0.0313 | 0.0342 |
| fig4b_cv0@N=400 | 0.0281 | 0.0312 |
| fig4b_cv4@N=400 | 0.0321 | 0.0322 |
| fig4b_cv8@N=400 | 0.0445 | 0.0354 |

## Notes

This is the initial smoke at N = 400 — too small to show the asymptotic
behaviour the paper cares about. Do not draw conclusions from the
posthoc-vs-same-sample contrast at this N: `posthoc_same_sample` is a
biased estimator that can win on MSPE-to-truth at small N because both
estimators are noisy. The interpretable result lives at larger N — see
`research/smoke-nlimit-report.md` (N = 1600) for the actual evaluation
of the split recipe and the cv1 = 0.95 concurvity-wall finding.

`fig3_binary` is missing from this run — at N = 400 the IRLS loop
needed more iterations to converge cleanly; that setting was added
back in the N = 1600 run.
