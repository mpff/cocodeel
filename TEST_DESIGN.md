# Test design document

## Context

`tests/` — 102 tests across 9 files, collected by plain `pytest`
discovery (`pyproject.toml` has no `[tool.pytest.ini_options]`; none is
needed, since `tests/` is `pytest`'s default discovery target) — already
carries the instinct dnn-shapes' `TEST_DESIGN.md` set out to establish:
most tests pin a routine against a hand-derived or closed-form answer,
and `tests/conftest.py`'s `DummyBackbone` (a single `Linear` layer,
optionally initialized to the rectangular identity) is the one shared
fixture every file builds on. Unlike dnn-shapes at the time of its
`TEST_DESIGN.md`, this is not a plan for a suite that doesn't exist yet
— it's a record of the contract the existing suite already follows, so a
change to `refit_model.py`'s fitting logic can be checked against the
same standard the current tests were written to.

## Goals

- Pin the ridge backfit and IRLS solve to closed-form or known-DGP
  answers, at every link function the package supports.
- Verify the sample-split recipe (`DESIGN.md`'s "A note on sample
  splitting") actually recovers known effects when the backbone and
  refit samples are disjoint.
- Keep every historical bug fix alive as a regression test that also
  documents why the fix was needed.

## Non-goals

- No coverage-percentage target; no tests of PyTorch internals.
- No GPU path — everything here runs on CPU (`torch.manual_seed` only;
  no CUDA-specific test).
- No dependency on real UKBB/ADNI data — `DummyBackbone` and synthetic
  `torch.randn`/`torch.bernoulli` draws throughout.
- No tests inside `experiments/` — an experiment is checked by
  regenerating its figures, not by unit tests of its glue (see
  `EXPERIMENT_DESIGN.md`).

## Five kinds of test

Every test in the suite is one of these; most files mix several.

1. **Analytical/closed-form pin.** Assert against a value derived on
   paper or a reference implementation, never `==` on a float.
   `TestLinkInternals` pins `g(μ)`, `g'(μ)`, `V(μ)` for both links
   directly; `TestClosedFormRidgeSolution` derives the full
   standardize → FWL-residualize → ridge-solve → de-standardize round
   trip by hand and checks `RefitCovarNetwork.fit` against it at
   `rtol=5e-4` across three `λ` values — this is the one test that pins
   the `λ`-scale convention (`λ` enters unscaled on the standardized
   data), which no comparison against `sklearn` can check.
2. **Structural invariant.** The estimator's contract, independent of
   any specific DGP: centering leaves `fx`/`fz` zero-mean on the sample
   just fit (`test_effects_are_centered`,
   `test_refit_fx_centered_on_refit_sample`), `center_effects` is
   idempotent (`test_center_effects_is_idempotent`,
   `test_center_effects_after_fit_is_idempotent` — the latter is a
   regression test for a real bug: re-centering after IRLS had already
   set `fz ≠ 0` used to silently shift the intercept a second time), and
   orthogonalization leaves total `η` unchanged by construction (`fx`
   subtracts `orth(z)`, `fz` adds it back).
3. **Known-DGP recovery.** Fit on data with a hand-picked linear (or,
   for `TestHighDimRegression`, high-dimensional confounded) DGP and
   check the estimated `β_fx`/`β_fz` land within a stated tolerance of
   the true values — e.g. `TestSampleSplitRecoversKnownEffects` fits the
   full split recipe (backbone on `A`, refit on `B`) across 3 seeds and
   asserts mean `f̂_Z` within `0.1` of truth for the identity link,
   `0.2` for logit (non-collapsibility under ridge shrinkage is why the
   logit tolerance is wider, and why that test fits at `lam=0.0` —
   penalization, not the split estimator, is what would attenuate
   `f̂_Z` there).
4. **Regime-documentation test, including tests that fail by design.**
   `TestHighDimRegression` is a cluster of these: it pins that ridge
   collapses `β_fx` to ~0 when `λ` dominates the spectrum, that `f̂_Z`
   is highly variable at UKBB-like `n_train/d≈0.75` but converges
   cleanly at `n_train/d=9`, and includes one test explicitly marked
   `@pytest.mark.skip` whose docstring states it FAILS by design — it
   documents a known finite-sample bias in `f̂_Z` at UKBB scale
   (`d=2048, N=2000`) that is not yet fixed, rather than silently
   omitting the case. (Same principle as Wilson et al., *Best Practices
   for Scientific Computing*, PLoS Biology 2014: turn a known failure
   mode into a test case instead of a comment.)
5. **Bug-fix regression test, with the old behaviour kept alongside the
   new.** `TestIRLSConvergenceCriterion` doesn't just check that the
   current fitted-values convergence criterion works — two of its four
   tests explicitly replicate the old coefficient-change criterion
   (either for a single IRLS step inline, or across the full iteration
   loop via the `_n_iters_coeff_criterion` helper) and compare directly
   against the new one, e.g. asserting `n_new < n_old` iterations on the
   DGP that broke the old criterion (true effects near zero,
   zero-initialized coefficients). The comparison, not just the current
   behaviour, is the point: it's what makes the docstring's claimed
   pathology ("`delta = ‖β₁‖/1e-5` is huge even when `β₁` is already
   close to the solution") checkable rather than asserted.

## API coverage

| Module | Test file(s) | Kinds |
|---|---|---|
| `links.py` | `test_model.py::TestLinkInternals`, `TestGeneralizedLinkFunctions` | pin |
| `dataset.py` (`CovarDataset`) | `test_dataset.py` | invariant (keys, length, transform application) |
| `transform.py` (`Center`, `LinearRegressOut`) | `test_transform.py` | pin (`LinearRegressOut` exact solution) + invariant (`Center` zero-mean after `fit_from_loader`) |
| `model.py` (`_BaseCovarNetwork`, `BaseNetwork`) | `test_model.py` | invariant (centering, idempotence) + pin (link forward) |
| `trainer.py` (`covar_trainer`) | `test_trainer.py` | recovery (linear/logistic DGP through the full trainer→refit pipeline) + invariant (scheduler kwargs accepted, determinism under fixed seed) |
| `refit_model.py` (`RefitCovarNetwork`) | `test_refit_model.py` (the largest file, ~1400 lines) | all five kinds — this is where the ridge/IRLS solve itself lives, so it carries the closed-form pin, the regime-documentation cluster, and the convergence-criterion regression tests |
| `benchmarking/model.py`, `benchmarking/posthoc_model.py` | `test_benchmark_model.py` | invariant (centering, shape) + recovery (each benchmark against the same linear/logistic DGPs used for the method, so method-vs-baseline comparisons in the paper's figures are checkable against the same ground truth) |
| `crossfit.py` (`CrossFitEnsemble`) | `test_crossfit.py` | pin (η-space averaging: logit ensemble = sigmoid of mean η, not mean of sigmoids) + invariant (recenter-then-average identity) + recovery (known effects at both links) |
| `benchmarking/adversarial_trainer.py` | `test_adversarial_trainer.py` | pin (squared-correlation penalty on hand-built vectors) + invariant (centering, fit history, fixed-budget source protocol, small-batch control-cohort skip) + recovery (correlation reduction) |
| `benchmarking/circe_adapter.py` | `test_circe_adapter.py` | pin (vendored Gaussian kernel and unbiased HSIC against direct sums) + invariant (featurizer config matches `TrafficBackbone`, yz-cache reuse, early-stopping exit caught) |

## Conventions

- `tests/conftest.py::DummyBackbone` is the one shared fixture: a single
  `nn.Linear`, optionally initialized to the rectangular identity
  (`identity=True`) so flattened inputs pass through unchanged — the
  setting every analytical-pin test needs, since it makes the backbone
  features a deterministic, known function of the raw input.
- Module-level `rtol`, `atol` constants (`test_refit_model.py` sets
  `rtol = atol = 1e-2` near the top) rather than per-assertion magic
  numbers, except where a specific test derives its own tolerance from
  the DGP's noise level (documented inline, e.g.
  `TestRefitLogisticCovarNetwork`'s "n=250 train → finite-sample noise
  O(1/√250)≈0.06, atol=0.1").
- `@torch.no_grad()` on `setUp` and most test methods — nothing in the
  suite backprops through the model directly; every fit path (IRLS,
  ridge, least squares) is a closed-form or semi-closed-form solve.
- `torch.manual_seed(<int>)` at the top of every `setUp` for
  reproducibility; `TestDeterminism` (`test_trainer.py`) is the one test
  that checks this directly — same seed must give a bit-identical
  `state_dict` on CPU, covering every stochastic channel (data draw,
  weight init, loader shuffling, refit) in one assertion.
- Two disjoint samples, drawn by a local `sample()`/`_split_fit()`
  closure, is the standard setup wherever a test needs the split recipe
  (`test_refit_model.py`, `test_trainer.py`) — never hand-indexed into
  one draw, since that would make the "disjoint" property implicit
  rather than checked at construction.
