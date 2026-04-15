# Research Papers Index

One citation line + one context line per entry. Full metadata in `papers/<paperid>/<paperid>_summary.md`.

---

## agarwal2021NAM
Agarwal et al. (2021). Neural Additive Models: Interpretable Machine Learning with Neural Nets. *NeurIPS* (Spotlight).
Canonical deep-learning GAM: each f_i is a separate neural net, prediction is their sum. Benchmark baseline for unbiased estimation of y and fx; fz available if Z features are routed through dedicated FeatureNNs.

## lu2021MetadataNorm
Lu et al. (2021). Metadata Normalization. *CVPR*.
Batch-level layer that regresses out confounding metadata effects from feature distributions during training. Benchmark baseline for y and fx estimation; does not produce an explicit fz estimate.
