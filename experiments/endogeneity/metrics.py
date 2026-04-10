"""Evaluation metrics for the endogeneity test.

- predict_centered:        mean-centered fx_hat and fz_hat as numpy arrays.
- concurvity:              mgcv-style diagnostic tr(P_Z H H' P_Z) / tr(H H').
- bias_var_decomposition:  bias²/var/mspe over seed replicates.
"""
import torch


@torch.no_grad()
def predict_centered(model, X_test, Z_test):
    """Return centered fx_hat and fz_hat as numpy arrays (n_test,)."""
    model.eval()
    fx_hat = model.predict_fx(X_test).squeeze()
    fz_hat = model.predict_fz(Z_test).squeeze()
    fx_hat = fx_hat - fx_hat.mean()
    fz_hat = fz_hat - fz_hat.mean()
    return fx_hat.numpy(), fz_hat.numpy()


def concurvity(H, Z):
    """Concurvity of fx w.r.t. fz, following mgcv::concurvity(, type="worst").

    Measures the fraction of H's column space variance explainable by Z.
    concurvity = tr(P_Z H H' P_Z) / tr(H H')
              = Σ_j ||P_Z h_j||² / Σ_j ||h_j||²
              = variance-weighted average R²(h_j ~ Z).

    Args:
        H: (N, d) backbone features (centered).
        Z: (N, p) covariates (centered).

    Returns:
        float in [0, 1]. 0 = no overlap, 1 = H fully representable by Z.
    """
    # P_Z H = Z (Z'Z)^{-1} Z' H
    ZtZ_inv = torch.linalg.inv(Z.T @ Z)
    P_Z_H = Z @ ZtZ_inv @ Z.T @ H                  # (N, d)
    num = (P_Z_H ** 2).sum()                         # tr(P_Z H H' P_Z)  = ||P_Z H||_F^2
    den = (H ** 2).sum()                             # tr(H H')           = ||H||_F^2
    return float(num / den)


def bias_var_decomposition(preds, targets):
    """Bias²/Var decomposition following cocodeel simulation_images/utils.py.

    Args:
        preds:   (n_test, n_seeds) — predictions from different training runs.
        targets: (n_test,) — true values (centered), same across seeds.

    Returns:
        dict with bias2, var, mspe.
    """
    mean_pred = preds.mean(axis=1)  # (n_test,)
    bias2 = ((mean_pred - targets) ** 2).mean()
    var = ((preds - mean_pred[:, None]) ** 2).mean()
    mspe = ((preds - targets[:, None]) ** 2).mean()
    return {'bias2': float(bias2), 'var': float(var), 'mspe': float(mspe)}
