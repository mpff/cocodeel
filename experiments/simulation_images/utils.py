import torch

from torch.utils.data import DataLoader
from cocodeel.dataset import CovarDataset
from experiments.simulation_images.dataset import simulate_traffic_light_data


def simulate_dataloaders_split(simulation_params, seed=0,
                                dgp_fn=None, covar_transform=None):
    """Draw one dataset and return three disjoint-partition loader pairs.

    Returns ``(full, half_A, half_B)`` where each element is
    ``(train_loader, val_loader)``. ``full`` covers all N observations;
    ``half_A`` and ``half_B`` partition them 50/50 by observation index so
    their observations are disjoint. Within each partition, a 50/50 train/val
    subsplit is used for the trainer's early stopping and the posthoc's
    lambda-path validation.

    Use ``full`` and ``half_A`` to train baseline backbones; use ``half_B`` for
    the ``PostHocCovarNetwork`` refit. Observation-level disjointness makes the
    posthoc features ``H = phi(X_B; theta*)`` a deterministic function of
    ``X_B`` (not of ``y_B``), which restores exogeneity for the FWL+ridge
    refit (Pagan 1984).

    Optional kwargs ``dgp_fn`` and ``covar_transform`` allow blocks with
    custom data generation (e.g. nonlinear fz) and a covariate basis
    transform (e.g. spline basis on Z) to reuse this loader. If
    ``covar_transform`` is given, it is applied to Z before the loaders
    are built — so the model sees the transformed covariate matrix
    (e.g. a B-spline basis) directly.
    """
    if dgp_fn is None:
        dgp_fn = simulate_traffic_light_data
    torch.manual_seed(seed)
    X, Z, y, fx, fz, fr = dgp_fn(**simulation_params, seed=seed)
    if covar_transform is not None:
        Z = covar_transform(Z)
    N = X.shape[0]
    half = N // 2
    base_batch = 200 if N >= 200 else N

    def _make_pair(X_, Z_, y_):
        n = X_.shape[0]
        tr = CovarDataset(X_[:n // 2], Z_[:n // 2], y_[:n // 2])
        va = CovarDataset(X_[n // 2:], Z_[n // 2:], y_[n // 2:])
        bs = min(base_batch, n)
        return (
            DataLoader(tr, batch_size=bs, shuffle=True),
            DataLoader(va, batch_size=bs, shuffle=False),
        )

    full   = _make_pair(X, Z, y)
    half_A = _make_pair(X[:half], Z[:half], y[:half])
    half_B = _make_pair(X[half:], Z[half:], y[half:])
    return full, half_A, half_B
