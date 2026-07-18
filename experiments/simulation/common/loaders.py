"""Dataloader construction for one simulated dataset, partitioned for cross-fitting."""
from torch.utils.data import DataLoader

from cocodeel.dataset import CovarDataset

from experiments.simulation.common.dgp import simulate_traffic_light_data


def simulate_dataloaders_split(simulation_params, seed=0,
                               dgp_fn=None, covar_transform=None):
    """Draw one dataset; return (full, half_A, half_B, pooled) loader groups for cross-fitting."""
    # full/half_A/half_B are (train, val) pairs; A and B partition the
    # observations 50/50 and each is subsplit 50/50 for early stopping and
    # lambda-path validation. pooled covers all N unshuffled — the common
    # reference population for CrossFitEnsemble.recenter.
    # draw data
    if dgp_fn is None:
        dgp_fn = simulate_traffic_light_data
    X, Z, y, fx, fz, fr = dgp_fn(**simulation_params, seed=seed)
    if covar_transform is not None:
        Z = covar_transform(Z)

    # partition
    N = X.shape[0]
    half = N // 2
    base_batch = 200 if N >= 200 else N

    def _make_pair(X_, Z_, y_):
        n = X_.shape[0]
        tr = CovarDataset(X_[:n // 2], Z_[:n // 2], y_[:n // 2])
        va = CovarDataset(X_[n // 2:], Z_[n // 2:], y_[n // 2:])
        bs = min(base_batch, n)
        return (DataLoader(tr, batch_size=bs, shuffle=True),
                DataLoader(va, batch_size=bs, shuffle=False))

    full = _make_pair(X, Z, y)
    half_A = _make_pair(X[:half], Z[:half], y[:half])
    half_B = _make_pair(X[half:], Z[half:], y[half:])
    pooled = DataLoader(CovarDataset(X, Z, y), batch_size=base_batch, shuffle=False)
    return full, half_A, half_B, pooled
