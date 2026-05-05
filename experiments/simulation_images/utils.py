import torch

from torch.utils.data import DataLoader
from cocodeel.dataset import CovarDataset
from experiments.simulation_images.dataset import simulate_traffic_light_data


def simulate_dataloader(simulation_params, seed=0):
    torch.manual_seed(seed)
    X, Z, y, fx, fz, fr = simulate_traffic_light_data(**simulation_params, seed=seed)
    N = X.shape[0]
    batch_size = 200 if N >= 200 else N
    train_data = CovarDataset(X[:N // 2], Z[:N // 2], y[:N // 2])
    val_data = CovarDataset(X[N // 2:], Z[N // 2:], y[N // 2:])
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size= batch_size, shuffle=False)
    return train_loader, val_loader


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


def predict(model, test_loader, device='cpu'):
    model.to(device)
    model.eval()
    y_preds, fx_preds, fz_preds = [], [], []
    for batch in test_loader:
        X_test, Z_test, y_test = batch['X'].to(device), batch['Z'].to(device), batch['y'].to(device)
        with torch.no_grad():
            y_preds.append(model(X_test, Z_test))
            fx_preds.append(model.predict_fx(X_test, Z_test))
            fz_preds.append(model.predict_fz(Z_test))
    y_preds = torch.cat(y_preds, dim=0).detach().cpu()
    fx_preds = torch.cat(fx_preds, dim=0).detach().cpu()
    fz_preds = torch.cat(fz_preds, dim=0).detach().cpu()
    return y_preds, fx_preds, fz_preds

def mspe_single_run(preds, targets):
    """ Expects preds and targets to be (n_test,) tensors. """
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    return ((preds - targets) ** 2).mean()


def evaluate_model(models, test_params):
    X, Z, y, fx, fz, fr = simulate_traffic_light_data(**test_params, seed=1234)
    test_dataset = CovarDataset(X, Z, y)
    test_loader = DataLoader(test_dataset, batch_size=test_params['n'] // 30, shuffle=False)
    y_preds, fx_preds, fz_preds = [], [], []
    for i, model in enumerate(models):
        y_pred, fx_pred, fz_pred = predict(model, test_loader)
        y_preds.append(y_pred)
        fx_preds.append(fx_pred)
        fz_preds.append(fz_pred)
    y_preds = torch.stack(y_preds, dim=1).squeeze()
    fx_preds = torch.stack(fx_preds, dim=1).squeeze()
    fz_preds = torch.stack(fz_preds, dim=1).squeeze()
    return {
        'y': mspe_decomposition(y_preds, y),
        'fx': mspe_decomposition(fx_preds, fx),
        'fr': mspe_decomposition(fx_preds, fr),
        'fz': mspe_decomposition(fz_preds, fz)
    }


def mspe_decomposition(preds, targets):
    """ Expects preds and targets to be (n_test, n_runs) tensors.
    See: https://de.wikipedia.org/wiki/Verzerrung-Varianz-Dilemma
    TODO: add better reference! """
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    mspe = ((preds-targets)**2).mean()
    bias2 = ((preds.mean(axis=1,keepdims=True) - targets)**2).mean()
    var = ((preds - preds.mean(axis=1,keepdims=True))**2).mean()
    return {"mspe": mspe, "bias2": bias2, "var": var}


def show_samples(X, Z, v1, v2, v3, y, n_show=5):
    fig, axes = plt.subplots(1, n_show, figsize=(3*n_show, 3))
    for i in range(n_show):
        ax = axes[i]
        ax.imshow(X[i,0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(
            f"X: v1={v1[i].item():.2f}, v2={v2[i].item():.2f}, v3={v3[i].item():.2f}\n"
            f"Z={Z[i].item():.2f}, y={y[i].item():.2f}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def results_to_df(results_list, N_values=[200, 2000, 20000, 200000]):
    rows = []
    for N, res in zip(N_values, results_list):
        for metric_type, metrics in res.items():  # "mse" and "bias"
            for key, val in metrics.items():
                # key looks like "base_y_mse" or "posthoc_fx_bias"
                parts = key.split("_")
                model = parts[0]      # base or posthoc
                effect = parts[1]     # y, fx, fr
                metric = metric_type  # mse or bias
                rows.append({
                    "Model": model,
                    "Metric": metric,
                    "Effect": effect,
                    "N": N,
                    "Value": val.item() if val.size == 1 else val.mean()  # handle scalar arrays
                })
    return pd.DataFrame(rows)