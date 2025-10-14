import torch

from torch.utils.data import DataLoader
from cocodeel.dataset import CovarDataset
from experiments.simulation_images.dataset import simulate_traffic_light_data


def simulate_dataloader(simulation_params, seed=0):
    torch.manual_seed(seed)
    X, Z, y, fx, fz, fr = simulate_traffic_light_data(**simulation_params, seed=seed)
    N = X.shape[0]
    train_data = CovarDataset(X[:N // 2], Z[:N // 2], y[:N // 2])
    val_data = CovarDataset(X[N // 2:], Z[N // 2:], y[N // 2:])
    train_loader = DataLoader(train_data, batch_size= N // 30, shuffle=True)
    val_loader = DataLoader(val_data, batch_size= N // 30, shuffle=False)
    return train_loader, val_loader


def predict(model, test_loader):
    model.eval()
    y_preds, fx_preds, fz_preds = [], [], []
    for batch in test_loader:
        X_test, Z_test, y_test = batch['X'], batch['Z'], batch['y']
        with torch.no_grad():
            y_preds.append(model(X_test, Z_test))
            fx_preds.append(model.predict_fx(X_test, Z_test))
            fz_preds.append(model.predict_fz(Z_test))
    y_preds = torch.cat(y_preds, dim=0)
    fx_preds = torch.cat(fx_preds, dim=0)
    fz_preds = torch.cat(fz_preds, dim=0)
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