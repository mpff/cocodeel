"""End-to-end NAM-style training with the Siems (2023) concurvity regulariser.

The Siems penalty discourages the (X, Z)-effect components from being linearly
correlated on minibatches:

    R_concurvity(f_x, f_z) = | Corr( f_x(X_batch), f_z(Z_batch) ) |

Added to the main loss with strength `lam_reg`. Serves as a published
concurvity-regularised benchmark for Figure 2, alongside the paper's
sample-split posthoc refit.

Reference: Siems et al., "Curve Your Enthusiasm", NeurIPS 2023, Section 4.1.
Local replication: `experiments/endogeneity/siems_replication.py` on the
`experiment/endogeneity-approaches` branch of cocodeel.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def concurvity_penalty(fx_out: torch.Tensor, fz_out: torch.Tensor) -> torch.Tensor:
    """Absolute Pearson correlation between two component outputs on a batch.

    Both inputs expected to be shape (batch, 1). Returns a scalar tensor.
    """
    fx_c = fx_out - fx_out.mean()
    fz_c = fz_out - fz_out.mean()
    denom = fx_c.norm() * fz_c.norm() + 1e-8
    return ((fx_c * fz_c).sum() / denom).abs()


def train_covar_with_concurvity_reg(
    model_cls,
    model_params: dict,
    train_loader,
    val_loader,
    lam_reg: float,
    device=None,
    loss_fn=None,
    epochs: int = 1000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 12,
    scheduler=None,
    scheduler_kwargs: dict | None = None,
    use_amp: bool = False,
    amp_dtype=torch.bfloat16,
):
    """Train a CovarNetwork (or subclass) with Siems concurvity regularisation.

    Mirrors `cocodeel.trainer.covar_trainer` but adds the penalty
    `lam_reg * |Corr(fx_batch, fz_batch)|` to the loss each batch.

    Returns the trained model with the usual fit-history attributes
    (`val_losses_`, `lr_history_`, `best_epoch_`, `n_epochs_run_`).
    """
    device = torch.device(device or "cpu")
    loss_fn = (loss_fn or nn.MSELoss()).to(device)
    model = model_cls(**model_params).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler is not None:
        scheduler = scheduler(optimizer, **(scheduler_kwargs or {}))
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=max(1, patience // 3), factor=0.5
        )
    amp_enabled = use_amp and device.type == "cuda"

    best_val_loss = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    patience_counter = 0
    val_losses_, lr_history_ = [], []

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch["X"].to(device, non_blocking=True)
            z = batch["Z"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                fx = model.predict_fx(x)
                fz = model.predict_fz(z)
                preds = model.output_func(model.intercept + fx + fz)
                loss = loss_fn(preds, y) + lam_reg * concurvity_penalty(fx, fz)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss_sum = torch.zeros((), device=device)
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["X"].to(device, non_blocking=True)
                z = batch["Z"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    fx = model.predict_fx(x)
                    fz = model.predict_fz(z)
                    preds = model.output_func(model.intercept + fx + fz)
                    # Validation loss: MSE only (no penalty term) — picks weights
                    # that minimise predictive loss on held-out data.
                    val_loss_sum += loss_fn(preds, y) * x.size(0)
                n_val += x.size(0)

        val_loss = (val_loss_sum / max(1, n_val)).item()
        val_losses_.append(val_loss)
        lr_history_.append(optimizer.param_groups[0]["lr"])
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    model.val_losses_ = val_losses_
    model.lr_history_ = lr_history_
    model.best_epoch_ = best_epoch
    model.n_epochs_run_ = epoch + 1
    return model.to(device)
