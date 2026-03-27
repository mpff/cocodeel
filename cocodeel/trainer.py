import copy

import torch
import torch.nn as nn


def covar_trainer(
    model,
    model_params,
    train_loader,
    val_loader,
    device=None,
    loss_fn=None,
    epochs=1000,
    lr=1e-3,
    weight_decay=1e-4,
    patience=12,
    scheduler=None,
    scheduler_kwargs=None,
):
    """
    Train a covariance model with early stopping.

    Args:
        model: Model class (not instance).
        model_params (dict): Parameters for model initialization.
        train_loader, val_loader: PyTorch DataLoaders.
        device: torch.device or str, e.g. 'cuda' or 'cpu'.
        loss_fn: Loss function (default: MSELoss).
        epochs (int): Max epochs.
        lr (float): Learning rate.
        weight_decay (float): L2 regularization.
        patience (int): Early stopping patience.
        scheduler (lr_scheduler class, optional): A scheduler class (not
            instance), e.g. ``torch.optim.lr_scheduler.StepLR``. Instantiated
            internally against the optimizer using ``scheduler_kwargs``.
            Stepped once per epoch after validation. ``ReduceLROnPlateau``
            receives the validation loss automatically; all other schedulers
            are stepped without arguments.
            If None (default), uses ReduceLROnPlateau with
            ``patience=max(1, patience-2)`` and ``factor=0.5``.
        scheduler_kwargs (dict, optional): Keyword arguments forwarded to the
            scheduler constructor, e.g. ``{"step_size": 5, "gamma": 0.8}``.

    Returns:
        Trained model (on the specified device). Fit history is attached as
        trailing-underscore attributes: ``val_losses_``, ``lr_history_``,
        ``best_epoch_``, ``n_epochs_run_``.
    """
    # Default device and loss function
    device = torch.device(device or "cpu")
    loss_fn = loss_fn or nn.MSELoss()

    # Initialize and move model to device
    model = model(**model_params).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler is not None:
        scheduler = scheduler(optimizer, **(scheduler_kwargs or {}))
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=max(1, patience // 3), factor=0.5
        )

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_counter = 0
    val_losses_ = []
    lr_history_ = []

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x, z, y = batch["X"].to(device), batch["Z"].to(device), batch["y"].to(device)

            optimizer.zero_grad()
            preds = model(x, z) if getattr(model, "num_covariates", 0) > 0 else model(x)
            loss = loss_fn(preds, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                x, z, y = batch["X"].to(device), batch["Z"].to(device), batch["y"].to(device)
                preds = model(x, z) if getattr(model, "num_covariates", 0) > 0 else model(x)
                loss = loss_fn(preds, y)
                val_loss += loss.item() * x.size(0)
                n_val += x.size(0)

        val_loss /= max(1, n_val)
        val_losses_.append(val_loss)
        lr_history_.append(optimizer.param_groups[0]["lr"])

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best weights
    model.load_state_dict(best_state)
    model.val_losses_ = val_losses_
    model.lr_history_ = lr_history_
    model.best_epoch_ = best_epoch
    model.n_epochs_run_ = epoch + 1
    return model.to(device)
