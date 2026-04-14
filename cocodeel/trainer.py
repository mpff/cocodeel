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
    use_amp=False,
    amp_dtype=torch.bfloat16,
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
            ``patience=max(1, patience//3)`` and ``factor=0.5``.
        scheduler_kwargs (dict, optional): Keyword arguments forwarded to the
            scheduler constructor, e.g. ``{"step_size": 5, "gamma": 0.8}``.
        use_amp (bool): Enable mixed-precision autocast. Disabled by default
            because the optimal dtype is hardware-dependent (bf16 on
            Ampere+, fp16+GradScaler on older GPUs). The caller must opt in.
        amp_dtype: dtype for autocast. Default ``torch.bfloat16`` (A100+).
            For older GPUs pass ``torch.float16`` — but note: fp16 requires
            GradScaler, which is NOT handled here. Use bf16 or no-AMP.

    Returns:
        Trained model (on the specified device). Fit history is attached as
        trailing-underscore attributes: ``val_losses_``, ``lr_history_``,
        ``best_epoch_``, ``n_epochs_run_``.
    """
    # Default device and loss function
    device = torch.device(device or "cpu")
    loss_fn = loss_fn or nn.MSELoss()
    loss_fn = loss_fn.to(device)  # move buffers (e.g. pos_weight) to device

    # Initialize and move model to device
    model = model(**model_params).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler is not None:
        scheduler = scheduler(optimizer, **(scheduler_kwargs or {}))
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=max(1, patience // 3), factor=0.5
        )

    # Autocast only if caller asks and we are on CUDA. bf16 has fp32
    # dynamic range → no GradScaler needed. fp16 would — not supported here.
    amp_enabled = use_amp and device.type == "cuda"

    best_val_loss = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    patience_counter = 0
    val_losses_ = []
    lr_history_ = []

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch["X"].to(device, non_blocking=True)
            z = batch["Z"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                preds = model(x, z) if getattr(model, "num_covariates", 0) > 0 else model(x)
                loss = loss_fn(preds, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # --- Validation ---
        model.eval()
        val_loss_sum = torch.zeros((), device=device)
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["X"].to(device, non_blocking=True)
                z = batch["Z"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    preds = model(x, z) if getattr(model, "num_covariates", 0) > 0 else model(x)
                    val_loss_sum += loss_fn(preds, y) * x.size(0)
                n_val += x.size(0)

        val_loss = (val_loss_sum / max(1, n_val)).item()
        val_losses_.append(val_loss)
        lr_history_.append(optimizer.param_groups[0]["lr"])

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
