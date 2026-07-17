"""Adversarial confound-predictor baseline (Zhao, Adeli & Pohl 2020; br-net).

Trains a Z-free `BaseNetwork` end-to-end while an auxiliary confound
predictor is used adversarially to push backbone features toward zero
(squared) correlation with the covariates. A competitor to the post-hoc and
semi-structured baselines: it enforces E[f_X(X)|Z] = 0 during training
instead of after the fact, but Proposition 2 shows both routes converge to
the same biased estimate of f_X^re, not f_X. Not part of the method itself.

Reimplements https://github.com/qingyuzhao/br-net/ against this package's
`_BaseCovarNetwork`/`covar_trainer` conventions rather than porting the
original Keras code line-for-line.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfoundPredictor(nn.Module):
    """MLP predicting covariates Z from backbone features (the adversary)."""

    def __init__(self, in_features, num_covariates, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, num_covariates),
        )

    def forward(self, features):
        return self.net(features)


def _squared_corr(z_true, z_pred, eps=1e-5):
    """Sum over covariates of squared Pearson correlation (paper Eq. 2)."""
    zt = z_true - z_true.mean(dim=0, keepdim=True)
    zp = z_pred - z_pred.mean(dim=0, keepdim=True)
    num = (zt * zp).sum(dim=0)
    # eps inside the sqrt, not added after: a near-constant z_pred drives both
    # the value and the sqrt's gradient to 0/0 otherwise (the source's Keras
    # loss adds eps outside the sqrt and inherits this instability).
    den = torch.sqrt((zt ** 2).sum(dim=0) * (zp ** 2).sum(dim=0) + eps)
    r = (num / den).clamp(-1.0, 1.0)
    return (r ** 2).sum()


def adversarial_trainer(
    model,
    model_params,
    num_covariates,
    train_loader,
    val_loader,
    device=None,
    loss_fn=None,
    epochs=1000,
    lr_task=1e-4,
    lr_adv=2e-4,
    lr_cp=2e-4,
    lam=1.0,
    cp_hidden=16,
    control_label=0,
    patience=12,
):
    """Alternating task/adversary/confound-predictor training loop.

    Three separate Adam optimizers, mirroring the source's three Keras
    models: `lr_task` for (backbone, fx), `lr_adv` for backbone only (the
    adversarial step), `lr_cp` for the confound predictor. `lam` scales the
    adversarial loss, matching the paper's Eq. 3 (default 1.0 reproduces the
    source, which has no explicit weighting term).

    For `link="logit"`, the confound predictor and adversarial step are
    conditioned on `y == control_label`, matching Zhao et al.'s use of the
    reference class so batch-level correlation isn't driven by the outcome
    signal. A continuous outcome (`link="identity"`) has no such stratum, so
    the full batch is used instead — a deviation from the source, which only
    ever trains on binary case/control outcomes.

    Returns a model with the usual `val_losses_`/`best_epoch_`/`n_epochs_run_`
    fit-history attributes, plus `confound_head_` (the fitted adversary) and
    `lr_history_` as a `{"task", "adv", "cp"}` dict rather than a per-epoch
    list, since none of the three rates are scheduled.
    """
    device = torch.device("cpu" if device is None else device)
    loss_fn = loss_fn or nn.MSELoss()
    loss_fn = loss_fn.to(device)

    # models
    model = model(**model_params).to(device)
    confound_head = ConfoundPredictor(model.backbone.out_features, num_covariates, hidden=cp_hidden).to(device)

    # optimizers: task (backbone+fx+intercept), adversary (backbone only), confound predictor (cp only)
    # No scheduler, unlike covar_trainer: three independent optimizers don't
    # map onto a single ReduceLROnPlateau, and the source itself trains at a
    # fixed rate throughout.
    task_params = list(model.backbone.parameters()) + list(model.fx.parameters()) + [model.intercept]
    task_optimizer = torch.optim.Adam(task_params, lr=lr_task)
    adv_optimizer = torch.optim.Adam(model.backbone.parameters(), lr=lr_adv)
    cp_optimizer = torch.optim.Adam(confound_head.parameters(), lr=lr_cp)

    best_val_loss = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    patience_counter = 0
    val_losses_ = []

    for epoch in range(epochs):
        model.train()
        confound_head.train()
        for batch in train_loader:
            x = batch["X"].to(device, non_blocking=True)
            z = batch["Z"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            # control cohort
            if model.link == "logit":
                ctrl = y.squeeze(-1) == control_label
                x_ctrl, z_ctrl = x[ctrl], z[ctrl]
            else:
                x_ctrl, z_ctrl = x, z

            if x_ctrl.size(0) >= 2:  # Pearson correlation is undefined below n=2
                # confound predictor step
                with torch.no_grad():
                    features = model.center_x(model.backbone(x_ctrl))
                z_pred = confound_head(features)
                cp_loss = F.mse_loss(z_pred, z_ctrl)
                cp_optimizer.zero_grad()
                cp_loss.backward()
                torch.nn.utils.clip_grad_norm_(confound_head.parameters(), max_norm=1.0)
                cp_optimizer.step()

                # adversarial step: confound_head frozen (source: "regressor.trainable = False")
                for p in confound_head.parameters():
                    p.requires_grad_(False)
                features = model.center_x(model.backbone(x_ctrl))
                z_pred = confound_head(features)
                adv_loss = lam * _squared_corr(z_ctrl, z_pred)
                adv_optimizer.zero_grad()
                adv_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), max_norm=1.0)
                adv_optimizer.step()
                for p in confound_head.parameters():
                    p.requires_grad_(True)

            # task step
            task_optimizer.zero_grad()
            preds = model(x, z)
            loss = loss_fn(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(task_params, max_norm=1.0)
            task_optimizer.step()

        # validation
        model.eval()
        val_loss_sum = torch.zeros((), device=device)
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["X"].to(device, non_blocking=True)
                z = batch["Z"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                preds = model(x, z)
                val_loss_sum += loss_fn(preds, y) * x.size(0)
                n_val += x.size(0)
        val_loss = (val_loss_sum / max(1, n_val)).item()
        val_losses_.append(val_loss)

        # early stopping
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
    model.best_epoch_ = best_epoch
    model.n_epochs_run_ = epoch + 1
    # dict, not the single per-epoch list covar_trainer attaches: three fixed
    # (unscheduled) rates rather than one scheduled one.
    model.lr_history_ = {"task": lr_task, "adv": lr_adv, "cp": lr_cp}
    model.confound_head_ = confound_head
    return model.to(device)
