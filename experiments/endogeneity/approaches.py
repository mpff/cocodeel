"""Training approaches for the endogeneity test.

Each train_approach_* function takes (X_tr, Z_tr, y_tr, num_covariates) and
returns a fitted CovarNetwork ready for evaluation. The APPROACHES registry at
the bottom maps short names to training functions for the runner to iterate
over.

Approaches:
  5.  Pretrained (identity backbone, no fine-tuning) + post-hoc FWL.
       Acts as an exogenous-features baseline.
  6.  Post-hoc: train MLP backbone on X only, then post-hoc FWL refit.
       This is the current cocodeel method.
  2a. NAM: CovarNetwork, single Adam, same lr for all parameters.
  2b. Fast fz: separate Adam groups, oracle lr=0.1 for fz.
  2c. Preconditioned fz: manual (Z'Z)⁻¹-scaled Newton step per batch.
  2d. Formula lr: lr_fz = batch_size / mean(diag(Z'Z)) (no tuning).
  3b. SGD backfitting: alternating fz/fx updates per batch.
"""
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cocodeel.dataset import CovarDataset
from cocodeel.model import BaseNetwork, CovarNetwork
from cocodeel.posthoc_model import PostHocCovarNetwork

from experiments.endogeneity.backbones import IdentityBackbone, MLPBackbone


# ─── Module-level DGP/backbone sizes (used by all approaches) ─────────────────
D_IN, D_OUT = 20, 16


# ─── Shared training infrastructure ───────────────────────────────────────────
def make_loaders(X, Z, y, batch_size=64):
    """Return (train_loader, eval_loader) — same data, different shuffle."""
    tl = DataLoader(CovarDataset(X, Z, y), batch_size=batch_size, shuffle=True)
    tl_eval = DataLoader(CovarDataset(X, Z, y), batch_size=batch_size, shuffle=False)
    return tl, tl_eval


def train_loop(model, tl, optimizer, scheduler, epochs=100, patience=20):
    """Train with early stopping. No val set — use train loss for stopping."""
    loss_fn = nn.MSELoss()
    best_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_ctr = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0; nb = 0
        for batch in tl:
            optimizer.zero_grad()
            eta = model(batch['X'], batch['Z'])
            loss = loss_fn(eta, batch['y'])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item(); nb += 1
        epoch_loss /= nb

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_loss)
        else:
            scheduler.step()

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    model.load_state_dict(best_state)
    return model


# ─── Approach 5: Pretrained baseline (identity backbone + post-hoc FWL) ───────
def train_approach_5(X_tr, Z_tr, y_tr, nc):
    """Pretrained baseline: identity backbone + post-hoc FWL."""
    base = BaseNetwork(backbone=IdentityBackbone,
                       backbone_params={'in_features': D_IN, 'out_features': D_OUT},
                       num_covariates=0, link='identity')
    base.backbone.linear.weight.data = torch.eye(D_OUT, D_IN)
    base.backbone.linear.bias.data.zero_()
    w = torch.linalg.lstsq(X_tr, y_tr).solution
    base.fx.weight.data = w[:D_OUT].T

    tl_eval = DataLoader(CovarDataset(X_tr, Z_tr, y_tr), batch_size=64, shuffle=False)
    phm = PostHocCovarNetwork(base, num_covariates=nc)
    phm.fit(tl_eval, tl_eval, n_lambdas=10)

    model = CovarNetwork(backbone=IdentityBackbone,
                         backbone_params={'in_features': D_IN, 'out_features': D_OUT},
                         num_covariates=nc, link='identity')
    model.backbone.linear.weight.data = torch.eye(D_OUT, D_IN)
    model.backbone.linear.bias.data.zero_()
    model.fx.weight.data = phm.fx.weight.data.clone()
    model.fz.weight.data = phm.fz.weight.data.clone()
    model.intercept.data = phm.intercept.data.clone()
    model.center_x.mean = phm.center_x.mean.clone()
    model.center_z.mean = phm.center_z.mean.clone()
    model.center_y.mean = phm.center_y.mean.clone()
    model.is_centered.data = torch.tensor(True)
    return model


# ─── Approach 6: Post-hoc FWL on MLP-trained backbone (current method) ────────
def train_approach_posthoc(X_tr, Z_tr, y_tr, nc, epochs=100, lr=0.01):
    """Current method: train backbone WITHOUT Z, then post-hoc FWL refit."""
    base = BaseNetwork(backbone=MLPBackbone,
                       backbone_params={'in_features': D_IN, 'out_features': D_OUT},
                       num_covariates=0, link='identity')
    tl = DataLoader(CovarDataset(X_tr, Z_tr, y_tr), batch_size=64, shuffle=True)
    tl_eval = DataLoader(CovarDataset(X_tr, Z_tr, y_tr), batch_size=64, shuffle=False)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(base.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    best_loss = float('inf'); best_state = copy.deepcopy(base.state_dict()); pctr = 0
    for epoch in range(epochs):
        base.train()
        eloss = 0; nb = 0
        for batch in tl:
            opt.zero_grad()
            loss = loss_fn(base(batch['X']), batch['y'])
            loss.backward(); opt.step()
            eloss += loss.item(); nb += 1
        eloss /= nb; sched.step(eloss)
        if eloss < best_loss:
            best_loss = eloss; best_state = copy.deepcopy(base.state_dict()); pctr = 0
        else:
            pctr += 1
            if pctr >= 20: break
    base.load_state_dict(best_state)

    phm = PostHocCovarNetwork(base, num_covariates=nc)
    phm.fit(tl_eval, tl_eval, n_lambdas=10)

    model = CovarNetwork(backbone=MLPBackbone,
                         backbone_params={'in_features': D_IN, 'out_features': D_OUT},
                         num_covariates=nc, link='identity')
    model.load_state_dict(phm.state_dict(), strict=False)
    model.center_x.mean = phm.center_x.mean.clone()
    model.center_z.mean = phm.center_z.mean.clone()
    model.center_y.mean = phm.center_y.mean.clone()
    model.is_centered.data = torch.tensor(True)
    return model


# ─── Approach 2a: NAM (same lr for all parameters) ────────────────────────────
def train_approach_2a(X_tr, Z_tr, y_tr, nc, epochs=100, lr=0.01):
    """NAM-like: CovarNetwork, single optimizer, same lr."""
    model = CovarNetwork(backbone=MLPBackbone,
                         backbone_params={'in_features': D_IN, 'out_features': D_OUT},
                         num_covariates=nc, link='identity')
    tl, tl_eval = make_loaders(X_tr, Z_tr, y_tr)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    model = train_loop(model, tl, opt, sched, epochs=epochs)
    model.is_centered.data = torch.tensor(False)
    model.center_effects(tl_eval)
    return model


# ─── Approach 2b: Fast fz with oracle lr ──────────────────────────────────────
def train_approach_2b(X_tr, Z_tr, y_tr, nc, epochs=100, lr_backbone=0.01, lr_fz=0.1):
    """Fast fz: separate optimizer groups, larger lr for fz+intercept."""
    model = CovarNetwork(backbone=MLPBackbone,
                         backbone_params={'in_features': D_IN, 'out_features': D_OUT},
                         num_covariates=nc, link='identity')
    tl, tl_eval = make_loaders(X_tr, Z_tr, y_tr)
    opt = torch.optim.Adam([
        {'params': [*model.backbone.parameters(), *model.fx.parameters()],
         'lr': lr_backbone, 'weight_decay': 1e-4},
        {'params': [*model.fz.parameters(), model.intercept],
         'lr': lr_fz, 'weight_decay': 0},
    ])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    model = train_loop(model, tl, opt, sched, epochs=epochs)
    model.is_centered.data = torch.tensor(False)
    model.center_effects(tl_eval)
    return model


# ─── Approach 2c: Manually preconditioned fz update ───────────────────────────
def train_approach_2c(X_tr, Z_tr, y_tr, nc, epochs=100, lr=0.01, patience=20):
    """Preconditioned fz: (Z'Z)⁻¹-scaled manual Newton step. Z assumed standardized."""
    model = CovarNetwork(backbone=MLPBackbone,
                         backbone_params={'in_features': D_IN, 'out_features': D_OUT},
                         num_covariates=nc, link='identity')
    tl, tl_eval = make_loaders(X_tr, Z_tr, y_tr)
    loss_fn = nn.MSELoss()

    ZtZ_inv = torch.linalg.inv(Z_tr.T @ Z_tr)
    opt_fx = torch.optim.Adam(
        [*model.backbone.parameters(), *model.fx.parameters()],
        lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_fx, patience=5, factor=0.5)

    best_loss = float('inf'); best_state = copy.deepcopy(model.state_dict()); pctr = 0
    alpha_fz = 0.5

    for epoch in range(epochs):
        model.train()
        eloss = 0; nb = 0
        for batch in tl:
            x, z, yb = batch['X'], batch['Z'], batch['y']
            opt_fx.zero_grad()
            model.fz.weight.requires_grad_(True)
            model.intercept.requires_grad_(True)

            loss = loss_fn(model(x, z), yb)
            loss.backward()
            opt_fx.step()

            with torch.no_grad():
                if model.fz.weight.grad is not None:
                    delta = alpha_fz * (ZtZ_inv @ model.fz.weight.grad.T).T
                    model.fz.weight.data -= delta
                if model.intercept.grad is not None:
                    model.intercept.data -= alpha_fz * model.intercept.grad
            model.fz.weight.requires_grad_(True)
            model.intercept.requires_grad_(True)
            eloss += loss.item(); nb += 1
        eloss /= nb; sched.step(eloss)

        if eloss < best_loss:
            best_loss = eloss; best_state = copy.deepcopy(model.state_dict()); pctr = 0
        else:
            pctr += 1
            if pctr >= patience: break

    model.load_state_dict(best_state)
    model.is_centered.data = torch.tensor(False)
    model.center_effects(tl_eval)
    return model


# ─── Approach 3b: SGD backfitting ─────────────────────────────────────────────
def train_approach_3b(X_tr, Z_tr, y_tr, nc, epochs=100, lr=0.01, patience=20):
    """SGD backfitting: alternating fz/fx steps per batch."""
    model = CovarNetwork(backbone=MLPBackbone,
                         backbone_params={'in_features': D_IN, 'out_features': D_OUT},
                         num_covariates=nc, link='identity')
    tl, tl_eval = make_loaders(X_tr, Z_tr, y_tr)
    loss_fn = nn.MSELoss()

    opt_fx = torch.optim.Adam(
        [*model.backbone.parameters(), *model.fx.parameters()],
        lr=lr, weight_decay=1e-4)
    opt_fz = torch.optim.Adam(
        [*model.fz.parameters(), model.intercept], lr=lr, weight_decay=0)
    sched_fx = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_fx, patience=5, factor=0.5)
    sched_fz = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_fz, patience=5, factor=0.5)

    best_loss = float('inf'); best_state = copy.deepcopy(model.state_dict()); pctr = 0
    for epoch in range(epochs):
        model.train()
        eloss = 0; nb = 0
        for batch in tl:
            x, z, yb = batch['X'], batch['Z'], batch['y']
            # Step fz first.
            opt_fz.zero_grad()
            loss = loss_fn(model(x, z), yb)
            loss.backward(); opt_fz.step()
            # Step backbone+fx.
            opt_fx.zero_grad()
            loss = loss_fn(model(x, z), yb)
            loss.backward(); opt_fx.step()
            eloss += loss.item(); nb += 1
        eloss /= nb; sched_fx.step(eloss); sched_fz.step(eloss)

        if eloss < best_loss:
            best_loss = eloss; best_state = copy.deepcopy(model.state_dict()); pctr = 0
        else:
            pctr += 1
            if pctr >= patience: break

    model.load_state_dict(best_state)
    model.is_centered.data = torch.tensor(False)
    model.center_effects(tl_eval)
    return model


# ─── Approach 2d: Formula-derived lr for fz ───────────────────────────────────
def train_approach_2d(X_tr, Z_tr, y_tr, nc, epochs=100, lr_backbone=0.01, batch_size=64):
    """CovarNetwork with lr_fz derived from (Z'Z)⁻¹ diagonal.

    lr_fz = batch_size / diag(Z'Z).mean(). Adapts to N and covariate scale
    automatically — no HP tuning needed for fz.
    """
    model = CovarNetwork(backbone=MLPBackbone,
                         backbone_params={'in_features': D_IN, 'out_features': D_OUT},
                         num_covariates=nc, link='identity')
    tl, tl_eval = make_loaders(X_tr, Z_tr, y_tr, batch_size=batch_size)

    # Formula: lr_fz = batch_size / mean(diag(Z'Z))
    ZtZ_diag = (Z_tr.T @ Z_tr).diag()
    lr_fz = float(batch_size / ZtZ_diag.mean())

    opt = torch.optim.Adam([
        {'params': [*model.backbone.parameters(), *model.fx.parameters()],
         'lr': lr_backbone, 'weight_decay': 1e-4},
        {'params': [*model.fz.parameters(), model.intercept],
         'lr': lr_fz, 'weight_decay': 0},
    ])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    model = train_loop(model, tl, opt, sched, epochs=epochs)
    model.is_centered.data = torch.tensor(False)
    model.center_effects(tl_eval)
    return model


# ─── Registry ─────────────────────────────────────────────────────────────────
APPROACHES = {
    '5_pretrained': train_approach_5,
    '6_posthoc':    train_approach_posthoc,
    '2a_nam':       train_approach_2a,
    '2b_fast_fz':   train_approach_2b,
    '2c_precond':   train_approach_2c,
    '2d_formula':   train_approach_2d,
    '3b_backfit':   train_approach_3b,
}
