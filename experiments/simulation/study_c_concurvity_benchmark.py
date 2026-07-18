"""Study C — concurvity benchmark: the cross-fitted refit against end-to-end competitors.

Sweep concurvity (Fig 2): n grows to 102400; methods are refit / refit_orth
(2-fold cross-fit), the uncontrolled base DNN, Weber's posthoc_web, NAM with
and without the Siems et al. (2023) concurvity penalty, SSN (Ruegamer et al.),
and the adversarial trainer (Zhao et al. 2020). Sweep concurvity_q (Fig c2):
refit vs NAM as backbone width q grows, with per-q learning rates.

Usage:  NSIM=5 python experiments/simulation/study_c_concurvity_benchmark.py
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cocodeel.model import BaseNetwork
from cocodeel.refit_model import RefitCovarNetwork
from cocodeel.crossfit import CrossFitEnsemble
from cocodeel.benchmarking.model import CovarNetwork
from cocodeel.benchmarking.posthoc_model import PostHocOrthNetwork, SemiStructuredNetwork
from cocodeel.benchmarking.adversarial_trainer import adversarial_trainer
from cocodeel.trainer import covar_trainer
from cocodeel.dataset import CovarDataset

from experiments.simulation.common.backbone import TrafficBackbone
from experiments.simulation.common.dgp import simulate_traffic_light_data
from experiments.simulation.common.loaders import simulate_dataloaders_split
from experiments.simulation.common import grid_runner


# ── run config ────────────────────────────────────────────────────────────────
DEVICE = os.environ.get("COCODEEL_DEVICE", "cuda:1")
N_WORKERS = 4
NSIM = int(os.environ.get("NSIM", "50"))
Q_DEFAULT = 32
TEST_SEED = 1234
TEST_N = 800
EPOCHS_CAP = 1000
RUN_DIR = ROOT / "experiments/simulation/output/runs/study_c"

# selected by hpsearch/search_default.py (hpsearch/chosen_hps.json)
HP = dict(lr=3e-3, wd=1e-5, early_pat=6, sched_pat=5)
# per-q learning rates: hpsearch/search_per_q.py (backbone) and search_per_q_nam.py (NAM)
LR_Q = {2: 1e-3, 4: 1e-3, 8: 1e-3, 16: 1e-3, 32: 1e-2,
        64: 3e-3, 128: 3e-3, 256: 1e-2, 512: 3e-3, 1024: 1e-3}
LR_Q_NAM = {2: 1e-2, 4: 1e-2, 8: 1e-2, 16: 3e-3, 32: 1e-2,
            64: 3e-3, 128: 3e-3, 256: 1e-2, 512: 1e-2, 1024: 3e-3}

N_GRID = [400, 800, 1600, 3200, 6400, 12800, 25600]
N_GRID_CONCURVITY = N_GRID + [51200, 102400]
Q_GRID = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

SIM_DEFAULTS = dict(bz=1., b2=1., b3=1., cv1=0.5, cv2=0.5, sdy=1.)

SWEEPS = {
    # full benchmark roster at default q
    "concurvity": dict(
        settings=[dict(n=n) for n in N_GRID_CONCURVITY],
        sweep_key_fn=lambda s: f"n={s['n']}",
        nam_lams=[0.0, 0.1, 1.0, 10.0],
        with_web=True, with_ssn=True, with_adversarial=True, with_refit_orth=True,
    ),
    # refit vs plain NAM as backbone capacity grows
    "concurvity_q": dict(
        settings=[dict(n=n, q=q) for n in N_GRID for q in Q_GRID],
        sweep_key_fn=lambda s: f"n={s['n']}_q={s['q']}",
        nam_lams=[0.0],
        with_web=False, with_ssn=False, with_adversarial=False, with_refit_orth=False,
    ),
}


# ── Siems et al. (2023) concurvity regulariser ────────────────────────────────
def concurvity_penalty(fx_out, fz_out):
    """Absolute Pearson correlation between the two effect components on a batch."""
    fx_c = fx_out - fx_out.mean()
    fz_c = fz_out - fz_out.mean()
    denom = fx_c.norm() * fz_c.norm() + 1e-8
    return ((fx_c * fz_c).sum() / denom).abs()


def train_covar_with_concurvity_reg(
        model_cls, model_params, train_loader, val_loader, lam_reg,
        device=None, loss_fn=None, epochs=1000, lr=1e-3, weight_decay=1e-4,
        patience=12, scheduler=None, scheduler_kwargs=None,
        use_amp=False, amp_dtype=torch.bfloat16):
    """covar_trainer with `lam_reg * |Corr(fx, fz)|` added to each batch loss (Siems et al. 2023)."""
    device = torch.device(device or "cpu")
    loss_fn = (loss_fn or nn.MSELoss()).to(device)
    model = model_cls(**model_params).to(device)

    # optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler is not None:
        scheduler = scheduler(optimizer, **(scheduler_kwargs or {}))
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=max(1, patience // 3), factor=0.5)
    amp_enabled = use_amp and device.type == "cuda"

    best_val_loss = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    patience_counter = 0
    val_losses_, lr_history_ = [], []

    for epoch in range(epochs):
        # train epoch
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

        # validation epoch (predictive loss only, no penalty)
        model.eval()
        val_loss_sum = torch.zeros((), device=device)
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["X"].to(device, non_blocking=True)
                z = batch["Z"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    preds = model.output_func(model.intercept + model.predict_fx(x) + model.predict_fz(z))
                    val_loss_sum += loss_fn(preds, y) * x.size(0)
                n_val += x.size(0)
        val_loss = (val_loss_sum / max(1, n_val)).item()
        val_losses_.append(val_loss)
        lr_history_.append(optimizer.param_groups[0]["lr"])
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

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
    model.lr_history_ = lr_history_
    model.best_epoch_ = best_epoch
    model.n_epochs_run_ = epoch + 1
    return model.to(device)


# ── shared fitting pieces ─────────────────────────────────────────────────────
def trainer_params(lr):
    return dict(
        device=torch.device(DEVICE),
        loss_fn=MSELoss(),
        epochs=EPOCHS_CAP,
        lr=lr,
        weight_decay=HP["wd"],
        patience=HP["early_pat"],
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_kwargs={"mode": "min", "patience": HP["sched_pat"], "factor": 0.5},
        use_amp=True,
    )


def gather_predictions(model, loader, device):
    """Collect y/fx/fz predictions; fr duplicates fx — a model's image effect targets fx or fr depending on its orthogonalization."""
    model.eval()
    ys, fxs, fzs = [], [], []
    with torch.no_grad():
        for b in loader:
            x = b["X"].to(device)
            z = b["Z"].to(device)
            y_hat = model(x, z) if getattr(model, "num_covariates", 0) > 0 else model(x)
            ys.append(y_hat.cpu())
            fxs.append(model.predict_fx(x, z).cpu())
            fzs.append(model.predict_fz(z).cpu())
    fx_arr = torch.cat(fxs).view(-1).numpy().astype(np.float32)
    return {
        "y": torch.cat(ys).view(-1).numpy().astype(np.float32),
        "fx": fx_arr,
        "fr": fx_arr,
        "fz": torch.cat(fzs).view(-1).numpy().astype(np.float32),
    }


# ── one simulation ────────────────────────────────────────────────────────────
def run_one(sweep, setting, seed):
    cfg = SWEEPS[sweep]
    key = cfg["sweep_key_fn"](setting)
    outdir = RUN_DIR / sweep / "preds" / key
    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / f"seed={seed}.npz"
    if npz_path.exists():
        return dict(sweep=sweep, sweep_key=key, seed=seed, status="cached", wall_s=0.0)

    device = torch.device(DEVICE)
    torch.manual_seed(seed)
    t0 = time.time()

    # data
    sim_params = dict(SIM_DEFAULTS)
    sim_params.update({k: v for k, v in setting.items() if k != "q"})
    sim_params["outcome_type"] = "continuous"
    full, half_A, half_B, pooled = simulate_dataloaders_split(sim_params, seed=seed)
    full_tr, full_va = full
    hA_tr, hA_va = half_A
    hB_tr, hB_va = half_B

    # backbones
    q = setting.get("q", Q_DEFAULT)
    lr_backbone = LR_Q[q] if sweep == "concurvity_q" else HP["lr"]
    lr_nam = LR_Q_NAM[q] if sweep == "concurvity_q" else HP["lr"]
    model_params = dict(
        backbone=TrafficBackbone,
        backbone_params={"out_features": q},
        num_covariates=1,
        link="identity",
    )
    tp = trainer_params(lr_backbone)
    base_A = covar_trainer(BaseNetwork, model_params, train_loader=hA_tr, val_loader=hA_va, **tp)
    base_A = base_A.center_effects(hA_tr)
    base_B = covar_trainer(BaseNetwork, model_params, train_loader=hB_tr, val_loader=hB_va, **tp)
    base_B = base_B.center_effects(hB_tr)

    # refit variants (2-fold cross-fit)
    refit_variants = [("refit", False)] + ([("refit_orth", True)] if cfg["with_refit_orth"] else [])
    models = {}
    for name, orth in refit_variants:
        m_AB = RefitCovarNetwork(base_A, num_covariates=1, orthogonalize=orth).to(device)
        m_AB = m_AB.fit(hB_tr, hB_va)
        m_BA = RefitCovarNetwork(base_B, num_covariates=1, orthogonalize=orth).to(device)
        m_BA = m_BA.fit(hA_tr, hA_va)
        models[name] = CrossFitEnsemble([m_AB, m_BA]).recenter(pooled)

    # full-sample baselines
    if cfg["with_web"]:
        base_full = covar_trainer(BaseNetwork, model_params, train_loader=full_tr, val_loader=full_va, **tp)
        base_full = base_full.center_effects(full_tr)
        models["base"] = base_full
        web = PostHocOrthNetwork(base_full, num_covariates=1).to(device)
        models["posthoc_web"] = web.fit(full_tr, full_va)

    # NAM variants (end-to-end on the full sample)
    tp_nam = trainer_params(lr_nam)
    for lam_reg in cfg["nam_lams"]:
        name = "nam" if lam_reg == 0.0 else f"nam_conc_{lam_reg:g}"
        m = train_covar_with_concurvity_reg(
            CovarNetwork, model_params, full_tr, full_va, lam_reg=lam_reg, **tp_nam)
        models[name] = m.center_effects(full_tr)

    # SSN wraps the plain NAM
    if cfg["with_ssn"]:
        ssn = SemiStructuredNetwork(models["nam"]).to(device)
        models["ssn"] = ssn.fit(full_tr)

    # adversarial trainer (Zhao et al. 2020) on the full sample
    # source-paper protocol: three fixed Adam rates and patience 12
    # (the trainer's defaults), not the HP-searched backbone settings
    if cfg["with_adversarial"]:
        adv = adversarial_trainer(
            BaseNetwork, model_params, num_covariates=1,
            train_loader=full_tr, val_loader=full_va,
            device=device, loss_fn=MSELoss(), epochs=EPOCHS_CAP,
        )
        models["adversarial"] = adv.center_effects(full_tr)

    # test evaluation
    test_sim = dict(sim_params, n=TEST_N)
    X_te, Z_te, y_te, fx_te, fz_te, fr_te = simulate_traffic_light_data(**test_sim, seed=TEST_SEED)
    loader = DataLoader(CovarDataset(X_te, Z_te, y_te), batch_size=min(200, TEST_N), shuffle=False)
    truths = {
        "y": y_te.view(-1).numpy().astype(np.float32),
        "fx": fx_te.view(-1).numpy().astype(np.float32),
        "fr": fr_te.view(-1).numpy().astype(np.float32),
        "fz": fz_te.view(-1).numpy().astype(np.float32),
    }
    arrays = {name: gather_predictions(m, loader, device) for name, m in models.items()}
    np.savez_compressed(
        npz_path,
        methods=np.array(list(arrays.keys())),
        **{f"{name}__{eff}": arr for name, d in arrays.items() for eff, arr in d.items()},
        **{f"truth__{eff}": arr for eff, arr in truths.items()},
        setting=np.array(json.dumps(setting)),
        sim_params=np.array(json.dumps(sim_params)),
    )

    # cleanup
    del models, base_A, base_B
    torch.cuda.empty_cache()
    gc.collect()
    return dict(sweep=sweep, sweep_key=key, seed=seed, status="ok",
                wall_s=time.time() - t0, n_methods=len(arrays))


def _worker(task):
    sweep_key = SWEEPS[task["sweep"]]["sweep_key_fn"](task["setting"])
    return grid_runner.catch_errors(run_one, task, sweep_key)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    grid_runner.write_manifest(RUN_DIR, dict(
        study="c_concurvity_benchmark", device=DEVICE, n_workers=N_WORKERS,
        nsim=NSIM, q_default=Q_DEFAULT, test=dict(seed=TEST_SEED, n=TEST_N),
        hp=HP, lr_q=LR_Q, lr_q_nam=LR_Q_NAM,
        sweeps={s: len(c["settings"]) for s, c in SWEEPS.items()},
    ))
    grid_runner.write_settings_csv(RUN_DIR, {s: c["settings"] for s, c in SWEEPS.items()},
                                   {s: c["sweep_key_fn"] for s, c in SWEEPS.items()})

    done = grid_runner.already_done(RUN_DIR)
    tasks = [dict(sweep=sweep, setting=setting, seed=seed)
             for sweep, cfg in SWEEPS.items()
             for setting in cfg["settings"]
             for seed in range(NSIM)
             if (sweep, cfg["sweep_key_fn"](setting), seed) not in done]
    print(f"{len(done)} sims cached, {len(tasks)} to run.", flush=True)
    grid_runner.run_grid(RUN_DIR, tasks, _worker, N_WORKERS)


if __name__ == "__main__":
    main()
