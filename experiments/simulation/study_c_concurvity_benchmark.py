"""Study C — concurvity benchmark: the cross-fitted refit against end-to-end competitors.

Sweep concurvity (Fig c1): n grows to 102400; the roster spans refit /
refit_orth (2-fold cross-fit), the uncontrolled base DNN, Weber's
posthoc_web (residualised predictions), Weber's residualised features
and LEACE (Belrose et al. 2023) with refitted heads, NAMs with linear and
MLP covariate effect, SSN (Ruegamer et al.), and three regularised
competitors at their oracle strength: the Siems et al. (2023) concurvity
penalty on the MLP NAM, CF-Net (Zhao et al. 2020) under its published
protocol and CIRCE (Pogodin et al. 2023) as released. Sweep concurvity_q (Fig c2):
refit vs both NAMs as backbone
width q grows. Hyperparameters come from the replicated searches in
hpsearch/: per n for concurvity; per anchor (n, q) for concurvity_q, where
every cell inherits its nearest anchor on the log scale.

Usage:  NSIM=5 python experiments/simulation/study_c_concurvity_benchmark.py [sweep ...]
        RUN_NAME=study_c_erasers METHODS=posthoc_web_features,leace NS=800,3200 python ...
        (fits the listed methods at the listed n in their own run dir; aggregate.py
        merges run dirs, a later one replacing an earlier one's method)
"""
import contextlib
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cocodeel.model import BaseNetwork
from cocodeel.refit_model import RefitCovarNetwork
from cocodeel.crossfit import CrossFitEnsemble
from cocodeel.benchmarking.model import CovarNetwork, MLPCovarNetwork
from cocodeel.benchmarking.posthoc_model import (
    FeatureErasureNetwork, PostHocOrthNetwork, SemiStructuredNetwork,
)
from cocodeel.benchmarking.adversarial_trainer import adversarial_trainer, balanced_cohort
from cocodeel.benchmarking.circe_adapter import circe_fit, CirceRosterModel
from cocodeel.trainer import covar_trainer
from cocodeel.dataset import CovarDataset

from experiments.simulation.common.backbone import TrafficBackbone
from experiments.simulation.common.dgp import simulate_traffic_light_data
from experiments.simulation.common.loaders import simulate_dataloaders_split
from experiments.simulation.common import grid_runner


# ── run config ────────────────────────────────────────────────────────────────
DEVICE = os.environ.get("COCODEEL_DEVICE", "cuda:0")
N_WORKERS = 4
NSIM = int(os.environ.get("NSIM", "50"))
Q_DEFAULT = 32
TEST_SEED = 1234
TEST_N = 800
EPOCHS_CAP = 1000
WARMUP_FRAC = 0.05
RUN_DIR = ROOT / "experiments/simulation/output/runs" / os.environ.get("RUN_NAME", "study_c")
# method families to fit (refit, refit_orth, nam, nam_mlp, base, posthoc_web,
# posthoc_web_features, leace, ssn, nam_mlp_conc, cfnet, circe); default: all
METHODS = [m for m in os.environ.get("METHODS", "").split(",") if m]
# sample sizes to fit, e.g. NS=800,3200; default: every n of the sweep
NS = [int(n) for n in os.environ.get("NS", "").split(",") if n]

# CF-Net published protocol (Zhao et al. 2020): all three optimizers at the
# same rate, a fixed budget of optimizer iterations, no early stopping, no
# gradient clipping
CFNET_LR = 2e-4
CFNET_STEP_BUDGET = 2000

N_GRID = [400, 800, 1600, 3200, 6400, 12800, 25600]
N_GRID_CONCURVITY = N_GRID + [51200, 102400]
Q_GRID = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

SIM_DEFAULTS = dict(bz=1., b2=1., b3=1., cv1=0.5, cv2=0.5, sdy=1.)

# selected by hpsearch/search_study_c.py (chosen_hps_study_c.json); the
# penalized NAM variants inherit the unpenalized nam_mlp winner; the
# competitors' strength lam by hpsearch/search_study_c_lam.py
# (chosen_hps_study_c_lam.json)
HP_PER_N = {
    400: dict(backbone=dict(lr=1e-3, wd=1e-4), nam=dict(lr=1e-2, wd=1e-4),
              nam_mlp=dict(lr=3e-3, wd=1e-4),
              nam_mlp_conc=dict(lam=10.0), cfnet=dict(lam=1.0), circe=dict(lam=1.0)),
    800: dict(backbone=dict(lr=1e-2, wd=1e-4), nam=dict(lr=3e-3, wd=1e-4),
              nam_mlp=dict(lr=1e-2, wd=1e-4),
              nam_mlp_conc=dict(lam=1.0), cfnet=dict(lam=0.1), circe=dict(lam=1.0)),
    1600: dict(backbone=dict(lr=1e-3, wd=1e-4), nam=dict(lr=1e-2, wd=1e-4),
               nam_mlp=dict(lr=1e-3, wd=1e-5),
               nam_mlp_conc=dict(lam=1.0), cfnet=dict(lam=0.5), circe=dict(lam=1.0)),
    3200: dict(backbone=dict(lr=3e-4, wd=1e-4), nam=dict(lr=1e-2, wd=1e-5),
               nam_mlp=dict(lr=1e-3, wd=1e-5),
               nam_mlp_conc=dict(lam=1.0), cfnet=dict(lam=5.0), circe=dict(lam=1.0)),
    6400: dict(backbone=dict(lr=3e-4, wd=1e-5), nam=dict(lr=1e-2, wd=1e-4),
               nam_mlp=dict(lr=3e-3, wd=1e-5),
               nam_mlp_conc=dict(lam=1.0), cfnet=dict(lam=0.5), circe=dict(lam=10.0)),
    12800: dict(backbone=dict(lr=3e-4, wd=1e-5), nam=dict(lr=1e-2, wd=1e-4),
                nam_mlp=dict(lr=3e-4, wd=1e-4),
                nam_mlp_conc=dict(lam=1.0), cfnet=dict(lam=0.1), circe=dict(lam=100.0)),
    25600: dict(backbone=dict(lr=3e-4, wd=1e-5), nam=dict(lr=3e-3, wd=1e-4),
                nam_mlp=dict(lr=3e-4, wd=1e-5),
                nam_mlp_conc=dict(lam=0.1), cfnet=dict(lam=0.5), circe=dict(lam=100.0)),
    51200: dict(backbone=dict(lr=3e-3, wd=1e-5), nam=dict(lr=3e-3, wd=1e-4),
                nam_mlp=dict(lr=3e-4, wd=1e-5),
                nam_mlp_conc=dict(lam=0.1), cfnet=dict(lam=5.0), circe=dict(lam=100.0)),
    102400: dict(backbone=dict(lr=3e-4, wd=1e-5), nam=dict(lr=1e-3, wd=1e-5),
                 nam_mlp=dict(lr=3e-4, wd=1e-4),
                 nam_mlp_conc=dict(lam=0.1), cfnet=dict(lam=5.0), circe=dict(lam=100.0)),
}
# selected by hpsearch/search_study_c_q.py (chosen_hps_study_c_q.json)
HP_Q_ANCHORS = {
    (400, 4): dict(backbone=dict(lr=3e-3, wd=1e-5), nam=dict(lr=3e-3, wd=1e-5),
                   nam_mlp=dict(lr=1e-2, wd=1e-4)),
    (400, 64): dict(backbone=dict(lr=1e-3, wd=1e-5), nam=dict(lr=1e-3, wd=1e-4),
                    nam_mlp=dict(lr=1e-2, wd=1e-4)),
    (400, 1024): dict(backbone=dict(lr=3e-4, wd=1e-5), nam=dict(lr=3e-4, wd=1e-4),
                      nam_mlp=dict(lr=3e-3, wd=1e-5)),
    (3200, 4): dict(backbone=dict(lr=1e-3, wd=1e-4), nam=dict(lr=1e-2, wd=1e-5),
                    nam_mlp=dict(lr=3e-3, wd=1e-4)),
    (3200, 64): dict(backbone=dict(lr=3e-4, wd=1e-5), nam=dict(lr=1e-2, wd=1e-4),
                     nam_mlp=dict(lr=1e-3, wd=1e-4)),
    (3200, 1024): dict(backbone=dict(lr=3e-4, wd=1e-5), nam=dict(lr=3e-3, wd=1e-4),
                       nam_mlp=dict(lr=3e-3, wd=1e-5)),
    (25600, 4): dict(backbone=dict(lr=3e-4, wd=1e-4), nam=dict(lr=3e-3, wd=1e-5),
                     nam_mlp=dict(lr=1e-3, wd=1e-4)),
    (25600, 64): dict(backbone=dict(lr=3e-4, wd=1e-4), nam=dict(lr=3e-3, wd=1e-4),
                      nam_mlp=dict(lr=3e-4, wd=1e-4)),
    (25600, 1024): dict(backbone=dict(lr=3e-4, wd=1e-4), nam=dict(lr=3e-3, wd=1e-4),
                        nam_mlp=dict(lr=3e-4, wd=1e-4)),
}


def nearest_hp_q(n, q):
    """Anchor-cell winners for (n, q), nearest in (log n, log q)."""
    anchor = min(HP_Q_ANCHORS,
                 key=lambda c: math.log(n / c[0]) ** 2 + math.log(q / c[1]) ** 2)
    return HP_Q_ANCHORS[anchor]


SWEEPS = {
    # full benchmark roster at default q
    "concurvity": dict(
        settings=[dict(n=n) for n in N_GRID_CONCURVITY],
        sweep_key_fn=lambda s: f"n={s['n']}",
        full_roster=True,
    ),
    # refit vs both NAMs as backbone capacity grows
    "concurvity_q": dict(
        settings=[dict(n=n, q=q) for n in N_GRID for q in Q_GRID],
        sweep_key_fn=lambda s: f"n={s['n']}_q={s['q']}",
        full_roster=False,
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
        patience=12, scheduler=None, scheduler_kwargs=None, warmup_frac=0.05,
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

    # penalty warm-up: unpenalized for the first warmup_frac of the epoch cap
    # (Siems et al. 2023, "for stability"); model selection and early stopping
    # start with the penalty, else the unpenalized phase would always win on
    # validation prediction loss
    warmup_epochs = int(round(warmup_frac * epochs)) if lam_reg > 0 else 0

    best_val_loss = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    patience_counter = 0
    val_losses_, lr_history_ = [], []

    for epoch in range(epochs):
        penalty_on = epoch >= warmup_epochs

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
                loss = loss_fn(preds, y)
                if penalty_on:
                    loss = loss + lam_reg * concurvity_penalty(fx, fz)
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
        # scheduling, model selection and early stopping all start with the
        # penalty: stepping the plateau scheduler on unpenalized losses would
        # anchor its best below anything the penalized phase can reach and
        # decay the learning rate before the penalty is ever active
        if not penalty_on:
            continue
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
def trainer_params(hp):
    return dict(
        device=torch.device(DEVICE),
        loss_fn=MSELoss(),
        epochs=EPOCHS_CAP,
        lr=hp["lr"],
        weight_decay=hp["wd"],
        patience=6,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_kwargs={"mode": "min", "patience": 5, "factor": 0.5},
        use_amp=True,
    )


def fit_competitor(method, lam, full_tr, full_va, hp, model_params, workdir):
    """One regularised competitor at strength lam, fit on the full sample."""
    device = torch.device(DEVICE)
    if method == "nam_mlp_conc":
        # Siems penalty on the MLP NAM, HPs inherited from the unpenalized winner
        m = train_covar_with_concurvity_reg(
            MLPCovarNetwork, model_params, full_tr, full_va, lam_reg=lam,
            warmup_frac=WARMUP_FRAC, **trainer_params(hp["nam_mlp"]))
        return m.center_effects(full_tr)
    if method == "cfnet":
        # CF-Net (Zhao et al. 2020), published protocol: the adversary sees a
        # cohort in which y and Z are independent, equal rates so lam is the
        # strength knob, fixed iteration budget instead of early stopping, no
        # gradient clipping
        ds = full_tr.dataset
        cohort = DataLoader(Subset(ds, balanced_cohort(ds.y, ds.Z)), batch_size=full_tr.batch_size, shuffle=True)
        adv = adversarial_trainer(
            BaseNetwork, model_params, num_covariates=1,
            train_loader=full_tr, val_loader=full_va,
            device=device, loss_fn=MSELoss(), epochs=math.ceil(CFNET_STEP_BUDGET / len(full_tr)),
            lr_task=CFNET_LR, lr_cp=CFNET_LR, lr_adv=CFNET_LR, lam=lam,
            patience=None, max_grad_norm=None, cohort_loader=cohort,
        )
        return adv.center_effects(full_tr)
    # CIRCE (Pogodin et al. 2023), vendored original code and released
    # protocol. Its trainer takes the first visible GPU: launch with
    # CUDA_VISIBLE_DEVICES restricted to the target device.
    with open(os.devnull, "w") as devnull, \
            contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        tr = circe_fit(full_tr.dataset.X, full_tr.dataset.Z, full_tr.dataset.y,
                       full_va.dataset.X, full_va.dataset.Z, full_va.dataset.y,
                       lam=lam, workdir=workdir)
    return CirceRosterModel(tr, full_tr)


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

    def wanted(*families):
        return not METHODS or any(f in METHODS for f in families)

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

    # hyperparameters
    q = setting.get("q", Q_DEFAULT)
    hp = nearest_hp_q(setting["n"], q) if sweep == "concurvity_q" else HP_PER_N[setting["n"]]
    model_params = dict(
        backbone=TrafficBackbone,
        backbone_params={"out_features": q},
        num_covariates=1,
        link="identity",
    )

    # backbones
    tp_backbone = trainer_params(hp["backbone"])
    models = {}
    if wanted("refit", "refit_orth"):
        base_A = covar_trainer(BaseNetwork, model_params, train_loader=hA_tr, val_loader=hA_va, **tp_backbone)
        base_A = base_A.center_effects(hA_tr)
        base_B = covar_trainer(BaseNetwork, model_params, train_loader=hB_tr, val_loader=hB_va, **tp_backbone)
        base_B = base_B.center_effects(hB_tr)

    # refit variants (2-fold cross-fit)
    refit_variants = [("refit", False)] + ([("refit_orth", True)] if cfg["full_roster"] else [])
    for name, orth in refit_variants:
        if not wanted(name):
            continue
        m_AB = RefitCovarNetwork(base_A, num_covariates=1, orthogonalize=orth).to(device)
        m_AB = m_AB.fit(hB_tr, hB_va)
        m_BA = RefitCovarNetwork(base_B, num_covariates=1, orthogonalize=orth).to(device)
        m_BA = m_BA.fit(hA_tr, hA_va)
        models[name] = CrossFitEnsemble([m_AB, m_BA]).recenter(pooled)

    # NAM variants (end-to-end on the full sample)
    if wanted("nam", "ssn"):
        nam = covar_trainer(CovarNetwork, model_params, train_loader=full_tr, val_loader=full_va,
                            **trainer_params(hp["nam"]))
        nam = nam.center_effects(full_tr)
        if wanted("nam"):
            models["nam"] = nam
    if wanted("nam_mlp"):
        nam_mlp = covar_trainer(MLPCovarNetwork, model_params, train_loader=full_tr, val_loader=full_va,
                                **trainer_params(hp["nam_mlp"]))
        models["nam_mlp"] = nam_mlp.center_effects(full_tr)

    if cfg["full_roster"]:
        # full-sample baselines and the post-hoc methods on top of them; the
        # backbone is saved so further post-hoc methods need no retraining
        if wanted("base", "posthoc_web", "posthoc_web_features", "leace"):
            base_full = covar_trainer(BaseNetwork, model_params, train_loader=full_tr, val_loader=full_va, **tp_backbone)
            base_full = base_full.center_effects(full_tr)
            ckpt_dir = RUN_DIR / sweep / "checkpoints" / key
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(base_full.state_dict(), ckpt_dir / f"base_full_seed={seed}.pt")
        if wanted("base"):
            models["base"] = base_full
        if wanted("posthoc_web"):
            web = PostHocOrthNetwork(base_full, num_covariates=1).to(device)
            models["posthoc_web"] = web.fit(full_tr, full_va)

        # SSN wraps the linear NAM
        if wanted("ssn"):
            ssn = SemiStructuredNetwork(nam).to(device)
            models["ssn"] = ssn.fit(full_tr)

        # regularised competitors at their oracle strength for n
        for method in ("nam_mlp_conc", "cfnet", "circe"):
            if wanted(method):
                workdir = RUN_DIR / sweep / "circe_ckpt" / key / f"seed={seed}"
                models[method] = fit_competitor(
                    method, hp[method]["lam"], full_tr, full_va, hp, model_params, workdir)

        # feature erasure on the full-sample backbone; last, so that the random
        # draws of the methods above are the same with or without it
        for name, eraser in [("posthoc_web_features", "residualize"), ("leace", "leace")]:
            if wanted(name):
                erased = FeatureErasureNetwork(base_full, num_covariates=1, eraser=eraser).to(device)
                models[name] = erased.fit(full_tr)


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
    del models
    torch.cuda.empty_cache()
    gc.collect()
    return dict(sweep=sweep, sweep_key=key, seed=seed, status="ok",
                wall_s=time.time() - t0, n_methods=len(arrays))


def _worker(task):
    sweep_key = SWEEPS[task["sweep"]]["sweep_key_fn"](task["setting"])
    return grid_runner.catch_errors(run_one, task, sweep_key)


# ── main ──────────────────────────────────────────────────────────────────────
def main(sweep_names):
    sweeps = {name: SWEEPS[name] for name in (sweep_names or SWEEPS)}
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    grid_runner.write_manifest(RUN_DIR, dict(
        study="c_concurvity_benchmark", device=DEVICE, n_workers=N_WORKERS,
        nsim=NSIM, q_default=Q_DEFAULT, test=dict(seed=TEST_SEED, n=TEST_N),
        methods=METHODS or "all", ns=NS or "all",
        hp_per_n=HP_PER_N,
        hp_q_anchors={f"n={n}_q={q}": v for (n, q), v in HP_Q_ANCHORS.items()},
        cfnet_lr=CFNET_LR, cfnet_step_budget=CFNET_STEP_BUDGET,
        warmup_frac=WARMUP_FRAC,
        sweeps={s: len(c["settings"]) for s, c in sweeps.items()},
    ))
    grid_runner.write_settings_csv(RUN_DIR, {s: c["settings"] for s, c in sweeps.items()},
                                   {s: c["sweep_key_fn"] for s, c in sweeps.items()})

    done = grid_runner.already_done(RUN_DIR)
    tasks = [dict(sweep=sweep, setting=setting, seed=seed)
             for sweep, cfg in sweeps.items()
             for setting in cfg["settings"]
             if not NS or setting["n"] in NS
             for seed in range(NSIM)
             if (sweep, cfg["sweep_key_fn"](setting), seed) not in done]
    print(f"{len(done)} sims cached, {len(tasks)} to run.", flush=True)
    grid_runner.run_grid(RUN_DIR, tasks, _worker, N_WORKERS)


if __name__ == "__main__":
    main(sys.argv[1:])
