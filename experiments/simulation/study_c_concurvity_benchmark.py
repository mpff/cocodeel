"""Study C — concurvity benchmark: the cross-fitted refit against end-to-end competitors.

Sweep concurvity (Fig 2, figures/figure_c1_concurvity.R): n grows to 102400; the
roster is the DNN with Controls with and without orthogonalisation (refit,
refit_orth; 2-fold cross-fit), the DNN without controls (base) and its
orthogonalised predictions (posthoc_web), PHO + Refit (posthoc_web_features,
Weber et al. 2023) and LEACE (Belrose et al. 2023) with refitted heads, NAMs
with linear and MLP covariate effect (nam, nam_mlp), SSN (Ruegamer et al.),
the concurvity penalty of Siems et al. (2023) on the MLP NAM (nam_mlp_conc),
and a joint DNN g(X, Z) marginalised over Z (joint). Sweep concurvity_q
(appendix, figures/figure_c2_concurvity_backbone_width.R): refit against both
NAMs as the backbone width q grows. Hyperparameters are per n for concurvity
and per anchor (n, q) for concurvity_q, where every cell takes its nearest
anchor on the log scale.

Usage:  NSIM=5 python experiments/simulation/study_c_concurvity_benchmark.py [sweep ...]
"""
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cocodeel.model import BaseNetwork
from cocodeel.refit_model import RefitCovarNetwork
from cocodeel.crossfit import CrossFitEnsemble
from cocodeel.benchmarking.model import CovarNetwork, MLPCovarNetwork
from cocodeel.benchmarking.posthoc_model import (
    FeatureErasureNetwork, PostHocOrthNetwork, SemiStructuredNetwork,
)
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
RUN_DIR = ROOT / "experiments/simulation/output/runs/study_c"

N_GRID = [400, 800, 1600, 3200, 6400, 12800, 25600]
N_GRID_CONCURVITY = N_GRID + [51200, 102400]
Q_GRID = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

SIM_DEFAULTS = dict(bz=1., b2=1., b3=1., cv1=0.5, cv2=0.5, sdy=1.)

# selected by hpsearch/search_study_c.py (chosen_hps_study_c.json); the
# penalized NAM variants inherit the unpenalized nam_mlp winner; the
# penalty strength lam by hpsearch/search_study_c_lam.py
# (chosen_hps_study_c_lam.json)
HP_PER_N = {
    400: dict(backbone=dict(lr=1e-3, wd=1e-4), nam=dict(lr=1e-2, wd=1e-4),
              nam_mlp=dict(lr=3e-3, wd=1e-4),
              nam_mlp_conc=dict(lam=10.0)),
    800: dict(backbone=dict(lr=1e-2, wd=1e-4), nam=dict(lr=3e-3, wd=1e-4),
              nam_mlp=dict(lr=1e-2, wd=1e-4),
              nam_mlp_conc=dict(lam=1.0)),
    1600: dict(backbone=dict(lr=1e-3, wd=1e-4), nam=dict(lr=1e-2, wd=1e-4),
               nam_mlp=dict(lr=1e-3, wd=1e-5),
               nam_mlp_conc=dict(lam=1.0)),
    3200: dict(backbone=dict(lr=3e-4, wd=1e-4), nam=dict(lr=1e-2, wd=1e-5),
               nam_mlp=dict(lr=1e-3, wd=1e-5),
               nam_mlp_conc=dict(lam=1.0)),
    6400: dict(backbone=dict(lr=3e-4, wd=1e-5), nam=dict(lr=1e-2, wd=1e-4),
               nam_mlp=dict(lr=3e-3, wd=1e-5),
               nam_mlp_conc=dict(lam=1.0)),
    12800: dict(backbone=dict(lr=3e-4, wd=1e-5), nam=dict(lr=1e-2, wd=1e-4),
                nam_mlp=dict(lr=3e-4, wd=1e-4),
                nam_mlp_conc=dict(lam=1.0)),
    25600: dict(backbone=dict(lr=3e-4, wd=1e-5), nam=dict(lr=3e-3, wd=1e-4),
                nam_mlp=dict(lr=3e-4, wd=1e-5),
                nam_mlp_conc=dict(lam=0.1)),
    51200: dict(backbone=dict(lr=3e-3, wd=1e-5), nam=dict(lr=3e-3, wd=1e-4),
                nam_mlp=dict(lr=3e-4, wd=1e-5),
                nam_mlp_conc=dict(lam=0.1)),
    102400: dict(backbone=dict(lr=3e-4, wd=1e-5), nam=dict(lr=1e-3, wd=1e-5),
                 nam_mlp=dict(lr=3e-4, wd=1e-4),
                 nam_mlp_conc=dict(lam=0.1)),
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


def fit_nam_mlp_conc(lam, full_tr, full_va, hp, model_params):
    """MLP NAM with the Siems et al. (2023) concurvity penalty of strength lam; other HPs from nam_mlp."""
    m = train_covar_with_concurvity_reg(
        MLPCovarNetwork, model_params, full_tr, full_va, lam_reg=lam,
        warmup_frac=WARMUP_FRAC, **trainer_params(hp["nam_mlp"]))
    return m.center_effects(full_tr)


class JointNetwork(nn.Module):
    """Joint DNN g(X, Z) = MLP([phi(X), standardised Z]), trained end to end as the UKBB Joint DNN."""

    def __init__(self, backbone, backbone_params, num_covariates, link, z_mean, z_std, hidden=64):
        super().__init__()
        self.num_covariates = num_covariates
        self.backbone = backbone(**backbone_params)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.out_features + num_covariates, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("z_mean", torch.as_tensor(z_mean, dtype=torch.float32))
        self.register_buffer("z_std", torch.as_tensor(z_std, dtype=torch.float32))

    def g(self, phi, z):
        return self.head(torch.cat([phi, (z - self.z_mean) / self.z_std], dim=1))

    def forward(self, x, z):
        return self.g(self.backbone(x), z)

    @torch.no_grad()
    def marginalize(self, loader, n_perm=10):
        """Store phi(x_i) and z_j of the training sample and the centring constant c = mean_{i,j} g(x_i, z_j)."""
        self.eval()
        device = self.z_mean.device
        phis, zs = [], []
        for b in loader:
            phis.append(self.backbone(b["X"].to(device)))
            zs.append(b["Z"].to(device))
        self.phi_ref, self.z_ref = torch.cat(phis), torch.cat(zs)
        # unbiased for the double mean from n_perm random pairings (x_i, z_pi(i)),
        # O(n) instead of the O(n^2) full product
        n = len(self.z_ref)
        self.c = torch.stack([self.g(self.phi_ref, self.z_ref[torch.randperm(n, device=device)]).mean()
                              for _ in range(n_perm)]).mean()
        return self

    def predict_fx(self, x, z=None, chunk=4096):
        """f_X(x) = mean_j g(x, z_j) - c over the training z_j."""
        phi = self.backbone(x)
        total = torch.zeros(len(phi), 1, device=phi.device)
        for zc in self.z_ref.split(chunk):
            total += self.g(phi.repeat_interleave(len(zc), 0), zc.repeat(len(phi), 1)).view(len(phi), len(zc)).sum(1, keepdim=True)
        return total / len(self.z_ref) - self.c

    def predict_fz(self, z, chunk=4096):
        """f_Z(z) = mean_i g(x_i, z) - c over the training x_i."""
        total = torch.zeros(len(z), 1, device=z.device)
        for pc in self.phi_ref.split(chunk):
            total += self.g(pc.repeat(len(z), 1), z.repeat_interleave(len(pc), 0)).view(len(z), len(pc)).sum(1, keepdim=True)
        return total / len(self.phi_ref) - self.c


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

    # hyperparameters
    q = setting.get("q", Q_DEFAULT)
    hp = nearest_hp_q(setting["n"], q) if sweep == "concurvity_q" else HP_PER_N[setting["n"]]
    model_params = dict(
        backbone=TrafficBackbone,
        backbone_params={"out_features": q},
        num_covariates=1,
        link="identity",
    )

    # backbones on the two halves, for the 2-fold cross-fit
    tp_backbone = trainer_params(hp["backbone"])
    models = {}
    base_A = covar_trainer(BaseNetwork, model_params, train_loader=hA_tr, val_loader=hA_va, **tp_backbone)
    base_A = base_A.center_effects(hA_tr)
    base_B = covar_trainer(BaseNetwork, model_params, train_loader=hB_tr, val_loader=hB_va, **tp_backbone)
    base_B = base_B.center_effects(hB_tr)

    # refit: each backbone's last layer is refit on the other half, then the two are averaged
    refit_variants = [("refit", False)] + ([("refit_orth", True)] if cfg["full_roster"] else [])
    for name, orth in refit_variants:
        m_AB = RefitCovarNetwork(base_A, num_covariates=1, orthogonalize=orth).to(device)
        m_AB = m_AB.fit(hB_tr, hB_va)
        m_BA = RefitCovarNetwork(base_B, num_covariates=1, orthogonalize=orth).to(device)
        m_BA = m_BA.fit(hA_tr, hA_va)
        models[name] = CrossFitEnsemble([m_AB, m_BA]).recenter(pooled)

    # NAMs, trained end to end on the full sample
    nam = covar_trainer(CovarNetwork, model_params, train_loader=full_tr, val_loader=full_va,
                        **trainer_params(hp["nam"]))
    models["nam"] = nam.center_effects(full_tr)
    nam_mlp = covar_trainer(MLPCovarNetwork, model_params, train_loader=full_tr, val_loader=full_va,
                            **trainer_params(hp["nam_mlp"]))
    models["nam_mlp"] = nam_mlp.center_effects(full_tr)

    if cfg["full_roster"]:
        # DNN without controls on the full sample, and the post-hoc methods on top of it
        base_full = covar_trainer(BaseNetwork, model_params, train_loader=full_tr, val_loader=full_va, **tp_backbone)
        base_full = base_full.center_effects(full_tr)
        ckpt_dir = RUN_DIR / sweep / "checkpoints" / key
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(base_full.state_dict(), ckpt_dir / f"base_full_seed={seed}.pt")
        models["base"] = base_full
        models["posthoc_web"] = PostHocOrthNetwork(base_full, num_covariates=1).to(device).fit(full_tr, full_va)

        # SSN wraps the linear NAM
        models["ssn"] = SemiStructuredNetwork(nam).to(device).fit(full_tr)

        # Siems et al. (2023) concurvity penalty on the MLP NAM, at its oracle strength for n
        models["nam_mlp_conc"] = fit_nam_mlp_conc(hp["nam_mlp_conc"]["lam"], full_tr, full_va, hp, model_params)

        # feature erasure on the full-sample backbone, then refit of its head
        for name, eraser in [("posthoc_web_features", "residualize"), ("leace", "leace")]:
            erased = FeatureErasureNetwork(base_full, num_covariates=1, eraser=eraser).to(device)
            models[name] = erased.fit(full_tr)

        # joint DNN g(X, Z), marginalised over Z
        Z_tr = full_tr.dataset.Z
        joint = covar_trainer(JointNetwork, dict(model_params, z_mean=Z_tr.mean(0), z_std=Z_tr.std(0)),
                              train_loader=full_tr, val_loader=full_va, **tp_backbone)
        models["joint"] = joint.marginalize(DataLoader(full_tr.dataset, batch_size=1024, shuffle=False))

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
        hp_per_n=HP_PER_N,
        hp_q_anchors={f"n={n}_q={q}": v for (n, q), v in HP_Q_ANCHORS.items()},
        warmup_frac=WARMUP_FRAC,
        sweeps={s: len(c["settings"]) for s, c in sweeps.items()},
    ))
    grid_runner.write_settings_csv(RUN_DIR, {s: c["settings"] for s, c in sweeps.items()},
                                   {s: c["sweep_key_fn"] for s, c in sweeps.items()})

    done = grid_runner.already_done(RUN_DIR)
    tasks = [dict(sweep=sweep, setting=setting, seed=seed)
             for sweep, cfg in sweeps.items()
             for setting in cfg["settings"]
             for seed in range(NSIM)
             if (sweep, cfg["sweep_key_fn"](setting), seed) not in done]
    print(f"{len(done)} sims cached, {len(tasks)} to run.", flush=True)
    grid_runner.run_grid(RUN_DIR, tasks, _worker, N_WORKERS)


if __name__ == "__main__":
    main(sys.argv[1:])
