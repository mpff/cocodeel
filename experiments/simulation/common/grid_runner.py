"""Resumable multiprocessing runner for (sweep, setting, seed) simulation grids."""
import csv
import datetime
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[3]


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def write_manifest(run_dir, extras):
    manifest = {
        "start": datetime.datetime.now().isoformat(),
        "host": os.uname().nodename,
        "git_commit": _git_commit(),
        **extras,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def write_settings_csv(run_dir, sweeps, sweep_key_fns):
    """One settings.csv per sweep: sweep_key plus every setting variable."""
    for sweep, settings in sweeps.items():
        outp = run_dir / sweep / "settings.csv"
        outp.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for s in settings for k in s})
        with outp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["sweep_key", *keys])
            w.writeheader()
            for s in settings:
                w.writerow({"sweep_key": sweep_key_fns[sweep](s), **s})


def already_done(run_dir):
    """Set of (sweep, sweep_key, seed) triples with a saved NPZ."""
    done = set()
    for npz in run_dir.glob("*/preds/*/seed=*.npz"):
        try:
            seed = int(npz.stem.split("=")[1])
        except ValueError:
            continue
        done.add((npz.parents[2].name, npz.parent.name, seed))
    return done


def run_grid(run_dir, tasks, worker, n_workers):
    """Run `worker` over `tasks` in a spawn pool; append per-task JSON lines to progress.log."""
    progress_path = run_dir / "progress.log"
    t_start = time.time()
    print(f"[{datetime.datetime.now():%H:%M:%S}] {len(tasks)} tasks on "
          f"{n_workers} workers.", flush=True)

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(worker, tasks), start=1):
            with progress_path.open("a") as f:
                f.write(json.dumps({"t": datetime.datetime.now().isoformat(), **res}) + "\n")
            tag = res.get("status", "?").upper()
            print(f"[{datetime.datetime.now():%H:%M:%S}] {i}/{len(tasks)} {tag}  "
                  f"{res.get('sweep', '?')}/{res.get('sweep_key', '?')} "
                  f"seed={res.get('seed', '?')} t={res.get('wall_s', 0.0):.1f}s"
                  + (f"  ERR: {res.get('error', '')}" if tag == "ERROR" else ""),
                  flush=True)

    print(f"[{datetime.datetime.now():%H:%M:%S}] Done in "
          f"{(time.time() - t_start) / 3600:.2f} h. Run: {run_dir}", flush=True)


def catch_errors(fn, task, sweep_key):
    """Run one task; convert an exception into an error record instead of killing the pool."""
    try:
        return fn(**task)
    except Exception as e:
        return {"sweep": task.get("sweep"), "sweep_key": sweep_key,
                "seed": task.get("seed"), "status": "error",
                "error": str(e), "traceback": traceback.format_exc()[-1500:]}
