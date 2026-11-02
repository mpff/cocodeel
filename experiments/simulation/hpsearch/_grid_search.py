"""Shared multiprocessing driver and CSV writer for the hyperparameter searches."""
import csv
import datetime
import time

import torch.multiprocessing as mp


def run_pool(tasks, fit_one, n_workers, describe):
    """Run fit_one over tasks in a spawn pool; print one `describe(res)` line per result."""
    t_start = time.time()
    print(f"[{datetime.datetime.now():%H:%M:%S}] launching {len(tasks)} tasks "
          f"on {n_workers} workers", flush=True)
    rows = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(fit_one, tasks), start=1):
            rows.append(res)
            print(f"[{datetime.datetime.now():%H:%M:%S}] {i}/{len(tasks)}  {describe(res)}",
                  flush=True)
    print(f"Total wall: {(time.time() - t_start) / 60:.1f} min", flush=True)
    return rows


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}", flush=True)


def read_csv(path):
    """Rows of an earlier write_csv, numbers parsed and empty fields dropped; [] if absent."""
    if not path.exists():
        return []

    def parse(v):
        for cast in (int, float):
            try:
                return cast(v)
            except ValueError:
                pass
        return v

    with path.open(newline="") as f:
        return [{k: parse(v) for k, v in r.items() if v != ""} for r in csv.DictReader(f)]
