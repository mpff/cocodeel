"""Live experiment-tracking dashboard for the concurvity exploration sweep.

Re-scans the run directory on every request, re-aggregates all per-(n, seed)
NPZ prediction files present so far, and serves an auto-refreshing HTML page
with the bias^2 / variance / MSPE-vs-N figure for the f_x (image) effect.

The bias/variance decomposition replicates `aggregate_full_simulation.py`
exactly (per test point i, method m, S seeds):

    bias2_i = (mean_s yhat_{i,s} - truth_i)^2
    var_i   = var_s yhat_{i,s}                 (population var, ddof=0)
    mspe_i  = mean_s (yhat_{i,s} - truth_i)^2  = bias2_i + var_i

then reported as mean_i(...) per (method, n). Only the f_x effect is used.

The figure mirrors `4-Figure2_concurvity.R`: quantity vs N_train on a log10
x-axis (N_train = n/2, the train size of the `full` split; exact for the
end-to-end methods), log10 y-axis, one line+point per method, theme_bw-style
panels. Methods: sgd, nam, the three nam_ridge_<λ> sweep points, posthoc,
posthoc_xfit (see run_concurvity_exploration.py).

Usage:
    python dashboard.py --check                  # render once, no server
    python dashboard.py [--port 8765] [--host 127.0.0.1]
"""
from __future__ import annotations

import argparse
import base64
import datetime
import glob
import io
import json
import socketserver
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS_GLOB = (
    "/home/RDC/pfeuffma/Research/ovb-ddns/code/results/exploration/runs/"
    "*_concurvity_*nsim*"
)
EFFECT = "fx"

# Method draw order + display labels. The NAM family (end-to-end MLP-fz NAM
# and its AdamW ridge sweep) is plotted alongside the two post-hoc methods,
# which sit apart to mark the "DNN with Controls" approach.
METHOD_LABELS = {
    "sgd": "CovarNetwork (linear f_z)",
    "nam": "NAM (lambda=0)",
    "nam_ridge_0.001": "NAM + ridge (lambda=0.001)",
    "nam_ridge_0.1": "NAM + ridge (lambda=0.1)",
    "nam_ridge_100": "NAM + ridge (lambda=100)",
    "posthoc": "DNN with Controls",
    "posthoc_xfit": "DNN with Controls (xfit)",
}
METHOD_ORDER = list(METHOD_LABELS)

# The three ridge strengths share a light->dark blue ramp so they read as one
# sweep; nam (lambda=0) keeps yellow-green, sgd grey, and the two post-hoc
# methods keep the coral/orange family marking the different approach.
METHOD_COLORS = {
    "sgd": "#7F7F7F",              # grey — linear-fz baseline
    "nam": "#7AD151",              # yellow-green — NAM, lambda=0
    "nam_ridge_0.001": "#9ECAE1",  # light blue — weakest ridge
    "nam_ridge_0.1": "#4292C6",    # medium blue
    "nam_ridge_100": "#08519C",    # dark blue — strongest ridge
    "posthoc": "#D1426F",          # coral-magenta — off-ramp highlight
    "posthoc_xfit": "#F08F4A",     # warm orange — second post-hoc
}

METRICS = ["bias2", "var", "mspe"]
METRIC_LABELS = {
    "bias2": r"Bias$^2(\hat{f}_X)$",
    "var": r"Var$(\hat{f}_X)$",
    "mspe": r"MSPE$(\hat{f}_X)$",
}


def resolve_run_dir(run_dir: str | None) -> Path:
    if run_dir:
        return Path(run_dir)
    candidates = sorted(
        glob.glob(RUNS_GLOB), key=lambda c: Path(c).stat().st_mtime, reverse=True
    )
    if not candidates:
        raise FileNotFoundError(f"No run dirs match {RUNS_GLOB}")
    return Path(candidates[0])


def scan(run_dir: Path) -> dict[int, list[Path]]:
    """Map n -> sorted list of per-seed NPZ files."""
    out: dict[int, list[Path]] = {}
    for n_dir in sorted(run_dir.glob("n=*")):
        try:
            n = int(n_dir.name.split("=", 1)[1])
        except ValueError:
            continue
        out[n] = sorted(n_dir.glob("seed=*.npz"))
    return out


def aggregate(run_dir: Path) -> tuple[dict, dict[int, int], int]:
    """Re-aggregate every NPZ present into per-(method, metric, n) values.

    Returns (records, counts, n_skipped). `records` maps (method, metric) ->
    {n: value}, restricted to the plotted METHOD_ORDER. Each NPZ supplies the
    shared truth and all methods. Files that fail to load (mid-write) are
    skipped and succeed on a later refresh. `counts` is the per-n loaded sim
    count.
    """
    files_by_n = scan(run_dir)
    records: dict[tuple[str, str], dict[int, float]] = {}
    counts: dict[int, int] = {}
    n_skipped = 0

    for n, paths in files_by_n.items():
        # Per (method): stack seed predictions [S, T]; truth__fx is shared.
        stacks: dict[str, list[np.ndarray]] = {}
        truth = None
        n_loaded = 0
        for p in paths:
            try:
                d = np.load(p, allow_pickle=True)
                methods = [str(m) for m in d["methods"]]
                t = d[f"truth__{EFFECT}"]
                preds = {
                    m: d[f"{m}__{EFFECT}"] for m in methods if m in METHOD_ORDER
                }
            except Exception:
                n_skipped += 1
                continue
            if truth is None:
                truth = t
            n_loaded += 1
            for m, arr in preds.items():
                stacks.setdefault(m, []).append(arr)

        counts[n] = n_loaded
        if n_loaded == 0 or truth is None:
            continue

        for m, seed_list in stacks.items():
            preds = np.stack(seed_list, axis=0)            # [S, T]
            mean_pred = preds.mean(axis=0)                 # [T]
            bias2 = (mean_pred - truth) ** 2               # [T]
            var = preds.var(axis=0, ddof=0)                # [T]  population var
            mspe = ((preds - truth[None, :]) ** 2).mean(0) # [T]
            vals = {
                "bias2": float(bias2.mean()),
                "var": float(var.mean()),
                "mspe": float(mspe.mean()),
            }
            for metric, v in vals.items():
                records.setdefault((m, metric), {})[n] = v

    return records, counts, n_skipped


def render_figure(records: dict, run_dir: Path) -> bytes:
    """Render the bias^2 / var / MSPE-vs-N_train decomposition to PNG bytes."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharex=True)

    for ax, metric in zip(axes, METRICS):
        for m in METHOD_ORDER:
            series = records.get((m, metric))
            if not series:
                continue
            ns = sorted(series)
            n_train = [n * 0.5 for n in ns]   # N_train = n/2 (sample split)
            y = [series[n] for n in ns]
            ax.plot(
                n_train, y, marker="o", markersize=3.5, linewidth=1.3,
                alpha=0.85, color=METHOD_COLORS[m], label=METHOD_LABELS[m],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$N_{train}$ ($\log_{10}$ scale)", fontsize=9)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, which="major", linewidth=0.4, alpha=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(labels),
        fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        f"Concurvity exploration — image effect $f_X$   ({run_dir.name})",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def status_html(counts: dict[int, int], n_skipped: int) -> str:
    ns = sorted(counts)
    total = sum(counts.values())
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cells = "".join(
        f"<td style='padding:2px 10px'>n={n}<br><b>{counts[n]}</b> sims</td>"
        for n in ns
    )
    skip = (
        f" &nbsp;|&nbsp; skipped this scan (mid-write): {n_skipped}"
        if n_skipped else ""
    )
    return (
        f"<p style='font-family:monospace;font-size:13px'>"
        f"loaded: <b>{total}</b> sims &nbsp;|&nbsp; last updated {now}{skip}</p>"
        f"<table style='font-family:monospace;font-size:12px;border-collapse:collapse'>"
        f"<tr>{cells}</tr></table>"
    )


def build_page(run_dir: Path) -> str:
    records, counts, n_skipped = aggregate(run_dir)
    png = render_figure(records, run_dir)
    b64 = base64.b64encode(png).decode("ascii")
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="20">
<title>Concurvity exploration dashboard</title></head>
<body style="font-family:sans-serif;margin:24px">
<h2 style="margin-bottom:4px">Concurvity exploration — live</h2>
<p style="color:#666;font-size:12px;margin-top:0">{run_dir}</p>
{status_html(counts, n_skipped)}
<img src="data:image/png;base64,{b64}" style="max-width:100%;margin-top:12px">
</body></html>"""


def make_handler(run_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/figure.png"):
                records, _, _ = aggregate(run_dir)
                body = render_figure(records, run_dir)
                ctype = "image/png"
            else:
                body = build_page(run_dir).encode("utf-8")
                ctype = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # quiet stdout; nohup log stays readable
            pass

    return Handler


def serve(run_dir: Path, host: str, port: int, max_tries: int = 10) -> None:
    handler = make_handler(run_dir)
    for candidate in range(port, port + max_tries):
        try:
            httpd = socketserver.TCPServer((host, candidate), handler)
        except OSError:
            print(f"[dashboard] port {candidate} busy, trying next", flush=True)
            continue
        print(
            f"[dashboard] serving {run_dir.name} at http://{host}:{candidate}",
            flush=True,
        )
        httpd.serve_forever()
        return
    raise SystemExit(f"No free port in [{port}, {port + max_tries})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--check", action="store_true",
        help="render the page once and exit (smoke test, no server)",
    )
    args = ap.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    if args.check:
        records, counts, n_skipped = aggregate(run_dir)
        png = render_figure(records, run_dir)
        methods_seen = sorted({m for m, _ in records})
        print(
            f"[check] run_dir={run_dir}\n"
            f"[check] counts={dict(sorted(counts.items()))} skipped={n_skipped}\n"
            f"[check] methods plotted={methods_seen}\n"
            f"[check] series={len(records)} (method,metric) pairs, "
            f"PNG={len(png)} bytes",
            flush=True,
        )
        sys.exit(0)

    serve(run_dir, args.host, args.port)


if __name__ == "__main__":
    main()
