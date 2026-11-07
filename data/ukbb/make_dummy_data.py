"""Write random HDF5 files with the layout of the UKBB data the experiments in experiments/ukbb/ read.

Each file holds T1-weighted volumes in MNI152 space at 2 mm, X of shape [n, 91, 109, 91], and per subject
the binary outcome highalc, the age in years and the sex (0/1). The real training pool has n = 14617
subjects and the holdout pool n = 4505; access is through the UK Biobank (https://www.ukbiobank.ac.uk).
The dummy files only let the pipeline run end to end; their volumes are noise.

Usage:  python data/ukbb/make_dummy_data.py [--n-train 400] [--n-holdout 200]
"""
import argparse
from pathlib import Path

import h5py
import numpy as np

OUT = Path(__file__).resolve().parent
SHAPE = (91, 109, 91)


def write(path, n, rng):
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=rng.random((n, *SHAPE), dtype=np.float32), chunks=(1, *SHAPE))
        f.create_dataset("highalc", data=rng.integers(0, 2, n))
        f.create_dataset("age", data=rng.integers(45, 82, n))
        f.create_dataset("sex", data=rng.integers(0, 2, n))
    print(f"{path.name}: {n} subjects")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=400)
    parser.add_argument("--n-holdout", type=int, default=200)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    write(OUT / "t1mniz2-l-highalcl0u2-bingeauditl1u3-alcfreq-c-sex-age-n14617.h5", args.n_train, rng)
    write(OUT / "holdout-t1mniz2-l-highalcl0u2-bingeauditl1u3-alcfreq-c-sex-age-n4505.h5", args.n_holdout, rng)
