"""Exploratory measurement-noise sensitivity. Fixed rank 10, spacing one sample."""
import csv
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from study import CFG, ROOT, features, fit_geometry, pod_basis, rows, simulate, solve_triangular


def records():
    for system_id, system in enumerate(CFG["systems"]):
        for block in range(CFG["blocks"]):
            begin = CFG["first_index"] + block*CFG["train"]
            paths = Parallel(n_jobs=4)(delayed(simulate)(system, begin+i, 1) for i in range(CFG["train"]))
            basis, _ = pod_basis(paths, 16)
            clean = (np.array(paths)@basis)[:, :, :10]
            gen = np.random.default_rng([20261120, system_id, block])
            noise = gen.normal(size=clean.shape) * np.sqrt(np.mean(clean**2))
            for rho in (0, 0.001, 0.01, 0.05):
                x = clean + rho*noise
                z = features(rows(x, 10))
                y = x[:, 11:].reshape(-1, 10)
                u, tri = np.linalg.qr(z, mode="reduced")
                ols = solve_triangular(tri, u.T@y)
                e = y-z@ols
                lagged = []
                for m in (1, 2):
                    lagged.append(features(rows(x, 10, m))[:, :-1])
                    w = np.column_stack([*lagged, np.ones(len(z))])
                    _, shift, bound, rank = fit_geometry(u, tri, y, ols, e, w)
                    yield dict(system=system, block=block, rho=rho, lags=m, shift=shift,
                               bound=bound, w_rank=rank, w_cols=w.shape[1])


if __name__ == "__main__":
    path = ROOT/"results"/"lag_geometry_v1"/"noise.csv"
    with path.open("x", newline="") as handle:
        writer = None
        for row in records():
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
