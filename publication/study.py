"""Matched-window lag study. Run from the project root on hanqing-win."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import solve_triangular

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
from noisyopinf.features import features
from noisyopinf.pod import pod_basis
from noisyopinf.systems import burgers
from noisyopinf.systems.spectral import sine_initial

CFG = json.loads((HERE / "protocol.json").read_text())
NORM = np.linalg.norm


def allen_cahn(index, split, *, refinement=1):
    """Strang split exact diffusion and exact scalar reaction on the unit circle."""
    cfg = CFG["allen_cahn"]
    n = cfg["n"] * refinement
    substeps = cfg["substeps"] * refinement
    dt = CFG["dt_obs"] / substeps
    k = 2 * np.pi * np.fft.rfftfreq(n, 1 / n)
    half = np.exp(-0.5 * cfg["nu"] * dt * k**2)
    growth = np.exp(cfg["reaction"] * dt)
    field = sine_initial(n, 4, split, index, 0.5)
    frames = [field]
    for _ in range(CFG["steps"]):
        for _ in range(substeps):
            mid = np.fft.irfft(half * np.fft.rfft(field), n=n)
            reacted = growth * mid / np.sqrt(1 + np.expm1(2 * cfg["reaction"] * dt) * mid**2)
            field = np.fft.irfft(half * np.fft.rfft(reacted), n=n)
        frames.append(field)
    return np.stack(frames)


def simulate(system, index, split):
    if system == "burgers":
        return burgers.simulate(index, split)
    return allen_cahn(index, split)


def rows(states, start, lag=0):
    """(trajectory, time, r) -> current or lagged rows with fixed anchor times."""
    stop = states.shape[1] - 1
    return states[:, start-lag:stop-lag].reshape(-1, states.shape[-1])


def instrument_basis(instruments):
    """Thin SVD discards only columns below the matrix-size roundoff threshold."""
    left, values, _ = np.linalg.svd(instruments, full_matrices=False)
    rank = np.sum(values > values[0] * max(instruments.shape) * np.finfo(float).eps)
    return left[:, :rank]


def fit_geometry(u, triangular, y, ols, residual, instruments):
    w = instrument_basis(instruments)
    coupling = w.T @ u
    # Direct residualized update avoids cancellation when the two fits nearly coincide.
    correction, _, rank, cosines = np.linalg.lstsq(coupling, w.T @ residual, rcond=None)
    if rank != u.shape[1]:
        raise np.linalg.LinAlgError("projected design is rank deficient")
    delta = solve_triangular(triangular, correction)
    sine = NORM(u - w @ coupling, ord=2)
    tangent = sine / cosines[-1]
    return delta, NORM(correction) / NORM(residual), tangent, w.shape[1]


def measure(train, test, rank):
    start = CFG["start"]
    z = features(rows(train, start))
    y = train[:, start+1:].reshape(-1, rank)
    test_z = features(rows(test, start))
    test_y = test[:, start+1:].reshape(-1, rank)
    u, tri = np.linalg.qr(z, mode="reduced")
    if np.linalg.matrix_rank(tri) != tri.shape[0]:
        raise np.linalg.LinAlgError("current design is rank deficient")
    ols = solve_triangular(tri, u.T @ y)
    residual = y - z @ ols
    for spacing in CFG["spacings"]:
        past = []
        for m in CFG["lags"]:
            past.append(features(rows(train, start, spacing * m))[:, :-1])
            w = np.column_stack([*past, np.ones(len(z))])
            delta, shift, tangent, w_rank = fit_geometry(u, tri, y, ols, residual, w)
            yield dict(rank=rank, spacing=spacing, lags=m, n=len(z), p=z.shape[1],
                       w_cols=w.shape[1], w_rank=w_rank, cond_z=np.linalg.cond(tri),
                       closure=NORM(residual)/NORM(y), shift=shift, bound=tangent,
                       coefficient=NORM(delta)/NORM(ols),
                       test_gap=NORM(test_z @ delta)/NORM(test_y),
                       test_ols=NORM(test_y-test_z @ ols)/NORM(test_y),
                       test_iv=NORM(test_y-test_z @ (ols+delta))/NORM(test_y))


def block_rows(system, block):
    start = CFG["first_index"] + block * CFG["train"]
    def sample(count, split):
        return Parallel(n_jobs=4)(delayed(simulate)(system, start+i, split) for i in range(count))
    train, test = sample(CFG["train"], 1), sample(CFG["test"], 3)
    basis, _ = pod_basis(train, max(CFG["ranks"]))
    a, b = np.array(train) @ basis, np.array(test) @ basis
    for rank in CFG["ranks"]:
        for record in measure(a[:, :, :rank], b[:, :, :rank], rank):
            yield dict(system=system, block=block, **record)


def main():
    out = ROOT / "results" / "lag_geometry_v1"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "blocks.csv"
    t0 = time.monotonic()
    with path.open("x", newline="", encoding="utf8") as handle:
        writer = None
        for system in CFG["systems"]:
            for block in range(CFG["blocks"]):
                for row in block_rows(system, block):
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                handle.flush()
                print(system, block, round(time.monotonic()-t0, 1), flush=True)
    meta = dict(seconds=time.monotonic()-t0, python=platform.python_version(), numpy=np.__version__)
    for item in (path, HERE / "protocol.json", Path(__file__)):
        meta[item.name] = hashlib.sha256(item.read_bytes()).hexdigest()
    (out / "complete.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
