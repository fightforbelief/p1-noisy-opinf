"""Stage 1 数值验收。这不是正式 E0。"""

from __future__ import annotations

import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONUTF8", "1")

import numpy as np
from joblib import Parallel, delayed

from noisyopinf.pod import pod_basis
from noisyopinf.systems import burgers, galerkin_burgers, kdv_burgers
from noisyopinf.systems.config import load_system, project_root


def _simulate(system: str, index: int, n: int, dt: float):
    if system == "s2":
        return burgers.simulate(index, 1, n=n, dt=dt)
    return kdv_burgers.simulate(index, 1, n=n, dt=dt)


def _resolution(system: str) -> tuple[float, list[np.ndarray], dict]:
    cfg = load_system(system)
    n, dt = int(cfg["N"]), float(cfg["dt"])
    jobs = []
    for index in range(5):
        jobs.append(delayed(_simulate)(system, index, n, dt))
        jobs.append(delayed(_simulate)(system, index, 2 * n, dt / 2.0))
    done = Parallel(n_jobs=-1, backend="loky")(jobs)
    coarse = done[0::2]
    fine = done[1::2]
    worst = 0.0
    for left, right in zip(coarse, fine, strict=True):
        sampled = right[:, ::2]
        if sampled.shape != left.shape:
            raise RuntimeError(f"{system} shape {left.shape} vs {sampled.shape}")
        for row in range(left.shape[0]):
            denom = float(np.linalg.norm(sampled[row]))
            worst = max(worst, float(np.linalg.norm(left[row] - sampled[row]) / denom))
    ratios = [float(np.linalg.norm(item[-1]) / np.linalg.norm(item[0])) for item in coarse]
    peaks = [float(np.max(np.abs(item))) for item in coarse]
    _, values = pod_basis(coarse, 1)
    energy = np.cumsum(values**2)
    energy /= energy[-1]
    ranks = [int(np.searchsorted(energy, level) + 1) for level in (0.99, 0.999, 0.9999)]
    summary = {
        "resolution_max_rel_l2": worst,
        "norm_ratio_median": float(np.median(ratios)),
        "norm_ratios": ratios,
        "max_abs": max(peaks),
        "pod_ranks_99_999_9999": ranks,
        "N": n,
        "dt": dt,
    }
    return worst, coarse, summary


def _write(payload: dict) -> None:
    path = project_root() / "reports" / "stage1_acceptance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def main() -> None:
    _, residual, constant = galerkin_burgers.true_operator()
    payload = {
        "s1_residual_max": float(np.max(np.abs(residual))),
        "s1_constant_norm": float(np.linalg.norm(constant)),
        "note": "Stage 1 acceptance only. This is not a formal E0 pass.",
    }
    _write(payload)
    _, _, s2 = _resolution("s2")
    payload["s2"] = s2
    _write(payload)
    _, _, s3 = _resolution("s3")
    payload["s3"] = s3
    _write(payload)


if __name__ == "__main__":
    main()
