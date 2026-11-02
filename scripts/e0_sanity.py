"""04 规定的 E0 健全性检查。不运行 E1–E8。"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONUTF8", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from joblib import Parallel, delayed

from noisyopinf.data import parameter_hash
from noisyopinf.estimators import fit
from noisyopinf.io_utils import runtime_facts, write_meta
from noisyopinf.metrics import ep_violation, err_O_rel
from noisyopinf.noise import add_white, sigma_from_rho
from noisyopinf.pod import estimate_sigma2, pod_basis, project
from noisyopinf.seeds import rng
from noisyopinf.systems import burgers, galerkin_burgers, kdv_burgers
from noisyopinf.systems.config import load_system, project_root

N_TRAIN = 10
RANK = 8
REL_LIMIT = 1.0e-8
FIELD_METHODS = ("ols", "mc", "mc_psd")
S1_METHODS = FIELD_METHODS + ("iv", "iv2")


def resolution(system: str) -> dict:
    """5 条训练初值上比较 (N, dt) 与 (2N, dt/2)。"""
    cfg = load_system(system)
    width, step = int(cfg["N"]), float(cfg["dt"])
    jobs = []
    for index in range(5):
        jobs.append(delayed(_simulate)(system, index, width, step))
        jobs.append(delayed(_simulate)(system, index, 2 * width, step / 2.0))
    done = Parallel(n_jobs=_workers(), backend="loky")(jobs)
    coarse, fine = done[0::2], done[1::2]
    worst = _max_rel_l2(coarse, fine)
    ratios = [float(np.linalg.norm(item[-1]) / np.linalg.norm(item[0])) for item in coarse]
    _, values = pod_basis(coarse, 1)
    energy = np.cumsum(values**2) / np.sum(values**2)
    ranks = [int(np.searchsorted(energy, level) + 1) for level in (0.99, 0.999, 0.9999)]
    limit = 1.0e-4 if system == "s2" else 1.0e-3
    payload = {
        "system": system, "N": width, "dt": step, "n_traj": 5,
        "resolution_max_rel_l2": worst, "resolution_limit": limit,
        "resolution_pass": bool(worst < limit),
        "norm_ratio_median": float(np.median(ratios)), "norm_ratios": ratios,
        "max_abs": max(float(np.max(np.abs(item))) for item in coarse),
        "pod_ranks_99_999_9999": ranks,
    }
    if system == "s3":
        payload["norm_ratio_limit"] = 0.2
        payload["norm_ratio_pass"] = bool(payload["norm_ratio_median"] > 0.2)
    return payload


def s1_true_operator() -> dict:
    """实际计算 O_true 的残差、常数项和能量违反。"""
    operator, residual, _ = galerkin_burgers.true_operator()
    points = _s1_points(operator.shape[1])
    violation = ep_violation(operator, points)
    residual_max = float(np.max(np.abs(residual)))
    constant_norm = float(np.linalg.norm(operator[-1]))
    return {
        "residual_max": residual_max, "residual_limit": 1.0e-11,
        "residual_pass": bool(residual_max < 1.0e-11),
        "constant_norm": constant_norm, "constant_limit": 1.0e-12,
        "constant_pass": bool(constant_norm < 1.0e-12),
        "ep_violation": violation, "ep_limit": 1.0e-12,
        "ep_pass": bool(violation < 1.0e-12),
    }


def zero_noise(system: str) -> dict:
    """rho=0 时，规定的估计器必须与 oracle 一致。"""
    if system == "s1":
        trajectories = [galerkin_burgers.rollout(index, 1)[0] for index in range(N_TRAIN)]
        methods = S1_METHODS
    else:
        fields = [_simulate(system, index, None, None) for index in range(N_TRAIN)]
        basis, _ = pod_basis(fields, RANK)
        trajectories = [project(field, basis) for field in fields]
        methods = FIELD_METHODS
    oracle = fit(trajectories, "oracle").O
    rows = [{"estimator": "oracle", "err_O_rel": 0.0, "limit": REL_LIMIT, "pass": True}]
    rows.extend(_compare(trajectories, oracle, method) for method in methods)
    rank = int(trajectories[0].shape[1])
    basis = "galerkin" if system == "s1" else "clean_training_pod"
    return {
        "system": system, "rho": 0.0, "n_train": N_TRAIN, "rank": rank,
        "basis": basis, "coordinates": "original", "sigma2": 0.0,
        "rows": rows, "pass": all(row["pass"] for row in rows),
    }


def s2_sigma(fields: list[np.ndarray]) -> dict:
    """报告三个 rho 的 trailing-median 估计，不另设通过阈值。"""
    rows = [_one_sigma(fields, rho, key) for rho, key in ((0.01, 1), (0.05, 5), (0.2, 20))]
    width = int(fields[0].shape[1])
    total = int(sum(field.shape[0] for field in fields))
    return {"n_train": len(fields), "N": width, "K_tot": total, "precondition": total >= 2 * width, "rows": rows}


def main() -> None:
    started = time.perf_counter()
    out = _output_dir()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    solver = _guard(_solver)
    truth = _guard(s1_true_operator)
    zeros = _guard(lambda: [zero_noise(name) for name in ("s1", "s2", "s3")])
    fields = _guard(lambda: [_simulate("s2", index, None, None) for index in range(N_TRAIN)])
    sigma = _guard(lambda: s2_sigma(fields)) if isinstance(fields, list) else _failed("s2 fields were not generated")
    passed = _accepted(solver, truth, zeros, sigma)
    payload = {"solver": solver, "s1_true_operator": truth, "zero_noise": zeros, "s2_sigma2": sigma, "pass": passed}
    _store(out, payload, time.perf_counter() - started)
    print("E0_PASS" if passed else "E0_FAIL")


def _output_dir() -> Path:
    if len(sys.argv) > 2:
        raise SystemExit("usage: e0_sanity.py [output_dir]")
    if len(sys.argv) == 1:
        return project_root() / "results" / "e0"
    given = Path(sys.argv[1])
    return given if given.is_absolute() else project_root() / given


def _solver() -> dict:
    s2, s3 = resolution("s2"), resolution("s3")
    s3["pod_rank_differs_from_s2"] = s2["pod_ranks_99_999_9999"][0] != s3["pod_ranks_99_999_9999"][0]
    ok = s2["resolution_pass"] and s3["resolution_pass"] and s3["norm_ratio_pass"]
    return {"s2": s2, "s3": s3, "pass": bool(ok)}


def _guard(fn):
    try:
        return fn()
    except Exception as exc:
        return _failed(f"{type(exc).__name__}: {exc}")


def _failed(message: str) -> dict:
    return {"pass": False, "error": message}


def _accepted(solver: dict, truth: dict, zeros, sigma: dict) -> bool:
    if not isinstance(zeros, list) or not solver.get("pass") or truth.get("error"):
        return False
    if not isinstance(sigma, dict) or sigma.get("error") or "rows" not in sigma:
        return False
    checks = (truth.get("residual_pass"), truth.get("constant_pass"), truth.get("ep_pass"))
    return bool(all(checks) and all(item.get("pass") for item in zeros))


def _simulate(system: str, index: int, width, step):
    if system == "s2":
        return burgers.simulate(index, 1, n=width, dt=step)
    return kdv_burgers.simulate(index, 1, n=width, dt=step)


def _workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def _max_rel_l2(coarse, fine) -> float:
    worst = 0.0
    for left, right in zip(coarse, fine, strict=True):
        sampled = right[:, ::2]
        if sampled.shape != left.shape:
            raise RuntimeError(f"shape {left.shape} vs {sampled.shape}")
        for row in range(left.shape[0]):
            denom = float(np.linalg.norm(sampled[row]))
            worst = max(worst, float(np.linalg.norm(left[row] - sampled[row]) / denom))
    return worst


def _s1_points(rank: int) -> np.ndarray:
    from noisyopinf.features import quad_index
    count = 5 * (rank + len(quad_index(rank)) + 1)
    return rng(1, 9).standard_normal(size=(count, rank))


def _compare(trajectories, oracle: np.ndarray, method: str) -> dict:
    variance = 0.0 if method in {"mc", "mc_psd", "iv", "iv2"} else None
    result = fit(trajectories, method, sigma2=variance)
    gap = err_O_rel(result.O, oracle)
    return {
        "estimator": method, "err_O_rel": gap, "limit": REL_LIMIT,
        "pass": bool(gap < REL_LIMIT), "n_clipped": result.diagnostics.get("n_clipped"),
    }


def _one_sigma(fields, rho: float, key: int) -> dict:
    sigma = sigma_from_rho("s2", rho, fields)
    noisy = [add_white(field, sigma, rng(0, 80, key, index)) for index, field in enumerate(fields)]
    estimated, diagnostics = estimate_sigma2(noisy, "trailing_median", fields)
    true = float(sigma) ** 2
    return {
        "rho": rho, "sigma2_true": true, "sigma2_estimated": float(estimated),
        "relative_error": abs(float(estimated) - true) / true,
        "N": diagnostics["N"], "K_tot": diagnostics["K_tot"], "beta": diagnostics["beta"],
        "clean_half_eigenvalue_over_sigma2": diagnostics["clean_half_eigenvalue_over_sigma2"],
        "seed_keys": f"0,80,{key},index",
    }


def _store(out, payload: dict, elapsed: float) -> None:
    root = project_root()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    status = subprocess.check_output(["git", "status", "--short"], cwd=root, text=True)
    facts = runtime_facts()
    facts.update({
        "git_head": head,
        "git_status": status,
        "elapsed_seconds": elapsed,
        "n_train": N_TRAIN,
        "rank_s2_s3": RANK,
        "config_hash": {name: parameter_hash(name) for name in ("s1", "s2", "s3")},
        "data_source": "simulate/rollout in this process; data/ cache was not read",
    })
    write_meta(out / "meta.json", facts)
    for name, body in (
        ("solver_acceptance.json", payload["solver"]),
        ("s1_true_operator.json", payload["s1_true_operator"]),
        ("zero_noise.json", {"systems": payload["zero_noise"]}),
        ("s2_sigma2.json", payload["s2_sigma2"]),
        ("summary.json", {"pass": payload["pass"], "elapsed_seconds": elapsed, "git_head": head}),
    ):
        write_meta(out / name, {"provenance": facts, "result": body})


if __name__ == "__main__":
    main()
