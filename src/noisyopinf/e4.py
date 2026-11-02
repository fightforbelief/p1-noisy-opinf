"""E4：noisy POD 上的秩与噪声权衡。六个秩共用同一次全阶噪声。"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import yaml

from noisyopinf.data import parameter_hash
from noisyopinf.estimators import fit
from noisyopinf.io_utils import RUN_COLUMNS
from noisyopinf.metrics import (
    ep_violation,
    err_A_rel,
    err_H_rel,
    err_O_rel,
    err_c_abs,
    err_c_scaled,
    horizon_medians,
    proj_floor,
    rollout_aggregate,
    rollout_err_h,
    specrad_trio,
    state_rms,
)
from noisyopinf.noise import add_white, sigma_from_rho
from noisyopinf.pod import estimate_sigma2, pod_basis, project
from noisyopinf.protocol import LAMBDAS, _validation_score, larger_lambda_index
from noisyopinf.rollout import rollout
from noisyopinf.seeds import rng

EXPERIMENT_ID = 4
SYSTEM_ID = 2
RANKS = (4, 6, 8, 10, 12, 16)
ESTIMATORS = ("ols", "ridge", "mc", "iv2")
SIGMA_METHODS = frozenset({"mc", "iv2"})
KAPPAS = (1, 4, 16, 64)
CRITERIA = ("val", "noisyval", "test")
PILOT = {"rho": 0.05, "replicates": (0, 1)}
TIE_RULE = "smaller_r"
RANK_AGGREGATE = "median_of_replicate_medians"
SNR_PLOT_AGGREGATE = "lower_central_median_of_nonempty"
REQUIRED = {
    "experiment": "E4",
    "system": "s2",
    "n_train": 30,
    "n_val": 20,
    "n_test": 50,
    "basis": "noisy",
    "r": list(RANKS),
    "r_max": 16,
    "rho": [0.01, 0.02, 0.05, 0.1, 0.2],
    "estimators": list(ESTIMATORS),
    "sigma2_source": "est_trailing",
    "replicates": 10,
    "experiment_id": 4,
    "kappa": list(KAPPAS),
    "validation_offset": 100,
    "tie_rule": TIE_RULE,
    "rank_aggregate": RANK_AGGREGATE,
    "snr_plot_aggregate": SNR_PLOT_AGGREGATE,
}


def load_config(path: Path | None = None) -> dict:
    """读取并核对 E4 网格。实验编号固定为 4。"""
    from noisyopinf.systems.config import project_root

    source = project_root() / "configs" / "e4.yaml" if path is None else Path(path)
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    for key, value in REQUIRED.items():
        if loaded.get(key) != value:
            raise ValueError(f"e4.yaml {key} must be {value}")
    if list(loaded.get("rho_grid", [])) != [0, 0.01, 0.02, 0.05, 0.1, 0.2]:
        raise ValueError("e4.yaml rho_grid must be the master grid")
    return loaded


def planned_runs(config: dict | None = None) -> int:
    """5 × 6 × 4 × 10 = 1200。Ridge 的 13 个 λ 不计入。"""
    loaded = load_config() if config is None else config
    return len(loaded["rho"]) * len(loaded["r"]) * len(loaded["estimators"]) * int(loaded["replicates"])


def noise_keys(rho_index_value: int, replicate: int, traj_index: int, validation: bool) -> tuple[int, ...]:
    """03 的噪声键。E4 的 experiment_id 是 4。"""
    offset = 100 if validation else 0
    return (SYSTEM_ID, 4, EXPERIMENT_ID, int(rho_index_value), int(replicate), int(traj_index) + offset)


def add_full_order_noise(clean, sigma: float, rho_index_value: int, replicate: int, validation: bool):
    """投影前给全阶轨迹加白噪声。同一重复的六个秩共用这一份。"""
    return [
        add_white(traj, sigma, rng(*noise_keys(rho_index_value, replicate, index, validation)))
        for index, traj in enumerate(clean)
    ]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def parameter_digest() -> str:
    return parameter_hash("s2")


def mode_energies(clean_coordinates: np.ndarray) -> np.ndarray:
    """λ_i = mean q_{k,i}^2。坐标按列，不中心化。"""
    states = np.asarray(clean_coordinates, dtype=np.float64)
    if states.ndim != 2:
        raise ValueError("coordinates must have shape (samples, r)")
    return np.mean(states * states, axis=0)


def snr_rank(energies: np.ndarray, sigma2: float, kappa: float) -> tuple[int | None, list[int]]:
    """最大满足 λ_i ≥ κ σ² 的 1-based 下标。空集返回 None，不改成 4。"""
    variance = float(sigma2)
    level = float(kappa)
    passed = [index + 1 for index, value in enumerate(np.asarray(energies, dtype=np.float64)) if np.isfinite(value) and np.isfinite(variance) and value >= level * variance]
    if not passed:
        return None, []
    return max(passed), passed


def aggregate_snr(values: list[int | None]) -> tuple[int | None, int]:
    """图上的 r_snr：非空重复的中位数；偶数个时取较低的中心值。"""
    present = sorted(int(value) for value in values if value is not None)
    empty = len(values) - len(present)
    if not present:
        return None, empty
    count = len(present)
    if count % 2 == 1:
        return present[count // 2], empty
    return present[count // 2 - 1], empty


def choose_rank(scores: dict[int, float]) -> tuple[int, float, int]:
    """聚合分数最小的秩。非有限值劣于有限值，NaN 劣于 inf。并列取较小的 r。"""
    if not scores:
        raise ValueError("rank scores are empty")

    def sort_key(item: tuple[int, float]) -> tuple:
        rank, score = item
        if np.isnan(score):
            return (2, 0.0, int(rank))
        if not np.isfinite(score):
            return (1, 0.0, int(rank))
        return (0, float(score), int(rank))

    ordered = sorted(scores.items(), key=sort_key)
    best_rank, best_score = ordered[0]
    tied = [int(rank) for rank, score in scores.items() if _same_score(score, best_score)]
    return min(tied), float(best_score), len(tied)


def _same_score(left: float, right: float) -> bool:
    if np.isnan(left) and np.isnan(right):
        return True
    if np.isposinf(left) and np.isposinf(right):
        return True
    if not np.isfinite(left) or not np.isfinite(right):
        return False
    return float(left) == float(right)


def aggregate_median(values) -> float:
    """跨重复的中位数。任一 NaN 则结果为 NaN，inf 保持 inf。"""
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0 or np.isnan(data).any():
        return float("nan")
    return float(np.median(data))


def select_from_curves(curves: list[dict]) -> list[dict]:
    """三种准则分别选秩。测试准则只描述，不回写验证选择。"""
    grouped: dict[tuple, dict[int, list[float]]] = {}
    for row in curves:
        key = (float(row["rho"]), row["estimator"])
        slot = grouped.setdefault(key, {name: {} for name in CRITERIA})
        for name, field in (("val", "clean_val_med"), ("noisyval", "noisy_val_med"), ("test", "test_med")):
            slot[name].setdefault(int(row["r"]), []).append(float(row[field]))
    chosen = []
    for (rho, estimator), criteria in sorted(grouped.items(), key=lambda item: (item[0][0], ESTIMATORS.index(item[0][1]))):
        for name in CRITERIA:
            scores = {rank: aggregate_median(values) for rank, values in criteria[name].items()}
            rank, score, ties = choose_rank(scores)
            chosen.append({
                "rho": rho,
                "estimator": estimator,
                "criterion": name,
                "r_selected": rank,
                "selection_score": score,
                "n_tied": ties,
            })
    return chosen


def evaluate_replicate(clean_train, clean_val, clean_test, rho: float, rho_index_value: int, replicate: int) -> dict:
    """一次重复的六个秩。噪声、σ² 和 16 维基只生成一次。"""
    sigma = float(sigma_from_rho("s2", float(rho), clean_train))
    sigma2_true = float(sigma * sigma)
    noisy_train = add_full_order_noise(clean_train, sigma, rho_index_value, replicate, False)
    noisy_val = add_full_order_noise(clean_val, sigma, rho_index_value, replicate, True)
    estimate = _one_variance(noisy_train, clean_train, sigma2_true)
    basis, singular = pod_basis(noisy_train, max(RANKS))
    gap = _truncation_gap(noisy_train, basis)
    clean_full = np.vstack([project(traj, basis) for traj in clean_train])
    energies = mode_energies(clean_full)
    ranks = {}
    for rank in RANKS:
        ranks[rank] = _one_rank(
            rank, basis[:, :rank], clean_train, clean_val, clean_test, noisy_train, noisy_val,
            rho, rho_index_value, replicate, sigma2_true, estimate,
        )
    return {
        "rho": float(rho),
        "rho_index": int(rho_index_value),
        "replicate": int(replicate),
        "sigma2_true": sigma2_true,
        "sigma": sigma,
        "estimate": estimate,
        "singular_values": np.asarray(singular, dtype=np.float64),
        "basis": np.asarray(basis, dtype=np.float64),
        "basis_hash": array_sha256(basis),
        "truncation_gap": gap,
        "energies": energies,
        "noise_hash": array_sha256(np.vstack(noisy_train)),
        "full_order_noise_rms": float(np.sqrt(np.mean((np.vstack(noisy_train) - np.vstack(clean_train)) ** 2))),
        "ranks": ranks,
    }


def _one_variance(noisy_train, clean_train, sigma2_true: float) -> dict:
    try:
        value, diagnostics = estimate_sigma2(noisy_train, "trailing_median", clean_train)
        error = ""
    except (ValueError, np.linalg.LinAlgError) as exc:
        value, diagnostics, error = float("nan"), {}, str(exc)
    true = float(sigma2_true)
    if not np.isfinite(value):
        relative = float("nan")
    elif true == 0.0:
        relative = 0.0 if float(value) == 0.0 else float("nan")
    else:
        relative = float((float(value) - true) / true)
    return {
        "value": float(value),
        "relative": relative,
        "error": error,
        "diagnostics": diagnostics,
        "applicable": bool(diagnostics) and error == "",
    }


def _truncation_gap(noisy_train, basis: np.ndarray) -> float:
    """16 维 SVD 的前 r 列与单独做 r 维截断的差。"""
    gap = 0.0
    for rank in RANKS:
        fresh, _values = pod_basis(noisy_train, rank)
        gap = max(gap, float(np.max(np.abs(fresh - basis[:, :rank]))))
    return gap


def _one_rank(rank, basis, clean_train, clean_val, clean_test, noisy_train, noisy_val, rho, rho_index_value, replicate, sigma2_true, estimate) -> dict:
    clean_q = [project(traj, basis) for traj in clean_train]
    noisy_q = [project(traj, basis) for traj in noisy_train]
    noisy_val_q = [project(traj, basis) for traj in noisy_val]
    limit = float(1000.0 * max(float(np.max(np.abs(traj))) for traj in clean_q))
    reference = fit(clean_q, "ols").O
    ridge = _ridge(noisy_q, noisy_val_q, limit)
    floor = float(proj_floor(basis, np.vstack(clean_test)))
    rows, operators, details, validation = [], {}, [], []
    for method in ESTIMATORS:
        row, operator, trajectory_rows, validation_rows = _one_estimator(
            method, clean_q, noisy_q, clean_val, noisy_val, clean_test, basis, reference, floor, limit,
            rho, rho_index_value, replicate, rank, sigma2_true, estimate, ridge["lam"],
        )
        rows.append(row)
        operators[_operator_key(rho_index_value, rank, method, replicate)] = operator
        details.extend(trajectory_rows)
        validation.extend(validation_rows)
    operators[_reference_key(rho_index_value, rank, replicate)] = np.asarray(reference, dtype=np.float64)
    return {
        "rows": rows,
        "operators": operators,
        "details": details,
        "validation": validation,
        "ridge": ridge,
        "threshold": limit,
        "reference_hash": array_sha256(reference),
        "basis_hash": array_sha256(basis),
        "orthogonality": float(np.linalg.norm(basis.T @ basis - np.eye(rank))),
        "proj_floor": floor,
        "clean_q_rms": float(state_rms(np.vstack(clean_q))),
    }


def _ridge(noisy_train_q, noisy_val_q, limit: float) -> dict:
    scores = [float(_validation_score(noisy_train_q, noisy_val_q, lam, limit)) for lam in LAMBDAS]
    chosen = larger_lambda_index(scores)
    best = scores[chosen]
    return {
        "lam": float(LAMBDAS[chosen]),
        "scores": scores,
        "n_tied": int(sum(score == best for score in scores)),
        "n_nonfinite": int(sum(not np.isfinite(score) for score in scores)),
    }


def _one_estimator(method, clean_q, noisy_q, clean_val, noisy_val, clean_test, basis, reference, floor, limit, rho, rho_index_value, replicate, rank, sigma2_true, estimate, lam):
    source = "est_trailing" if method in SIGMA_METHODS else "na"
    variance = None if source == "na" else estimate["value"]
    try:
        if source != "na" and (variance is None or not np.isfinite(variance)):
            raise ValueError("sigma2 unavailable: est_trailing")
        result = fit(noisy_q, method, sigma2=None if variance is None else float(variance), lam=lam if method == "ridge" else None)
        operator = result.O
        n_pairs = int(result.n_pairs)
        seconds = float(result.fit_seconds)
        diagnostics = result.diagnostics
        error = ""
    except ValueError as exc:
        operator = np.full_like(reference, np.nan)
        n_pairs = 0
        seconds = 0.0
        diagnostics = {}
        error = str(exc)
        variance = None
    row = {column: None for column in RUN_COLUMNS}
    row.update({
        "exp": "e4", "system": "s2", "rho": float(rho), "sigma2_true": float(sigma2_true),
        "sigma2_used": None if variance is None else float(variance),
        "sigma2_source": source, "r": int(rank), "n_train": len(noisy_q),
        "n_pairs": n_pairs, "basis": "noisy", "estimator": method,
        "lam": float(lam) if method == "ridge" else None,
        "replicate": int(replicate),
        "seed_keys": f"2,4,4,{int(rho_index_value)},{int(replicate)},traj|val+100",
        "fit_seconds": seconds,
        "gram_min_eig": diagnostics.get("gram_min_eig"),
        "gram_cond": diagnostics.get("gram_cond"),
        "n_clipped": diagnostics.get("n_clipped"),
        "proj_floor": float(floor),
        "status": "ok",
        "error": error,
    })
    if error or not np.isfinite(operator).all():
        row["status"] = "failed"
        row["error"] = error or str(diagnostics.get("error", "non-finite operator"))
        for name in ("err_O_rel", "err_A_rel", "err_H_rel", "err_c_abs", "err_c_scaled", "specrad_hat", "specrad_ref", "specrad_err", "ep_violation", "ep_violation_ref", "rollout_med", "rollout_med_h10", "rollout_med_h50", "rollout_mean_finite", "diverged_frac"):
            row[name] = float("nan")
        row["clean_val_med"] = float("nan")
        row["noisy_val_med"] = float("nan")
        details = [_empty_detail(rho, rank, method, replicate, index, limit, floor) for index in range(len(clean_test))]
        validation = _empty_validation(rho, rank, method, replicate, len(clean_val), len(noisy_val))
        return row, operator, details, validation
    _fill_test(row, operator, reference, clean_q, clean_test, basis, limit, floor, rho, rank, method, replicate)
    clean_errors = _score_group(operator, clean_val, basis, limit)
    noisy_errors = _score_group(operator, noisy_val, basis, limit)
    row["clean_val_med"] = aggregate_median(clean_errors)
    row["noisy_val_med"] = aggregate_median(noisy_errors)
    validation = []
    for kind, errors in (("clean", clean_errors), ("noisy", noisy_errors)):
        for index, value in enumerate(errors):
            validation.append({
                "rho": float(rho), "r": int(rank), "estimator": method, "replicate": int(replicate),
                "kind": kind, "traj": int(index), "err": float(value), "diverged": bool(np.isposinf(value)),
            })
    return row, operator, row.pop("_details"), validation


def _fill_test(row, operator, reference, clean_q, clean_test, basis, limit, floor, rho, rank, method, replicate) -> None:
    comparable = np.isfinite(np.asarray(reference)).all() and np.isfinite(np.asarray(operator)).all()
    if comparable:
        row["err_O_rel"] = err_O_rel(operator, reference)
        row["err_A_rel"] = err_A_rel(operator, reference)
        row["err_H_rel"] = err_H_rel(operator, reference)
        row["err_c_abs"] = err_c_abs(operator, reference)
        row["err_c_scaled"] = err_c_scaled(operator, reference, state_rms(np.vstack(clean_q)))
        row["specrad_hat"], row["specrad_ref"], row["specrad_err"] = specrad_trio(operator, reference)
        snapshots = np.vstack([project(traj, basis) for traj in clean_test])
        row["ep_violation"] = ep_violation(operator, snapshots)
        row["ep_violation_ref"] = ep_violation(reference, snapshots)
    else:
        for name in ("err_O_rel", "err_A_rel", "err_H_rel", "err_c_abs", "err_c_scaled", "specrad_hat", "specrad_ref", "specrad_err", "ep_violation", "ep_violation_ref"):
            row[name] = float("nan")
        row["status"] = "failed"
        row["error"] = row["error"] or "non-finite reference"
    errors, flags, early, details = [], [], {10: [], 50: []}, []
    for index, traj in enumerate(clean_test):
        reduced = project(traj, basis)
        steps = int(traj.shape[0] - 1)
        predicted, diverged = rollout(operator, reduced[0], steps, limit)
        full = float("inf") if diverged else rollout_err_h(predicted, traj, steps, basis)
        short = {horizon: rollout_err_h(predicted, traj, horizon, basis) for horizon in (10, 50)}
        errors.append(full)
        flags.append(bool(diverged))
        for horizon in (10, 50):
            early[horizon].append(short[horizon])
        details.append({
            "rho": float(rho), "r": int(rank), "estimator": method, "replicate": int(replicate), "traj": int(index),
            "err_full": full, "err_h10": short[10], "err_h50": short[50], "diverged": bool(diverged),
            "first_step": _first_step(predicted, limit, bool(diverged)),
            "threshold": float(limit), "proj_floor": float(floor),
        })
    row.update(rollout_aggregate(errors, flags))
    row.update(horizon_medians(early))
    row["_details"] = details


def _score_group(operator, trajectories, basis, limit: float) -> list[float]:
    errors = []
    for traj in trajectories:
        reduced = project(traj, basis)
        steps = int(traj.shape[0] - 1)
        predicted, diverged = rollout(operator, reduced[0], steps, limit)
        errors.append(float("inf") if diverged else rollout_err_h(predicted, traj, steps, basis))
    return errors


def _empty_detail(rho, rank, method, replicate, index, limit, floor) -> dict:
    return {
        "rho": float(rho), "r": int(rank), "estimator": method, "replicate": int(replicate), "traj": int(index),
        "err_full": float("nan"), "err_h10": float("nan"), "err_h50": float("nan"), "diverged": False,
        "first_step": -1, "threshold": float(limit), "proj_floor": float(floor),
    }


def _empty_validation(rho, rank, method, replicate, n_clean: int, n_noisy: int) -> list[dict]:
    rows = []
    for kind, count in (("clean", n_clean), ("noisy", n_noisy)):
        for index in range(count):
            rows.append({
                "rho": float(rho), "r": int(rank), "estimator": method, "replicate": int(replicate),
                "kind": kind, "traj": int(index), "err": float("nan"), "diverged": False,
            })
    return rows


def _first_step(predicted: np.ndarray, limit: float, diverged: bool) -> int:
    if not diverged:
        return -1
    states = np.asarray(predicted, dtype=np.float64)
    finite = np.isfinite(states).all(axis=1)
    if (not finite[0]) or float(np.max(np.abs(states[0]))) > float(limit):
        return 0
    missing = np.where(~finite)[0]
    if missing.size == 0:
        return 0
    first_nan = int(missing[0])
    previous = states[first_nan - 1]
    if np.isfinite(previous).all() and float(np.max(np.abs(previous))) > float(limit):
        return first_nan - 1
    return first_nan


def _operator_key(rho_index_value: int, rank: int, method: str, replicate: int) -> str:
    return f"est_rho{int(rho_index_value)}_r{int(rank):02d}_{method}_rep{int(replicate):03d}"


def _reference_key(rho_index_value: int, rank: int, replicate: int) -> str:
    return f"ref_rho{int(rho_index_value)}_r{int(rank):02d}_rep{int(replicate):03d}"


def operator_key(rho_index_value: int, rank: int, method: str, replicate: int) -> str:
    return _operator_key(rho_index_value, rank, method, replicate)


def reference_key(rho_index_value: int, rank: int, replicate: int) -> str:
    return _reference_key(rho_index_value, rank, replicate)
