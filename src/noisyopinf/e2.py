"""E2：S1 上七个估计器相对 O_true 的配对比较。"""

from __future__ import annotations

import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
import yaml
from joblib import Parallel, delayed

from noisyopinf.e1 import rho_index, sigma2_of
from noisyopinf.estimators import fit
from noisyopinf.io_utils import RUN_COLUMNS
from noisyopinf.metrics import (
    bias_rel,
    block_bias_sd,
    bootstrap_interval,
    ep_violation,
    err_A_rel,
    err_H_rel,
    err_O_rel,
    err_c_abs,
    err_c_scaled,
    horizon_medians,
    iqr,
    rollout_aggregate,
    rollout_err_h,
    sd_rel,
    specrad_trio,
    state_rms,
)
from noisyopinf.noise import add_white
from noisyopinf.protocol import LAMBDAS, _validation_score, larger_lambda_index
from noisyopinf.rollout import rollout, threshold_from_train
from noisyopinf.seeds import rng
from noisyopinf.systems import galerkin_burgers
from noisyopinf.systems.config import project_root

RHO_GRID = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)
ESTIMATORS = ("oracle", "ols", "ridge", "mc", "mc_psd", "iv", "iv2")
SIGMA_METHODS = {"mc", "mc_psd", "iv", "iv2"}
MEASURES = (
    "err_O_rel", "err_A_rel", "err_H_rel", "err_c_abs", "err_c_scaled",
    "specrad_hat", "specrad_ref", "specrad_err", "ep_violation", "ep_violation_ref",
    "rollout_med", "rollout_med_h10", "rollout_med_h50", "rollout_mean_finite", "diverged_frac",
)
REQUIRED = {
    "system": "s1",
    "r": 8,
    "rho": [0, 0.01, 0.02, 0.05, 0.1, 0.2],
    "n_train": [10, 30, 100],
    "estimators": list(ESTIMATORS),
    "sigma2_source": True,
    "replicates": 100,
    "experiment_id": 2,
    "validation_offset": 100,
}
PILOT = {"rho": 0.05, "n_train": 30, "replicates": (0, 1)}
N_VAL = 20
N_TEST = 50
BOOTSTRAP = "one rng(0, 77); expand_cases order; ESTIMATORS order; 2000 median resamples"
_CACHE: dict[str, dict[str, np.ndarray]] = {}


def load_config(path: Path | None = None) -> dict:
    """读取并核对 E2 网格。实验编号固定为 2。"""
    source = project_root() / "configs" / "e2.yaml" if path is None else Path(path)
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    for key, value in REQUIRED.items():
        if loaded.get(key) != value:
            raise ValueError(f"e2.yaml {key} must be {value}")
    if list(loaded.get("rho_grid", [])) != list(RHO_GRID):
        raise ValueError("e2.yaml rho_grid must be the master grid")
    return loaded


def expand_cases(config: dict) -> list[dict]:
    """按 rho、n_train 展开 18 个配置。估计器不在这里复制。"""
    cases = []
    for rho in config["rho"]:
        for n_train in config["n_train"]:
            cases.append({
                "rho": float(rho),
                "n_train": int(n_train),
                "rho_index": rho_index(float(rho)),
                "replicates": int(config["replicates"]),
            })
    return cases


def planned_runs(config: dict) -> int:
    """18 个配置、7 个估计器、R 次重复。正式网格是 12600。"""
    return len(expand_cases(config)) * len(ESTIMATORS) * int(config["replicates"])


def pairs_per_estimator(method: str, n_train: int, length: int) -> int:
    """每条轨迹单独配对。IV 丢掉 2 个端点，IV2 丢掉 3 个，其余丢掉 1 个。"""
    dropped = {"iv": 2, "iv2": 3}.get(method, 1)
    return int(n_train) * (int(length) - dropped)


def load_s1_split(split_id: int, count: int) -> list[np.ndarray]:
    """按 index 0..count-1 生成一个划分。超界时停止，不跳到后面的 index。"""
    produced = Parallel(n_jobs=_workers(), backend="loky")(
        delayed(galerkin_burgers.rollout)(index, int(split_id)) for index in range(int(count))
    )
    rows = []
    for index, (traj, peak) in enumerate(produced):
        if (not np.isfinite(traj).all()) or float(peak) >= 5.0:
            raise RuntimeError(f"S1 split {split_id} index {index} failed the finite/peak check")
        rows.append(np.asarray(traj, dtype=np.float64))
    return rows


def shared_threshold(clean_trajs) -> float:
    """B(n_train)=1e3 * 干净训练前缀的最大无穷范数。"""
    return float(threshold_from_train(clean_trajs))


def noise_keys(rho_index_value: int, replicate: int, traj_index: int, validation: bool) -> tuple[int, ...]:
    """03 的噪声键。验证集的 traj_index 加 100，避免和训练子流重合。"""
    offset = 100 if validation else 0
    return (1, 4, 2, int(rho_index_value), int(replicate), int(traj_index) + offset)


def add_split_noise(clean, sigma: float, rho_index_value: int, replicate: int, validation: bool):
    """一个划分、一次重复的白噪声。每条轨迹使用自己的子流。"""
    return [
        add_white(traj, sigma, rng(*noise_keys(rho_index_value, replicate, index, validation)))
        for index, traj in enumerate(clean)
    ]


def write_state_cache(path: Path, train, val, test) -> None:
    """把三个划分写成工作进程可以重复读取的缓存。"""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        train=np.stack(train),
        val=np.stack(val),
        test=np.stack(test),
    )


def evaluate_job(task: dict) -> dict:
    """一次重复的七个估计器。随机数只由任务里的键决定。"""
    state = _cached_states(task["cache"])
    n_train = int(task["n_train"])
    clean = [state["train"][index] for index in range(n_train)]
    val = [state["val"][index] for index in range(state["val"].shape[0])]
    test = [state["test"][index] for index in range(state["test"].shape[0])]
    return evaluate_arrays(clean, val, test, task)


def evaluate_arrays(clean, val, test, task: dict) -> dict:
    """Oracle 用干净训练。其余六个估计器共用这一次的带噪训练。"""
    sigma2 = sigma2_of(clean, float(task["rho"]))
    sigma = float(np.sqrt(sigma2))
    noisy_train = add_split_noise(clean, sigma, task["rho_index"], task["replicate"], False)
    noisy_val = add_split_noise(val, sigma, task["rho_index"], task["replicate"], True)
    limit = float(task["threshold"])
    ridge = _ridge(noisy_train, noisy_val, limit)
    rows, operators = [], {}
    for method in ESTIMATORS:
        row, operator = _one_estimator(clean, noisy_train, test, task, method, sigma2, ridge["lam"], limit)
        rows.append(row)
        operators[method] = operator
    return {
        "rows": rows, "operators": operators, "ridge": ridge, "threshold": limit, "sigma2": sigma2,
        "rho": float(task["rho"]), "rho_index": int(task["rho_index"]),
        "n_train": int(task["n_train"]), "replicate": int(task["replicate"]),
    }


def extrapolate_seconds(pilot_seconds: float, pilot_reps: int) -> float:
    """用 n_train=30 的每次重复时间估算 6×3×100 次重复。"""
    return float(pilot_seconds) / float(pilot_reps) * 6.0 * 3.0 * 100.0


def summary_rows(cases: list[dict], rows: list[dict], operators: dict[str, np.ndarray], truth: np.ndarray) -> list[dict]:
    """按配置和估计器的固定顺序汇总，再消耗一条 rng(0, 77)。"""
    grouped = _grouped(rows)
    summaries = []
    for case in cases:
        limit = float(grouped[_locator(case, "ols", 0)]["blowup_threshold"])
        for method in ESTIMATORS:
            chosen = [grouped[_locator(case, method, replicate)] for replicate in range(case["replicates"])]
            stacked = [operators[_stored_key(case, method, replicate)] for replicate in range(case["replicates"])]
            summaries.append(_one_summary(case, method, chosen, stacked, truth, limit))
    _write_paired(summaries, grouped)
    _bootstrap(summaries)
    return summaries


def csv_row(row: dict) -> dict:
    """runs.csv 只保留 03 的列。共同阈值留在汇总表。"""
    return {column: row.get(column) for column in RUN_COLUMNS}


def operator_key(rho_index_value: int, n_train: int, method: str, replicate: int) -> str:
    """算子归档键。重复编号固定三位。"""
    return f"rep_rho{int(rho_index_value)}_n{int(n_train)}_{method}_r{int(replicate):03d}"


def _workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def _cached_states(path: str) -> dict[str, np.ndarray]:
    if path not in _CACHE:
        with np.load(path) as loaded:
            _CACHE[path] = {name: np.array(loaded[name], dtype=np.float64) for name in ("train", "val", "test")}
    return _CACHE[path]


def _ridge(noisy_train, noisy_val, limit: float) -> dict:
    started = time.perf_counter()
    scores = [float(_validation_score(noisy_train, noisy_val, lam, limit)) for lam in LAMBDAS]
    chosen = larger_lambda_index(scores)
    best = scores[chosen]
    return {
        "lam": float(LAMBDAS[chosen]),
        "scores": scores,
        "n_tied": int(sum(score == best for score in scores)),
        "n_nonfinite": int(sum(not np.isfinite(score) for score in scores)),
        "seconds": time.perf_counter() - started,
    }


def _one_estimator(clean, noisy_train, test, task: dict, method: str, sigma2: float, lam: float, limit: float):
    train = clean if method == "oracle" else noisy_train
    variance = sigma2 if method in SIGMA_METHODS else None
    chosen = lam if method == "ridge" else None
    result = fit(train, method, sigma2=variance, lam=chosen)
    row = _blank_row(task, method, result, sigma2, chosen, limit)
    if not np.isfinite(result.O).all():
        row["status"] = "failed"
        row["error"] = str(result.diagnostics.get("error", "non-finite operator"))
        for name in MEASURES:
            row[name] = float("nan")
        return row, result.O
    _fill_metrics(row, result.O, task["truth"], clean, test, limit)
    row["status"] = "ok"
    return row, result.O


def _blank_row(task: dict, method: str, result, sigma2: float, lam, limit: float) -> dict:
    row = {column: None for column in RUN_COLUMNS}
    row.update({
        "exp": "e2", "system": "s1", "rho": float(task["rho"]), "sigma2_true": float(sigma2),
        "sigma2_used": float(sigma2) if method in SIGMA_METHODS else None,
        "sigma2_source": "true" if method in SIGMA_METHODS else "na",
        "r": 8, "n_train": int(task["n_train"]), "n_pairs": int(result.n_pairs),
        "basis": "galerkin", "estimator": method, "lam": lam, "replicate": int(task["replicate"]),
        "seed_keys": f"1,4,2,{int(task['rho_index'])},{int(task['replicate'])},traj|val+100",
        "fit_seconds": float(result.fit_seconds),
        "gram_cond": result.diagnostics.get("gram_cond"),
        "gram_min_eig": result.diagnostics.get("gram_min_eig"),
        "n_clipped": result.diagnostics.get("n_clipped"),
        "blowup_threshold": float(limit),
    })
    return row


def _fill_metrics(row: dict, operator, truth, clean, test, limit: float) -> None:
    row["err_O_rel"] = err_O_rel(operator, truth)
    row["err_A_rel"] = err_A_rel(operator, truth)
    row["err_H_rel"] = err_H_rel(operator, truth)
    row["err_c_abs"] = err_c_abs(operator, truth)
    row["err_c_scaled"] = err_c_scaled(operator, truth, state_rms(np.vstack(clean)))
    row["specrad_hat"], row["specrad_ref"], row["specrad_err"] = specrad_trio(operator, truth)
    snapshots = np.vstack(test)
    row["ep_violation"] = ep_violation(operator, snapshots)
    row["ep_violation_ref"] = ep_violation(truth, snapshots)
    row.update(_predict(operator, test, limit))


def _predict(operator, test, limit: float) -> dict:
    errors, flags = [], []
    early = {10: [], 50: []}
    for traj in test:
        steps = int(traj.shape[0] - 1)
        predicted, diverged = rollout(operator, traj[0], steps, limit)
        errors.append(float("inf") if diverged else rollout_err_h(predicted, traj, steps))
        flags.append(bool(diverged))
        for horizon in (10, 50):
            early[horizon].append(rollout_err_h(predicted, traj, horizon))
    summary = rollout_aggregate(errors, flags)
    summary.update(horizon_medians(early))
    return summary


def _grouped(rows: list[dict]) -> dict:
    return {(float(row["rho"]), int(row["n_train"]), row["estimator"], int(row["replicate"])): row for row in rows}


def _locator(case: dict, method: str, replicate: int) -> tuple:
    return (float(case["rho"]), int(case["n_train"]), method, int(replicate))


def _stored_key(case: dict, method: str, replicate: int) -> str:
    return operator_key(case["rho_index"], case["n_train"], method, replicate)


def _one_summary(case, method, rows, operators, truth, limit) -> dict:
    failed = [int(row["replicate"]) for row in rows if row["status"] != "ok"]
    summary = {
        "exp": "e2", "system": "s1", "rho": case["rho"], "rho_index": case["rho_index"],
        "n_train": case["n_train"], "estimator": method, "replicates": len(rows),
        "sigma2": rows[0]["sigma2_true"], "sigma2_source": rows[0]["sigma2_source"],
        "blowup_threshold": float(limit), "n_failed": len(failed),
        "failed_replicates": ",".join(str(item) for item in failed),
        "status": "ok" if not failed else "failed",
    }
    _copy_measures(summary, rows)
    summary["bias_O_rel"] = bias_rel(operators, truth)
    summary["sd_O_rel"] = sd_rel(operators, truth)
    summary.update(block_bias_sd(operators, truth))
    return summary


def _copy_measures(summary: dict, rows: list[dict]) -> None:
    for name in MEASURES:
        q25, median, q75 = _quartiles([row[name] for row in rows])
        summary[f"{name}_q25"] = q25
        summary[f"{name}_median"] = median
        summary[f"{name}_q75"] = q75
        summary[f"{name}_iqr"] = _spread(q25, q75)


def _quartiles(values) -> tuple[float, float, float]:
    """线性分位数。中位数用 np.median，inf 保持 inf，任一 NaN 则三个都是 NaN。"""
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0 or np.isnan(data).any():
        return float("nan"), float("nan"), float("nan")
    ordered = np.sort(data)
    return _at(ordered, 0.25), float(np.median(ordered)), _at(ordered, 0.75)


def _at(ordered: np.ndarray, quantile: float) -> float:
    position = float(quantile) * (ordered.size - 1)
    low = int(np.floor(position))
    high = int(np.ceil(position))
    if low == high or np.isinf(ordered[low]) or np.isinf(ordered[high]):
        return float(ordered[low] if low == high else np.inf)
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def _spread(low: float, high: float) -> float:
    if low == high:
        return 0.0 if np.isfinite(low) or np.isinf(low) else float("nan")
    return float(high - low)


def _write_paired(summaries: list[dict], grouped: dict) -> None:
    """配对差按 replicate 对齐到 OLS，不按表格行号对齐。"""
    for summary in summaries:
        gaps = []
        for replicate in range(int(summary["replicates"])):
            key = (float(summary["rho"]), int(summary["n_train"]), summary["estimator"], replicate)
            baseline = grouped[(float(summary["rho"]), int(summary["n_train"]), "ols", replicate)]
            gaps.append(float(grouped[key]["err_O_rel"]) - float(baseline["err_O_rel"]))
        data = np.asarray(gaps, dtype=np.float64)
        summary["paired_median"] = float(np.median(data))
        summary["paired_iqr"] = iqr(data)
        summary["_gaps"] = data


def _bootstrap(summaries: list[dict]) -> None:
    """按汇总行的既有顺序消耗 rng(0, 77)。调用方必须先排好配置和估计器。"""
    generated = rng(0, 77)
    for summary in summaries:
        low, high = bootstrap_interval(summary.pop("_gaps"), generated, 2000)
        summary["boot_low"] = low
        summary["boot_high"] = high
