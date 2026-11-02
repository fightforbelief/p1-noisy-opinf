"""E3：S2 上按 POD 基比较七个估计器。噪声加在全阶状态上。"""

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
from noisyopinf.e1 import rho_index
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
    proj_floor,
    rollout_aggregate,
    rollout_err_h,
    sd_rel,
    specrad_trio,
    state_rms,
)
from noisyopinf.noise import add_white, sigma_from_rho
from noisyopinf.pod import estimate_sigma2, pod_basis, project
from noisyopinf.protocol import LAMBDAS, _validation_score, larger_lambda_index
from noisyopinf.rollout import rollout
from noisyopinf.seeds import rng

EXPERIMENT_ID = 3
SYSTEM_ID = 2
ESTIMATORS = ("oracle", "ols", "ridge", "mc", "mc_psd", "iv", "iv2")
SIGMA_METHODS = frozenset({"mc", "mc_psd", "iv", "iv2"})
SIGMA_SOURCES = ("true", "est_trailing", "est_gd")
BASES = ("clean", "noisy")
MEASURES = (
    "err_O_rel", "err_A_rel", "err_H_rel", "err_c_abs", "err_c_scaled",
    "specrad_hat", "specrad_ref", "specrad_err", "ep_violation", "ep_violation_ref",
    "rollout_med", "rollout_med_h10", "rollout_med_h50", "rollout_mean_finite", "diverged_frac",
)
SOURCE_CODE = {"na": 0, "true": 1, "est_trailing": 2, "est_gd": 3}
BASIS_CODE = {"clean": 0, "noisy": 1}
BOOTSTRAP = "rng(0, 77) for err minus ols; rng(0, 78) afterwards for sigma source minus true"
REQUIRED = {
    "system": "s2",
    "r": 8,
    "n_train": 30,
    "n_val": 20,
    "n_test": 50,
    "rho": [0, 0.01, 0.02, 0.05, 0.1, 0.2],
    "basis": ["clean", "noisy"],
    "estimators": list(ESTIMATORS),
    "sigma2_sources": ["true", "est_trailing", "est_gd"],
    "replicates": 20,
    "experiment_id": 3,
    "validation_offset": 100,
}
PILOT = {"rho": 0.05, "replicates": (0, 1)}
METHOD_FOR_ESTIMATE = {"est_trailing": "trailing_median", "est_gd": "gavish_donoho"}


def load_config(path: Path | None = None) -> dict:
    """读取并核对 E3 网格。实验编号固定为 3。"""
    from noisyopinf.systems.config import project_root

    source = project_root() / "configs" / "e3.yaml" if path is None else Path(path)
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    for key, value in REQUIRED.items():
        if loaded.get(key) != value:
            raise ValueError(f"e3.yaml {key} must be {value}")
    if list(loaded.get("rho_grid", [])) != list(REQUIRED["rho"]):
        raise ValueError("e3.yaml rho_grid must be the master grid")
    return loaded


def row_plan() -> list[tuple[str, str]]:
    """一个重复、一个基上的 15 行。不需要 σ² 的估计器只出现一次。"""
    plan = []
    for method in ESTIMATORS:
        if method in SIGMA_METHODS:
            plan.extend((method, source) for source in SIGMA_SOURCES)
        else:
            plan.append((method, "na"))
    return plan


def expand_cases(config: dict) -> list[dict]:
    """按 rho、basis 展开。估计器和 σ² 来源不在这里复制。"""
    cases = []
    for rho in config["rho"]:
        for basis in config["basis"]:
            cases.append({
                "rho": float(rho),
                "basis": str(basis),
                "rho_index": rho_index(float(rho)),
                "replicates": int(config["replicates"]),
            })
    return cases


def planned_runs(config: dict) -> int:
    """6 × 2 × R × 15。正式网格是 3600。"""
    return len(expand_cases(config)) * int(config["replicates"]) * len(row_plan())


def noise_keys(rho_index_value: int, replicate: int, traj_index: int, validation: bool) -> tuple[int, ...]:
    """03 的噪声键。S2 的 system_id 是 2，E3 的 experiment_id 是 3。"""
    offset = 100 if validation else 0
    return (SYSTEM_ID, 4, EXPERIMENT_ID, int(rho_index_value), int(replicate), int(traj_index) + offset)


def add_full_order_noise(clean, sigma: float, rho_index_value: int, replicate: int, validation: bool):
    """在投影之前给全阶轨迹加白噪声。每条轨迹使用自己的子流。"""
    return [
        add_white(traj, sigma, rng(*noise_keys(rho_index_value, replicate, index, validation)))
        for index, traj in enumerate(clean)
    ]


def sigma2_for_source(source: str, sigma2_true: float, estimates: dict) -> float | None:
    """按来源取 σ²。估计失败时抛出 ValueError，不用 true 顶替。"""
    if source == "na":
        return None
    if source == "true":
        return float(sigma2_true)
    payload = estimates[source]
    value = payload.get("value")
    if value is None or not np.isfinite(value):
        raise ValueError(f"sigma2 unavailable: {source}")
    return float(value)


def estimate_variances(noisy_train, clean_train, sigma2_true: float) -> dict:
    """在全阶带噪训练快照上估计两种 σ²。适用条件是 K_tot >= 2N。"""
    found = {}
    for source, method in METHOD_FOR_ESTIMATE.items():
        try:
            value, diagnostics = estimate_sigma2(noisy_train, method, clean_train)
            error = ""
        except (ValueError, np.linalg.LinAlgError) as exc:
            value, diagnostics, error = float("nan"), {}, str(exc)
        true = float(sigma2_true)
        if not np.isfinite(value):
            relative = float("nan")
        elif true == 0.0:
            relative = 0.0 if value == 0.0 else float("nan")
        else:
            relative = float((value - true) / true)
        found[source] = {
            "value": float(value) if value is not None else float("nan"),
            "relative": relative,
            "error": error,
            "diagnostics": diagnostics,
            "applicable": bool(diagnostics) and error == "",
        }
    return found


def same_reference(operators: list[np.ndarray]) -> bool:
    """同一基、同一次重复的估计器必须共用一个参考。跨基的矩阵不能拿来相减。"""
    first = np.asarray(operators[0], dtype=np.float64)
    return all(np.allclose(first, np.asarray(item, dtype=np.float64), atol=0.0, rtol=0.0) for item in operators[1:])


def operator_key(rho_index_value: int, basis: str, method: str, source: str, replicate: int) -> str:
    """估计算子的归档键。"""
    return f"rep_rho{int(rho_index_value)}_{basis}_{method}_{source}_r{int(replicate):03d}"


def reference_key(rho_index_value: int, basis: str, replicate: int) -> str:
    """该次重复、该基上的 O_clean。"""
    return f"ref_rho{int(rho_index_value)}_{basis}_r{int(replicate):03d}"


def array_sha256(values: np.ndarray) -> str:
    """数组内容的 SHA-256。用来核对基和参考，不替代文件哈希。"""
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_replicate(clean_train, clean_val, clean_test, rho: float, rho_index_value: int, replicate: int, v_clean=None, rank: int = 8) -> dict:
    """一次噪声重复的两个基。测试轨迹保持干净。"""
    sigma = float(sigma_from_rho("s2", float(rho), clean_train))
    sigma2_true = float(sigma * sigma)
    noisy_train = add_full_order_noise(clean_train, sigma, rho_index_value, replicate, False)
    noisy_val = add_full_order_noise(clean_val, sigma, rho_index_value, replicate, True)
    estimates = estimate_variances(noisy_train, clean_train, sigma2_true)
    if v_clean is None:
        v_clean, _ = pod_basis(clean_train, int(rank))
    else:
        v_clean = np.asarray(v_clean, dtype=np.float64)
    resolved = int(v_clean.shape[1])
    clean_basis = _one_basis(
        "clean", v_clean, clean_train, clean_val, clean_test, noisy_train, noisy_val,
        rho, rho_index_value, replicate, sigma2_true, estimates,
    )
    v_noisy, _ = pod_basis(noisy_train, resolved)
    noisy_basis = _one_basis(
        "noisy", v_noisy, clean_train, clean_val, clean_test, noisy_train, noisy_val,
        rho, rho_index_value, replicate, sigma2_true, estimates,
    )
    for payload in (clean_basis, noisy_basis):
        payload.pop("clean_q", None)
        payload.pop("noisy_q", None)
    return {
        "rho": float(rho),
        "rho_index": int(rho_index_value),
        "replicate": int(replicate),
        "sigma2_true": sigma2_true,
        "sigma": sigma,
        "estimates": estimates,
        "bases": {"clean": clean_basis, "noisy": noisy_basis},
        "full_order_noise_rms": float(np.sqrt(np.mean((np.vstack(noisy_train) - np.vstack(clean_train)) ** 2))),
    }


def _one_basis(basis, basis_matrix, clean_train, clean_val, clean_test, noisy_train, noisy_val, rho, rho_index_value, replicate, sigma2_true, estimates) -> dict:
    """在一个已经取定的基上拟合。参考算子来自干净轨迹在这个基上的投影。"""
    clean_q = [project(traj, basis_matrix) for traj in clean_train]
    noisy_q = [project(traj, basis_matrix) for traj in noisy_train]
    clean_val_q = [project(traj, basis_matrix) for traj in clean_val]
    noisy_val_q = [project(traj, basis_matrix) for traj in noisy_val]
    limit = float(1000.0 * max(float(np.max(np.abs(traj))) for traj in clean_q))
    reference = fit(clean_q, "ols").O
    ridge = _ridge(noisy_q, noisy_val_q, limit)
    stacked_test = np.vstack(clean_test)
    floor = float(proj_floor(basis_matrix, stacked_test))
    rows, operators, details = [], {}, []
    for method, source in row_plan():
        row, operator, trajectory_rows = _one_row(
            method, source, clean_q, noisy_q, clean_test, basis_matrix, reference, floor, limit,
            rho, rho_index_value, replicate, basis, sigma2_true, estimates, ridge["lam"],
        )
        rows.append(row)
        operators[operator_key(rho_index_value, basis, method, source, replicate)] = operator
        details.extend(trajectory_rows)
    operators[reference_key(rho_index_value, basis, replicate)] = np.asarray(reference, dtype=np.float64)
    ortho = float(np.linalg.norm(basis_matrix.T @ basis_matrix - np.eye(basis_matrix.shape[1])))
    return {
        "rows": rows,
        "operators": operators,
        "details": details,
        "ridge": ridge,
        "threshold": limit,
        "reference": np.asarray(reference, dtype=np.float64),
        "reference_hash": array_sha256(reference),
        "basis_hash": array_sha256(basis_matrix),
        "basis": np.asarray(basis_matrix, dtype=np.float64),
        "orthogonality": ortho,
        "proj_floor": floor,
        "clean_q_rms": float(state_rms(np.vstack(clean_q))),
        "clean_q": clean_q,
        "noisy_q": noisy_q,
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


def _one_row(method, source, clean_q, noisy_q, clean_test, basis_matrix, reference, floor, limit, rho, rho_index_value, replicate, basis, sigma2_true, estimates, lam):
    train = clean_q if method == "oracle" else noisy_q
    try:
        variance = sigma2_for_source(source, sigma2_true, estimates)
        chosen = lam if method == "ridge" else None
        result = fit(train, method, sigma2=variance, lam=chosen)
        error = ""
        operator = result.O
        n_pairs = int(result.n_pairs)
        seconds = float(result.fit_seconds)
        diagnostics = result.diagnostics
    except ValueError as exc:
        operator = np.full_like(reference, np.nan)
        n_pairs = 0
        seconds = 0.0
        diagnostics = {}
        error = str(exc)
        variance = None
    row = {column: None for column in RUN_COLUMNS}
    row.update({
        "exp": "e3", "system": "s2", "rho": float(rho), "sigma2_true": float(sigma2_true),
        "sigma2_used": None if variance is None else float(variance),
        "sigma2_source": source, "r": int(basis_matrix.shape[1]), "n_train": len(train),
        "n_pairs": n_pairs, "basis": basis, "estimator": method,
        "lam": float(lam) if method == "ridge" else None,
        "replicate": int(replicate),
        "seed_keys": f"2,4,3,{int(rho_index_value)},{int(replicate)},traj|val+100",
        "fit_seconds": seconds,
        "gram_min_eig": diagnostics.get("gram_min_eig"),
        "gram_cond": diagnostics.get("gram_cond"),
        "n_clipped": diagnostics.get("n_clipped"),
        "proj_floor": float(floor),
        "status": "ok",
        "error": error,
    })
    trajectory_rows = []
    if error or not np.isfinite(operator).all():
        row["status"] = "failed"
        row["error"] = error or str(diagnostics.get("error", "non-finite operator"))
        for name in MEASURES:
            row[name] = float("nan")
        row["proj_floor"] = float(floor)
        for index, traj in enumerate(clean_test):
            trajectory_rows.append(_empty_detail(rho, basis, method, source, replicate, index, limit, floor, traj, basis_matrix))
        return row, operator, trajectory_rows
    _fill_metrics(row, operator, reference, clean_q, clean_test, basis_matrix, limit, trajectory_rows, rho, basis, method, source, replicate, floor)
    return row, operator, trajectory_rows


def _fill_metrics(row, operator, reference, clean_q, clean_test, basis_matrix, limit, trajectory_rows, rho, basis, method, source, replicate, floor) -> None:
    row["err_O_rel"] = err_O_rel(operator, reference)
    row["err_A_rel"] = err_A_rel(operator, reference)
    row["err_H_rel"] = err_H_rel(operator, reference)
    row["err_c_abs"] = err_c_abs(operator, reference)
    row["err_c_scaled"] = err_c_scaled(operator, reference, state_rms(np.vstack(clean_q)))
    row["specrad_hat"], row["specrad_ref"], row["specrad_err"] = specrad_trio(operator, reference)
    snapshots = np.vstack([project(traj, basis_matrix) for traj in clean_test])
    row["ep_violation"] = ep_violation(operator, snapshots)
    row["ep_violation_ref"] = ep_violation(reference, snapshots)
    errors, flags = [], []
    early = {10: [], 50: []}
    for index, traj in enumerate(clean_test):
        reduced = project(traj, basis_matrix)
        steps = int(traj.shape[0] - 1)
        predicted, diverged = rollout(operator, reduced[0], steps, limit)
        full = float("inf") if diverged else rollout_err_h(predicted, traj, steps, basis_matrix)
        short = {horizon: rollout_err_h(predicted, traj, horizon, basis_matrix) for horizon in (10, 50)}
        errors.append(full)
        flags.append(bool(diverged))
        for horizon in (10, 50):
            early[horizon].append(short[horizon])
        trajectory_rows.append({
            "rho": float(rho), "basis": basis, "estimator": method, "sigma2_source": source,
            "replicate": int(replicate), "traj": int(index),
            "err_full": full, "err_h10": short[10], "err_h50": short[50],
            "diverged": bool(diverged),
            "first_step": first_divergence_step(predicted, limit, diverged),
            "threshold": float(limit),
            "proj_floor_traj": float(proj_floor(basis_matrix, traj)),
            "proj_floor": float(floor),
        })
    row.update(rollout_aggregate(errors, flags))
    row.update(horizon_medians(early))


def _empty_detail(rho, basis, method, source, replicate, index, limit, floor, traj, basis_matrix) -> dict:
    return {
        "rho": float(rho), "basis": basis, "estimator": method, "sigma2_source": source,
        "replicate": int(replicate), "traj": int(index),
        "err_full": float("nan"), "err_h10": float("nan"), "err_h50": float("nan"),
        "diverged": False, "first_step": -1, "threshold": float(limit),
        "proj_floor_traj": float(proj_floor(basis_matrix, traj)),
        "proj_floor": float(floor),
    }


def first_divergence_step(predicted: np.ndarray, limit: float, diverged: bool) -> int:
    """由 rollout 留下的轨迹恢复首次发散步。未发散时返回 -1。"""
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
    if first_nan == 0:
        return 0
    previous = states[first_nan - 1]
    if np.isfinite(previous).all() and float(np.max(np.abs(previous))) > float(limit):
        return first_nan - 1
    return first_nan


def summary_rows(cases: list[dict], rows: list[dict], operators: dict[str, np.ndarray]) -> list[dict]:
    """先汇总，再用 rng(0, 77) 比较 OLS，最后用 rng(0, 78) 比较 σ² 来源。"""
    grouped = _grouped(rows)
    summaries = []
    for case in cases:
        limit_values = [
            float(grouped[_locator(case, "ols", "na", replicate)]["proj_floor"])
            for replicate in range(case["replicates"])
        ]
        for method, source in row_plan():
            chosen = [grouped[_locator(case, method, source, replicate)] for replicate in range(case["replicates"])]
            stacked = [operators[operator_key(case["rho_index"], case["basis"], method, source, replicate)] for replicate in range(case["replicates"])]
            references = [operators[reference_key(case["rho_index"], case["basis"], replicate)] for replicate in range(case["replicates"])]
            summaries.append(_one_summary(case, method, source, chosen, stacked, references, limit_values))
    _write_paired(summaries, grouped)
    _bootstrap(summaries)
    return summaries


def _one_summary(case, method, source, rows, operators, references, floors) -> dict:
    failed = [int(row["replicate"]) for row in rows if row["status"] != "ok"]
    summary = {
        "exp": "e3", "system": "s2", "rho": case["rho"], "rho_index": case["rho_index"],
        "basis": case["basis"], "n_train": rows[0]["n_train"], "estimator": method,
        "sigma2_source": source, "replicates": len(rows),
        "sigma2_true": rows[0]["sigma2_true"],
        "proj_floor_median": float(np.median(np.asarray(floors, dtype=np.float64))),
        "n_failed": len(failed),
        "failed_replicates": ",".join(str(item) for item in failed),
        "status": "ok" if not failed else "failed",
    }
    _copy_measures(summary, rows)
    if _common_reference(references):
        summary["bias_O_rel"] = bias_rel(operators, references[0])
        summary["sd_O_rel"] = sd_rel(operators, references[0])
        summary.update(block_bias_sd(operators, references[0]))
    else:
        summary["bias_O_rel"] = float("nan")
        summary["sd_O_rel"] = float("nan")
        summary["bias_A_rel"] = float("nan")
        summary["sd_A_rel"] = float("nan")
        summary["bias_H_rel"] = float("nan")
        summary["sd_H_rel"] = float("nan")
    return summary


def _common_reference(references: list[np.ndarray]) -> bool:
    first = np.asarray(references[0], dtype=np.float64)
    return all(np.array_equal(first, np.asarray(item, dtype=np.float64)) for item in references[1:])


def _grouped(rows: list[dict]) -> dict:
    return {
        (float(row["rho"]), row["basis"], row["estimator"], row["sigma2_source"], int(row["replicate"])): row
        for row in rows
    }


def _locator(case: dict, method: str, source: str, replicate: int) -> tuple:
    return (float(case["rho"]), case["basis"], method, source, int(replicate))


def _copy_measures(summary: dict, rows: list[dict]) -> None:
    for name in (*MEASURES, "proj_floor"):
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
    """配对差按 replicate 对齐。跨 basis 的原始算子不相减。"""
    for summary in summaries:
        gaps, source_gaps = [], []
        for replicate in range(int(summary["replicates"])):
            key = (float(summary["rho"]), summary["basis"], summary["estimator"], summary["sigma2_source"], replicate)
            baseline = grouped[(float(summary["rho"]), summary["basis"], "ols", "na", replicate)]
            gaps.append(float(grouped[key]["err_O_rel"]) - float(baseline["err_O_rel"]))
            if summary["estimator"] in SIGMA_METHODS:
                truth = grouped[(float(summary["rho"]), summary["basis"], summary["estimator"], "true", replicate)]
                source_gaps.append(float(grouped[key]["err_O_rel"]) - float(truth["err_O_rel"]))
        summary["_gaps"] = np.asarray(gaps, dtype=np.float64)
        summary["paired_median"] = float(np.median(summary["_gaps"]))
        summary["paired_iqr"] = iqr(summary["_gaps"])
        if source_gaps:
            summary["_source_gaps"] = np.asarray(source_gaps, dtype=np.float64)
            summary["source_paired_median"] = float(np.median(summary["_source_gaps"]))
        else:
            summary["_source_gaps"] = None
            summary["source_paired_median"] = float("nan")


def _bootstrap(summaries: list[dict]) -> None:
    """04 的 OLS 比较先消耗 rng(0, 77)。σ² 来源比较随后消耗 rng(0, 78)。"""
    generated = rng(0, 77)
    for summary in summaries:
        low, high = bootstrap_interval(summary.pop("_gaps"), generated, 2000)
        summary["boot_low"] = low
        summary["boot_high"] = high
    generated = rng(0, 78)
    for summary in summaries:
        gaps = summary.pop("_source_gaps")
        if gaps is None:
            summary["source_boot_low"] = float("nan")
            summary["source_boot_high"] = float("nan")
            continue
        low, high = bootstrap_interval(gaps, generated, 2000)
        summary["source_boot_low"] = low
        summary["source_boot_high"] = high


def parameter_digest() -> str:
    """S2 参数哈希，和数据缓存使用同一个函数。"""
    return parameter_hash("s2")
