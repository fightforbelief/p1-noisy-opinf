"""E5 第 1 部分：干净 POD 上的 O_clean、O_IVc、O_IV2c。不改估计器公式。"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from noisyopinf.data import get_splits, parameter_hash
from noisyopinf.estimators import _instrument_rows, _pair_states, _peak_abs, fit, split_O
from noisyopinf.features import features
from noisyopinf.metrics import proj_floor, rollout_aggregate, rollout_err_h
from noisyopinf.pod import pod_basis, project
from noisyopinf.rollout import rollout, threshold_from_train

REL_FLOOR = 1.0e-12
METHODS = ("ols", "iv", "iv2")


def load_config(path: Path) -> dict:
    """读取第 1 部分网格。验证集不参与拟合。"""
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {
        "experiment": "E5_PART1",
        "system": "s2",
        "n_train": 100,
        "n_val": 0,
        "n_test": 50,
        "basis": "clean",
        "rho": 0,
        "r_max": 20,
        "sigma2": 0,
        "correct_intercept": False,
    }
    for key, value in expected.items():
        if loaded.get(key) != value:
            raise ValueError(f"e5 config {key} must be {value}")
    ranks = tuple(int(item) for item in loaded["r"])
    if any(rank < 1 or rank > int(loaded["r_max"]) for rank in ranks):
        raise ValueError("ranks must lie inside r_max")
    loaded["r"] = ranks
    loaded["relative_reference_floor"] = float(loaded.get("relative_reference_floor", REL_FLOOR))
    return loaded


def window_counts(length: int) -> dict:
    """各估计器在一条长度为 length 的轨迹上使用的闭区间时间下标。"""
    if length < 4:
        raise ValueError("trajectory shorter than the IV2 window")
    last = length - 1
    return {
        "ols": {"n_rows": length - 1, "input": (0, last - 1), "output": (1, last)},
        "iv": {
            "n_rows": length - 2,
            "input": (1, last - 1),
            "instrument": (0, last - 2),
            "output": (2, last),
        },
        "iv2": {
            "n_rows": length - 3,
            "input": (2, last - 1),
            "lag1": (1, last - 2),
            "lag2": (0, last - 3),
            "output": (3, last),
        },
    }


def load_clean_splits(n_train: int, n_test: int) -> dict:
    """读取嵌套的 S2 缓存。缺失的训练轨迹由现有生成器补齐。"""
    splits = get_splits("s2", n_train, 0, n_test)
    _require_prefix(splits["train"].indices, n_train)
    _require_prefix(splits["test"].indices, n_test)
    if _same_trajectory(splits["train"].U[0], splits["test"].U[0]):
        raise ValueError("train index 0 and test index 0 are the same trajectory")
    return splits


def clean_basis(train_u: list[np.ndarray], rank: int) -> tuple[np.ndarray, np.ndarray]:
    """只对训练快照做一次不中心化 SVD，返回前 rank 列。"""
    return pod_basis(train_u, rank)


def closure_rel(trajs: list[np.ndarray], operator: np.ndarray) -> float:
    """||Y - Z O||_F / ||Y||_F。Z、Y 是这条轨迹列表的 OLS 配对。"""
    inputs, outputs = _pair_states(trajs)
    residual = outputs - features(inputs) @ np.asarray(operator, dtype=np.float64)
    scale = float(np.linalg.norm(outputs))
    if scale == 0.0:
        return float("nan")
    return float(np.linalg.norm(residual) / scale)


def deviation_record(estimate: np.ndarray, reference: np.ndarray, floor: float) -> dict:
    """绝对偏离始终保留。参考块范数低于 floor 时相对偏离记为 NaN。"""
    gap = np.asarray(estimate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    rank = int(reference.shape[1])
    blocks = {"O": gap, "A": split_O(estimate, rank)[0] - split_O(reference, rank)[0], "H": split_O(estimate, rank)[1] - split_O(reference, rank)[1]}
    refs = {"O": reference, "A": split_O(reference, rank)[0], "H": split_O(reference, rank)[1]}
    record = {}
    for name in ("O", "A", "H"):
        absolute = float(np.linalg.norm(blocks[name]))
        ref_norm = float(np.linalg.norm(refs[name]))
        usable = ref_norm >= float(floor)
        record[f"dev_{name}_abs"] = absolute
        record[f"dev_{name}_rel"] = absolute / ref_norm if usable else float("nan")
        record[f"ref_{name}_norm"] = ref_norm
        record[f"rel_{name}_status"] = "OK" if usable else "SMALL_REFERENCE"
    return record


def fit_target(trajs: list[np.ndarray], method: str):
    """O_clean 用 OLS。O_IVc 与 O_IV2c 在干净数据上拟合且不做常数修正。"""
    if method == "ols":
        return fit(trajs, "ols")
    if method in {"iv", "iv2"}:
        return fit(trajs, method, sigma2=0.0, correct_intercept=False)
    raise ValueError(method)


def evaluate_rank(rank: int, basis: np.ndarray, train_u, test_u, floor: float) -> dict:
    """一个秩上的正式目标和诊断。诊断 OLS 不替换 O_clean。"""
    prefix = np.asarray(basis[:, :rank], dtype=np.float64)
    train_q = [project(traj, prefix) for traj in train_u]
    test_q = [project(traj, prefix) for traj in test_u]
    length = int(train_q[0].shape[0])
    fitted = {method: fit_target(train_q, method) for method in METHODS}
    reference = fitted["ols"].O
    limit = float(threshold_from_train(train_q))
    row = {
        "r": rank,
        "closure_rel": closure_rel(train_q, reference),
        "pooled_proj_reference": float(proj_floor(prefix, np.vstack(test_u))),
        "threshold": limit,
        "kappa": float(fitted["ols"].kappa),
        "trajectory_length": length,
        "n_train": len(train_q),
        "n_test": len(test_q),
    }
    row.update(_window_fields(length, fitted))
    for method in ("iv", "iv2"):
        row.update(_prefixed(deviation_record(fitted[method].O, reference, floor), method))
        row.update(_prediction(method, fitted[method].O, test_u, test_q, prefix, limit))
        row.update(_solver_fields(method, fitted[method]))
    row.update(_prediction("ols", reference, test_u, test_q, prefix, limit))
    row.update(_solver_fields("ols", fitted["ols"]))
    row.update(_diagnostic_windows(train_q, fitted, floor))
    return {"row": row, "operators": {method: fitted[method].O for method in METHODS}, "train_q": train_q, "fitted": fitted}


def scaled_matrices(trajs: list[np.ndarray], method: str) -> tuple[np.ndarray, np.ndarray, float]:
    """复现 fit 内部的峰值缩放，供条件数诊断。不重新求解。"""
    kappa = float(_peak_abs(trajs))
    scaled = [traj / kappa for traj in trajs]
    if method == "ols":
        states, _outputs = _pair_states(scaled)
        design = features(states)
        return design, design, kappa
    design, instruments, _outputs = _instrument_rows(scaled, method)
    return design, instruments, kappa


def spectrum_record(matrix: np.ndarray) -> dict:
    """高矩阵的奇异值、数值秩和条件数。数值秩沿用 NumPy lstsq 的默认截止。"""
    values = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    rows, cols = matrix.shape
    cutoff = float(np.finfo(np.float64).eps * max(rows, cols) * values[0]) if values.size else float("nan")
    rank = int(np.sum(values > cutoff)) if values.size else 0
    cond = float(values[0] / values[-1]) if values.size and values[-1] > 0.0 else float("inf")
    return {"n_rows": int(rows), "n_cols": int(cols), "numerical_rank": rank, "cond": cond, "svd_cutoff": cutoff, "singular_values": values}


def residual_cross(trajs: list[np.ndarray], operator: np.ndarray, method: str) -> np.ndarray:
    """干净 OLS 算子在该方法样本行上的残差，与该方法工具变量的交叉矩。"""
    if method == "ols":
        states, outputs = _pair_states(trajs)
        design = features(states)
        instruments = design
    else:
        design, instruments, outputs = _instrument_rows(trajs, method)
    residual = outputs - design @ np.asarray(operator, dtype=np.float64)
    return instruments.T @ residual


def write_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows([{key: _cell(value) for key, value in row.items()} for row in rows])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def data_fingerprint(states: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for state in states:
        digest.update(np.ascontiguousarray(state, dtype=np.float64).tobytes())
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _same_trajectory(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.allclose(left, right))


def _require_prefix(indices: list[int], count: int) -> None:
    if indices != list(range(count)):
        raise ValueError(f"expected indices 0..{count - 1}, got {indices[:3]}..{indices[-3:]}")


def _window_fields(length: int, fitted: dict) -> dict:
    spec = window_counts(length)
    fields = {}
    for method in METHODS:
        fields[f"n_pairs_{method}"] = int(fitted[method].n_pairs)
        fields[f"rows_per_traj_{method}"] = int(spec[method]["n_rows"])
        fields[f"input_first_{method}"] = int(spec[method]["input"][0])
        fields[f"input_last_{method}"] = int(spec[method]["input"][1])
        fields[f"output_first_{method}"] = int(spec[method]["output"][0])
        fields[f"output_last_{method}"] = int(spec[method]["output"][1])
    return fields


def _prefixed(record: dict, method: str) -> dict:
    return {f"{method}_{key}": value for key, value in record.items()}


def _prediction(method: str, operator: np.ndarray, test_u, test_q, prefix, limit: float) -> dict:
    finite = bool(np.isfinite(operator).all())
    if not finite:
        return {
            f"fit_status_{method}": "failed",
            f"rollout_stable_{method}": 0,
            f"rollout_med_{method}": float("nan"),
            f"rollout_med_h10_{method}": float("nan"),
            f"rollout_med_h50_{method}": float("nan"),
            f"diverged_frac_{method}": float("nan"),
        }
    errors, flags, short10, short50 = [], [], [], []
    steps = int(test_q[0].shape[0] - 1)
    for reduced, full in zip(test_q, test_u, strict=True):
        predicted, diverged = rollout(operator, reduced[0], steps, limit)
        errors.append(float("inf") if diverged else rollout_err_h(predicted, full, steps, prefix))
        flags.append(bool(diverged))
        short10.append(rollout_err_h(predicted, full, min(10, steps), prefix))
        short50.append(rollout_err_h(predicted, full, min(50, steps), prefix))
    summary = rollout_aggregate(errors, flags)
    stable = int(summary["diverged_frac"] == 0.0 and np.isfinite(summary["rollout_med"]))
    return {
        f"fit_status_{method}": "ok",
        f"rollout_stable_{method}": stable,
        f"rollout_med_{method}": summary["rollout_med"],
        f"rollout_med_h10_{method}": float(np.median(short10)),
        f"rollout_med_h50_{method}": float(np.median(short50)),
        f"diverged_frac_{method}": summary["diverged_frac"],
    }


def _solver_fields(method: str, result) -> dict:
    diagnostics = result.diagnostics
    solver = {
        "ols": "lstsq_rcond_none",
        "iv": "solve_normal_equations",
        "iv2": "two_stage_lstsq_rcond_none",
    }[method]
    return {
        f"solver_{method}": solver,
        f"used_pinv_{method}": 0,
        f"fit_seconds_{method}": float(result.fit_seconds),
        f"stored_instrument_cond_{method}": diagnostics.get("instrument_cond", float("nan")),
        f"stored_first_stage_r2_median_{method}": diagnostics.get("first_stage_r2_median", float("nan")),
        f"stored_gram_cond_{method}": diagnostics.get("gram_cond", float("nan")),
        f"stored_error_{method}": diagnostics.get("error", ""),
    }


def _diagnostic_windows(train_q, fitted: dict, floor: float) -> dict:
    """共同窗口 OLS 只作诊断，不进入正式偏离列。"""
    iv_window = fit_target([traj[1:] for traj in train_q], "ols")
    iv2_window = fit_target([traj[2:] for traj in train_q], "ols")
    iv_gap = deviation_record(fitted["iv"].O, iv_window.O, floor)
    iv2_gap = deviation_record(fitted["iv2"].O, iv2_window.O, floor)
    return {
        "diagnostic_n_pairs_ols_on_iv_window": int(iv_window.n_pairs),
        "diagnostic_n_pairs_ols_on_iv2_window": int(iv2_window.n_pairs),
        "diagnostic_iv_vs_window_ols_rel": iv_gap["dev_O_rel"],
        "diagnostic_iv_vs_window_ols_abs": iv_gap["dev_O_abs"],
        "diagnostic_iv2_vs_window_ols_rel": iv2_gap["dev_O_rel"],
        "diagnostic_iv2_vs_window_ols_abs": iv2_gap["dev_O_abs"],
        "diagnostic_closure_on_iv_window": closure_rel([traj[1:] for traj in train_q], fitted["ols"].O),
        "diagnostic_closure_on_iv2_window": closure_rel([traj[2:] for traj in train_q], fitted["ols"].O),
    }


def _cell(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    number = float(value)
    if np.isnan(number):
        return "nan"
    if np.isposinf(number):
        return "inf"
    if np.isneginf(number):
        return "-inf"
    return format(number, ".16g")
