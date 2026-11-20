"""E5 事后诊断：共同窗口分解和求解路径对照。不改正式估计器。"""

from __future__ import annotations

import numpy as np

from noisyopinf.e5 import spectrum_record
from noisyopinf.estimators import _median_r2, _peak_abs, _project, _to_original, fit, split_O
from noisyopinf.features import features

REL_FLOOR = 1.0e-12
TRUNCATION_REL = 1.0e-8
SOLVER_REL_FLAG = 1.0e-8
PRED_REL_TINY = 1.0e-8
PRED_REL_OBVIOUS = 1.0e-4
BLOCKS = ("O", "A", "H", "c")
FORMAL_METHODS = ("ols", "iv", "iv2")


def window_index_spec(length: int) -> dict:
    """一条轨迹上各窗口的闭区间下标。统一窗口不替换正式 IV。"""
    if length < 4:
        raise ValueError("trajectory shorter than the IV2 window")
    last = length - 1
    iv2_input = (2, last - 1)
    iv2_output = (3, last)
    return {
        "ols_full": {"n_rows": length - 1, "input": (0, last - 1), "output": (1, last)},
        "iv": {
            "n_rows": length - 2,
            "input": (1, last - 1),
            "instrument": (0, last - 2),
            "output": (2, last),
        },
        "iv2": {
            "n_rows": length - 3,
            "input": iv2_input,
            "lag1": (1, last - 2),
            "lag2": (0, last - 3),
            "output": iv2_output,
        },
        "ols_iv": {"n_rows": length - 2, "input": (1, last - 1), "output": (2, last)},
        "ols_iv2": {"n_rows": length - 3, "input": iv2_input, "output": iv2_output},
        "iv_unified": {
            "n_rows": length - 3,
            "input": iv2_input,
            "instrument": (1, last - 2),
            "output": iv2_output,
        },
    }


def per_trajectory_rows(traj: np.ndarray, kind: str):
    """返回设计、工具变量、输出。工具变量按单条轨迹构造。"""
    state = np.asarray(traj, dtype=np.float64)
    if kind == "ols_full":
        return features(state[:-1]), None, state[1:]
    if kind == "ols_iv":
        return features(state[1:-1]), None, state[2:]
    if kind == "ols_iv2":
        return features(state[2:-1]), None, state[3:]
    if kind == "iv":
        return features(state[1:-1]), features(state[:-2]), state[2:]
    if kind == "iv2":
        current = features(state[2:-1])
        lag1 = features(state[1:-2])[:, :-1]
        lag2 = features(state[:-3])[:, :-1]
        ones = np.ones((current.shape[0], 1), dtype=np.float64)
        return current, np.hstack([lag1, lag2, ones]), state[3:]
    if kind == "iv_unified":
        return features(state[2:-1]), features(state[1:-2]), state[3:]
    raise ValueError(kind)


def stack_rows(trajs, kind: str):
    """把每条轨迹的窗口行竖向拼起来，不在轨迹之间取滞后。"""
    parts = [per_trajectory_rows(traj, kind) for traj in trajs]
    design = np.vstack([item[0] for item in parts])
    outputs = np.vstack([item[2] for item in parts])
    if parts[0][1] is None:
        return design, None, outputs
    return design, np.vstack([item[1] for item in parts]), outputs


def three_gaps(method: np.ndarray, window: np.ndarray, full: np.ndarray) -> dict:
    """三个矩阵差。范数不可加，恒等式只在矩阵上成立。"""
    return {
        "method_minus_ols_full": np.asarray(method, dtype=np.float64) - np.asarray(full, dtype=np.float64),
        "method_minus_ols_window": np.asarray(method, dtype=np.float64) - np.asarray(window, dtype=np.float64),
        "ols_window_minus_ols_full": np.asarray(window, dtype=np.float64) - np.asarray(full, dtype=np.float64),
    }


def identity_residual(gaps: dict) -> np.ndarray:
    """method-full 减去另外两项之和。精确算术下应为零矩阵。"""
    return gaps["method_minus_ols_full"] - (
        gaps["method_minus_ols_window"] + gaps["ols_window_minus_ols_full"]
    )


def block_views(operator: np.ndarray, rank: int) -> dict:
    linear, quadratic, constant = split_O(operator, rank)
    return {"O": np.asarray(operator, dtype=np.float64), "A": linear, "H": quadratic, "c": constant}


def block_records(diff: np.ndarray, reference: np.ndarray, rank: int, floor: float) -> list[dict]:
    """绝对范数始终保留。参考块过小时相对范数为空。"""
    diff_blocks = block_views(diff, rank)
    ref_blocks = block_views(reference, rank)
    records = []
    for name in BLOCKS:
        absolute = float(np.linalg.norm(diff_blocks[name]))
        ref_norm = float(np.linalg.norm(ref_blocks[name]))
        usable = np.isfinite(ref_norm) and ref_norm >= float(floor)
        records.append({
            "block": name,
            "diff_abs": absolute,
            "ref_norm": ref_norm,
            "diff_rel": absolute / ref_norm if usable else float("nan"),
            "rel_status": "OK" if usable else "SMALL_REFERENCE",
        })
    return records


def equation_residual(matrix: np.ndarray, operator: np.ndarray, rhs: np.ndarray) -> float:
    """||M O - rhs||_F / ||rhs||_F。这是后向残差。"""
    if operator is None or not np.isfinite(operator).all():
        return float("nan")
    denom = float(np.linalg.norm(rhs))
    if denom == 0.0:
        return float("nan")
    return float(np.linalg.norm(matrix @ operator - rhs) / denom)


def normal_residual(matrix: np.ndarray, operator: np.ndarray, outputs: np.ndarray) -> float:
    """||M.T (Y - M O)||_F / ||M.T Y||_F。不组装 M.T M。"""
    if not np.isfinite(operator).all():
        return float("nan")
    residual = outputs - matrix @ operator
    denom = float(np.linalg.norm(matrix.T @ outputs))
    if denom == 0.0:
        return float("nan")
    return float(np.linalg.norm(matrix.T @ residual) / denom)


def fit_residual_rel(design: np.ndarray, operator: np.ndarray, outputs: np.ndarray) -> float:
    if not np.isfinite(operator).all():
        return float("nan")
    denom = float(np.linalg.norm(outputs))
    if denom == 0.0:
        return float("nan")
    return float(np.linalg.norm(outputs - design @ operator) / denom)


def prediction_gap(design: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        return float("nan")
    return float(np.linalg.norm(design @ (left - right)))


def equivalence_reliable(rank_w: int, cols_w: int, rank_solve: int, cols_solve: int) -> int:
    """满列秩且求解矩阵满秩时，QR 路径与原公式在精确算术下等价。"""
    if cols_w > 0 and cols_solve > 0 and int(rank_w) == int(cols_w) and int(rank_solve) == int(cols_solve):
        return 1
    return 0


def prediction_label(relative: float) -> str:
    if not np.isfinite(relative):
        return "NONFINITE"
    if float(relative) < PRED_REL_TINY:
        return "NOT_OBVIOUS"
    if float(relative) >= PRED_REL_OBVIOUS:
        return "OBVIOUS"
    return "INTERMEDIATE"


def svd_truncated_solve(matrix: np.ndarray, rhs: np.ndarray, relative_cutoff: float):
    """丢掉严格小于 relative_cutoff * smax 的奇异值。这不是正式估计器。"""
    matrix = np.asarray(matrix, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    u_factor, values, vh = np.linalg.svd(matrix, full_matrices=False)
    width = int(matrix.shape[1])
    out_cols = int(rhs.shape[1])
    if values.size == 0 or float(values[0]) == 0.0:
        empty = np.full((width, out_cols), np.nan)
        return empty, {"cutoff": float("nan"), "kept": 0, "backward_rel": float("nan")}
    cutoff = float(relative_cutoff) * float(values[0])
    keep = values >= cutoff
    inverse = np.zeros_like(values)
    inverse[keep] = 1.0 / values[keep]
    if not np.any(keep):
        operator = np.full((width, out_cols), np.nan)
    else:
        operator = (vh.T * inverse) @ (u_factor.T @ rhs)
    return operator, {"cutoff": cutoff, "kept": int(np.sum(keep)), "error": ""}


def iv_solve_operator(design: np.ndarray, instruments: np.ndarray, outputs: np.ndarray):
    """正式 IV：solve(W.T Z, W.T Y)。"""
    gram = instruments.T @ design
    rhs = instruments.T @ outputs
    try:
        operator = np.linalg.solve(gram, rhs)
        error = ""
    except np.linalg.LinAlgError as exc:
        operator = np.full((design.shape[1], outputs.shape[1]), np.nan)
        error = str(exc)
    return operator, {"error": error, "backward_rel": equation_residual(gram, operator, rhs), "lhs": gram}


def iv_qr_operator(design: np.ndarray, instruments: np.ndarray, outputs: np.ndarray):
    """经济型 QR。W 列满秩时与 solve(W.T Z, W.T Y) 精确等价。"""
    q_factor, r_factor = np.linalg.qr(np.asarray(instruments, dtype=np.float64), mode="reduced")
    lhs = q_factor.T @ design
    rhs = q_factor.T @ outputs
    operator, info = iv_linear(lhs, rhs, design.shape[1], outputs.shape[1])
    info["r_diag_ratio"] = _diag_ratio(r_factor)
    info["lhs"] = lhs
    info["rhs"] = rhs
    return operator, info


def iv2_lstsq_operator(design: np.ndarray, instruments: np.ndarray, outputs: np.ndarray):
    """现有两阶段。投影使用估计器里的 _project，不形成 Gram。"""
    projected = _project(instruments, design)
    operator, rank, error = _lstsq(projected, outputs)
    return operator, projected, {
        "error": error,
        "lstsq_rank": rank,
        "backward_rel": normal_residual(projected, operator, outputs),
    }


def iv2_qr_operator(design: np.ndarray, instruments: np.ndarray, outputs: np.ndarray):
    """用 QR 投影得到 Zhat，再直接最小二乘。"""
    q_factor, r_factor = np.linalg.qr(np.asarray(instruments, dtype=np.float64), mode="reduced")
    projected = q_factor @ (q_factor.T @ design)
    operator, rank, error = _lstsq(projected, outputs)
    return operator, projected, {
        "error": error,
        "lstsq_rank": rank,
        "backward_rel": normal_residual(projected, operator, outputs),
        "r_diag_ratio": _diag_ratio(r_factor),
    }


def ols_operator(design: np.ndarray, outputs: np.ndarray):
    operator, rank, error = _lstsq(design, outputs)
    return operator, {
        "error": error,
        "lstsq_rank": rank,
        "backward_rel": normal_residual(design, operator, outputs),
    }


def moment_record(instruments: np.ndarray, design: np.ndarray, residual: np.ndarray) -> dict:
    """原始交叉矩，以及按样本量归一化的经验矩。后者仍依赖特征尺度。"""
    cross = instruments.T @ residual
    raw = float(np.linalg.norm(cross))
    n_rows = int(residual.shape[0])
    denom = float(np.linalg.norm(instruments.T @ design))
    return {
        "cross_raw_frob": raw,
        "cross_per_sample_frob": raw / n_rows if n_rows else float("nan"),
        "cross_over_design_frob": raw / denom if denom > 0.0 else float("nan"),
        "residual_frob": float(np.linalg.norm(residual)),
        "n_rows": n_rows,
    }


def diagnose_rank(rank: int, trajs, saved: dict, published: dict | None = None) -> dict:
    """一个秩上的窗口分解和求解对照。trajs 已是该秩的投影。"""
    problems = []
    length = int(trajs[0].shape[0])
    spec = window_index_spec(length)
    kappa = float(_peak_abs(trajs))
    scaled = [traj / kappa for traj in trajs]
    original = {kind: stack_rows(trajs, kind) for kind in _KINDS}
    scaled_rows = {kind: stack_rows(scaled, kind) for kind in _KINDS}
    _check_counts(spec, original, len(trajs), problems)
    fresh = {
        "ols": fit(trajs, "ols").O,
        "iv": fit(trajs, "iv", sigma2=0.0, correct_intercept=False).O,
        "iv2": fit(trajs, "iv2", sigma2=0.0, correct_intercept=False).O,
    }
    for method in FORMAL_METHODS:
        gap = _relative(fresh[method], saved[method])
        if not np.isfinite(gap) or gap > 1.0e-8:
            problems.append(f"r{rank} {method} refit disagrees with saved operator: {gap}")
    operators = _restored_solvers(rank, kappa, scaled_rows, original)
    for method, key in (("ols", "ols_full"), ("iv", "iv_solve"), ("iv2", "iv2_lstsq")):
        gap = _relative(operators[key], fresh[method])
        if not np.isfinite(gap) or gap > 1.0e-8:
            problems.append(f"r{rank} {key} disagrees with fit(): {gap}")
    sliced = {
        "iv": fit([traj[1:] for traj in trajs], "ols").O,
        "iv2": fit([traj[2:] for traj in trajs], "ols").O,
    }
    decomp_rows, decomp_arrays, identity_values = _decomposition(
        rank, saved, operators["ols_iv"], operators["ols_iv2"], operators["iv_unified_solve"]
    )
    solver_rows, solver_arrays, spectra = _solver_table(
        rank, kappa, original, scaled_rows, operators, saved
    )
    moment_rows = _moments(rank, original, operators)
    summary = _summary_row(
        rank, kappa, saved, operators, sliced, identity_values, solver_rows, moment_rows, published
    )
    scale = max(float(np.max(np.abs(saved["ols"]))), 1.0)
    for method, value in identity_values.items():
        if value > 1.0e-9 * scale:
            problems.append(f"r{rank} {method} gap identity residual {value}")
    if published:
        for key, column in (
            ("iv_full_O_rel", "iv_dev_O_rel"),
            ("iv2_full_O_rel", "iv2_dev_O_rel"),
        ):
            published_value = float(published[column])
            if abs(summary[key] - published_value) > 1.0e-8 * max(1.0, abs(published_value)):
                problems.append(f"r{rank} {key} disagrees with part 1 {column}")
    arrays = {}
    arrays.update(decomp_arrays)
    arrays.update(solver_arrays)
    arrays.update(spectra)
    return {
        "decomp_rows": decomp_rows,
        "solver_rows": solver_rows,
        "moment_rows": moment_rows,
        "summary": summary,
        "arrays": arrays,
        "problems": problems,
    }


def _restored_solvers(rank: int, kappa: float, scaled_rows: dict, original: dict) -> dict:
    """在缩放坐标求解，再还原。截断路径单独标记，不混进正式解。"""
    del original
    out = {}
    for key, kind in (("ols_full", "ols_full"), ("ols_iv", "ols_iv"), ("ols_iv2", "ols_iv2")):
        design, _instruments, outputs = scaled_rows[kind]
        operator, info = ols_operator(design, outputs)
        trunc_scaled, trunc_info = svd_truncated_solve(design, outputs, TRUNCATION_REL)
        trunc_info["backward_rel"] = normal_residual(design, trunc_scaled, outputs)
        out[key] = _restore(operator, rank, kappa)
        out[f"{key}_trunc"] = _restore(trunc_scaled, rank, kappa)
        out[f"{key}_info"] = info
        out[f"{key}_trunc_info"] = trunc_info
    _store_iv_family(out, rank, kappa, scaled_rows["iv"], "iv")
    design, instruments, outputs = scaled_rows["iv2"]
    lstsq_op, projected, lstsq_info = iv2_lstsq_operator(design, instruments, outputs)
    qr_op, qr_projected, qr_info = iv2_qr_operator(design, instruments, outputs)
    trunc_scaled, trunc_info = svd_truncated_solve(qr_projected, outputs, TRUNCATION_REL)
    trunc_info["backward_rel"] = normal_residual(qr_projected, trunc_scaled, outputs)
    out["iv2_lstsq"] = _restore(lstsq_op, rank, kappa)
    out["iv2_qr"] = _restore(qr_op, rank, kappa)
    out["iv2_trunc"] = _restore(trunc_scaled, rank, kappa)
    out["iv2_lstsq_info"] = lstsq_info
    out["iv2_qr_info"] = qr_info
    out["iv2_trunc_info"] = trunc_info
    out["iv2_projected"] = projected
    out["iv2_qr_projected"] = qr_projected
    out["iv2_r2"] = float(_median_r2(design, projected))
    _store_iv_family(out, rank, kappa, scaled_rows["iv_unified"], "iv_unified")
    return out


def _store_iv_family(out: dict, rank: int, kappa: float, rows, prefix: str) -> None:
    design, instruments, outputs = rows
    solved, solved_info = iv_solve_operator(design, instruments, outputs)
    qr_op, qr_info = iv_qr_operator(design, instruments, outputs)
    trunc_scaled, trunc_info = svd_truncated_solve(qr_info["lhs"], qr_info["rhs"], TRUNCATION_REL)
    trunc_info["backward_rel"] = equation_residual(qr_info["lhs"], trunc_scaled, qr_info["rhs"])
    out[f"{prefix}_solve"] = _restore(solved, rank, kappa)
    out[f"{prefix}_qr"] = _restore(qr_op, rank, kappa)
    out[f"{prefix}_trunc"] = _restore(trunc_scaled, rank, kappa)
    out[f"{prefix}_solve_info"] = solved_info
    out[f"{prefix}_qr_info"] = qr_info
    out[f"{prefix}_trunc_info"] = trunc_info


_KINDS = ("ols_full", "ols_iv", "ols_iv2", "iv", "iv2", "iv_unified")


def _decomposition(rank, saved, ols_iv, ols_iv2, iv_unified):
    rows = []
    arrays = {}
    identity_values = {}
    jobs = (
        ("iv", saved["iv"], ols_iv, saved["ols"]),
        ("iv2", saved["iv2"], ols_iv2, saved["ols"]),
        ("iv_unified", iv_unified, ols_iv2, saved["ols"]),
    )
    for method, estimate, window, full in jobs:
        gaps = three_gaps(estimate, window, full)
        residual = identity_residual(gaps)
        identity_values[method] = float(np.max(np.abs(residual)))
        references = {
            "method_minus_ols_full": full,
            "method_minus_ols_window": window,
            "ols_window_minus_ols_full": full,
        }
        for name, diff in gaps.items():
            views = block_views(diff, rank)
            for block, array in views.items():
                arrays[f"r{rank:02d}_{method}_{name}_{block}"] = np.asarray(array, dtype=np.float64)
            for record in block_records(diff, references[name], rank, REL_FLOOR):
                rows.append({
                    "r": rank,
                    "method": method,
                    "comparison": name,
                    "identity_max_abs": identity_values[method],
                    **record,
                })
    return rows, arrays, identity_values


def _moments(rank, original, operators) -> list[dict]:
    rows = []
    specs = (
        ("iv", "iv", "ols_iv", original["iv"][1]),
        ("iv2", "iv2", "ols_iv2", original["iv2"][1]),
        ("iv_unified", "iv_unified", "ols_iv2", original["iv_unified"][1]),
        ("ols", "full", "ols_full", original["ols_full"][0]),
        ("ols", "iv", "ols_iv", original["ols_iv"][0]),
        ("ols", "iv2", "ols_iv2", original["ols_iv2"][0]),
    )
    for method, window, ols_key, instruments in specs:
        if method == "ols":
            design, _instruments, outputs = original[{"full": "ols_full", "iv": "ols_iv", "iv2": "ols_iv2"}[window]]
        elif method == "iv":
            design, _instruments, outputs = original["iv"]
        elif method == "iv2":
            design, _instruments, outputs = original["iv2"]
        else:
            design, _instruments, outputs = original["iv_unified"]
        residual = outputs - design @ operators[ols_key]
        record = moment_record(instruments, design, residual)
        record.update({
            "r": rank,
            "method": method,
            "window": window,
            "residual_source": "same_window_ols",
            "scale_note": "feature_scale_dependent_not_population_endogeneity",
        })
        rows.append(record)
    return rows


def _solver_table(rank, kappa, original, scaled_rows, operators, saved):
    rows = []
    arrays = {}
    spectra = {}
    window_ols = {
        "full": operators["ols_full"],
        "iv": operators["ols_iv"],
        "iv2": operators["ols_iv2"],
        "unified": operators["ols_iv2"],
    }
    formal_op = {
        "ols": {"full": operators["ols_full"], "iv": operators["ols_iv"], "iv2": operators["ols_iv2"]},
        "iv": operators["iv_solve"],
        "iv2": operators["iv2_lstsq"],
        "iv_unified": operators["iv_unified_solve"],
    }
    jobs = [
        ("ols", "full", "lstsq", "OLS_REFERENCE", "ols_full", "ols_full", operators["ols_full"], operators["ols_full_info"]),
        ("ols", "iv", "lstsq", "OLS_REFERENCE", "ols_iv", "ols_iv", operators["ols_iv"], operators["ols_iv_info"]),
        ("ols", "iv2", "lstsq", "OLS_REFERENCE", "ols_iv2", "ols_iv2", operators["ols_iv2"], operators["ols_iv2_info"]),
        ("ols", "full", "trunc", "TRUNCATION_DIAGNOSTIC", "ols_full", "ols_full", operators["ols_full_trunc"], operators["ols_full_trunc_info"]),
        ("ols", "iv", "trunc", "TRUNCATION_DIAGNOSTIC", "ols_iv", "ols_iv", operators["ols_iv_trunc"], operators["ols_iv_trunc_info"]),
        ("ols", "iv2", "trunc", "TRUNCATION_DIAGNOSTIC", "ols_iv2", "ols_iv2", operators["ols_iv2_trunc"], operators["ols_iv2_trunc_info"]),
        ("iv", "iv", "solve", "FORMAL_SOLVER", "iv_solve", "iv", operators["iv_solve"], operators["iv_solve_info"]),
        ("iv", "iv", "qr", "QR_EQUIVALENT", "iv_qr", "iv", operators["iv_qr"], operators["iv_qr_info"]),
        ("iv", "iv", "trunc", "TRUNCATION_DIAGNOSTIC", "iv_trunc", "iv", operators["iv_trunc"], operators["iv_trunc_info"]),
        ("iv2", "iv2", "lstsq", "FORMAL_SOLVER", "iv2_lstsq", "iv2", operators["iv2_lstsq"], operators["iv2_lstsq_info"]),
        ("iv2", "iv2", "qr", "QR_EQUIVALENT", "iv2_qr", "iv2", operators["iv2_qr"], operators["iv2_qr_info"]),
        ("iv2", "iv2", "trunc", "TRUNCATION_DIAGNOSTIC", "iv2_trunc", "iv2", operators["iv2_trunc"], operators["iv2_trunc_info"]),
        ("iv_unified", "unified", "solve", "FORMAL_SOLVER", "iv_unified_solve", "iv_unified", operators["iv_unified_solve"], operators["iv_unified_solve_info"]),
        ("iv_unified", "unified", "qr", "QR_EQUIVALENT", "iv_unified_qr", "iv_unified", operators["iv_unified_qr"], operators["iv_unified_qr_info"]),
        ("iv_unified", "unified", "trunc", "TRUNCATION_DIAGNOSTIC", "iv_unified_trunc", "iv_unified", operators["iv_unified_trunc"], operators["iv_unified_trunc_info"]),
    ]
    cached_spectra = _spectra(rank, scaled_rows, operators, spectra)
    sensitive = _sensitivity(rank, operators, cached_spectra)
    for method, window, solver, role, op_key, design_kind, estimate, info in jobs:
        design_o, instruments_o, outputs_o = original[design_kind if design_kind != "iv_unified" else "iv_unified"]
        if method == "ols":
            design_kind_name = {"full": "ols_full", "iv": "ols_iv", "iv2": "ols_iv2"}[window]
            design_o, instruments_o, outputs_o = original[design_kind_name]
            instruments_o = design_o
        reference = formal_op["ols"][window] if method == "ols" else formal_op[method]
        if method == "ols" and window == "full":
            reference = operators["ols_full"]
        row = _blank_solver_row()
        row.update({
            "r": rank,
            "method": method,
            "window": window,
            "solver": solver,
            "role": role,
            "kappa": kappa,
            "error": "" if info is None else str(info.get("error", "")),
            "lstsq_rank": float("nan") if info is None or "lstsq_rank" not in info else info["lstsq_rank"],
            "r_diag_ratio": float("nan") if info is None else info.get("r_diag_ratio", float("nan")),
            "first_stage_r2_median": operators["iv2_r2"] if method == "iv2" else float("nan"),
            "truncation_rel_cutoff": TRUNCATION_REL if role == "TRUNCATION_DIAGNOSTIC" else float("nan"),
            "truncation_kept": float("nan") if info is None or "kept" not in info else info["kept"],
            "backward_rel": float("nan") if info is None else info.get("backward_rel", float("nan")),
        })
        _fill_spectra(row, method, window, solver, cached_spectra)
        _fill_operator_gaps(row, estimate, reference, window_ols[window], rank, design_o, outputs_o)
        row["equivalence_reliable"] = _equivalence_for(method, solver, role, row)
        row["solver_sensitive"] = sensitive[method] if method in sensitive and solver in {"solve", "qr", "lstsq"} else 0
        if role == "TRUNCATION_DIAGNOSTIC":
            changed = row["gap_formal_O_rel"]
            kept = int(info["kept"])
            row["solver_sensitive"] = int(
                (not np.isfinite(changed)) or changed > SOLVER_REL_FLAG or kept < int(row["n_cols_design"])
            )
        row["pred_formal_label"] = prediction_label(row["pred_gap_formal_rel"])
        rows.append(row)
        arrays[f"r{rank:02d}_{op_key}"] = np.asarray(estimate, dtype=np.float64)
    arrays[f"r{rank:02d}_saved_ols"] = np.asarray(saved["ols"], dtype=np.float64)
    arrays[f"r{rank:02d}_saved_iv"] = np.asarray(saved["iv"], dtype=np.float64)
    arrays[f"r{rank:02d}_saved_iv2"] = np.asarray(saved["iv2"], dtype=np.float64)
    return rows, arrays, spectra


def _spectra(rank, scaled_rows, operators, spectra):
    cached = {}
    for kind in ("ols_full", "ols_iv", "ols_iv2", "iv", "iv2", "iv_unified"):
        design, instruments, _outputs = scaled_rows[kind]
        if instruments is None:
            instruments = design
        cached[kind] = {
            "W": _store_spectrum(spectra, rank, kind, "W", instruments),
            "Z": _store_spectrum(spectra, rank, kind, "Z", design),
        }
        if kind in {"iv", "iv2", "iv_unified"}:
            cross = instruments.T @ design
            cached[kind]["cross"] = _store_spectrum(spectra, rank, kind, "cross", cross)
    q_factor, _ = np.linalg.qr(scaled_rows["iv"][1], mode="reduced")
    cached["iv"]["QtZ"] = _store_spectrum(spectra, rank, "iv", "QtZ", q_factor.T @ scaled_rows["iv"][0])
    q_factor, _ = np.linalg.qr(scaled_rows["iv_unified"][1], mode="reduced")
    cached["iv_unified"]["QtZ"] = _store_spectrum(spectra, rank, "iv_unified", "QtZ", q_factor.T @ scaled_rows["iv_unified"][0])
    cached["iv2"]["Zhat"] = _store_spectrum(spectra, rank, "iv2", "Zhat", operators["iv2_projected"])
    cached["iv2"]["Zhat_qr"] = _store_spectrum(spectra, rank, "iv2", "Zhat_qr", operators["iv2_qr_projected"])
    denom = float(np.linalg.norm(operators["iv2_projected"]))
    numer = float(np.linalg.norm(operators["iv2_qr_projected"] - operators["iv2_projected"]))
    cached["zhat_gap"] = numer / denom if denom > 0.0 else float("nan")
    return cached


def _store_spectrum(spectra, rank, kind, name, matrix):
    spec = spectrum_record(np.asarray(matrix, dtype=np.float64))
    spectra[f"r{rank:02d}_{kind}_{name}_sv"] = np.asarray(spec["singular_values"], dtype=np.float64)
    values = spec["singular_values"]
    spec = dict(spec)
    spec["smin"] = float(values[-1]) if values.size else float("nan")
    spec["smax"] = float(values[0]) if values.size else float("nan")
    return spec


def _sensitivity(rank, operators, cached):
    del rank
    flags = {}
    iv_gap = _relative(operators["iv_qr"], operators["iv_solve"])
    iv_rank_ok = (
        cached["iv"]["cross"]["numerical_rank"] == cached["iv"]["cross"]["n_cols"]
        and cached["iv"]["QtZ"]["numerical_rank"] == cached["iv"]["QtZ"]["n_cols"]
    )
    flags["iv"] = int((not np.isfinite(iv_gap)) or iv_gap > SOLVER_REL_FLAG or not iv_rank_ok)
    iv2_gap = _relative(operators["iv2_qr"], operators["iv2_lstsq"])
    iv2_rank_ok = cached["iv2"]["Zhat"]["numerical_rank"] == cached["iv2"]["Zhat"]["n_cols"]
    flags["iv2"] = int((not np.isfinite(iv2_gap)) or iv2_gap > SOLVER_REL_FLAG or not iv2_rank_ok)
    unified_gap = _relative(operators["iv_unified_qr"], operators["iv_unified_solve"])
    unified_ok = (
        cached["iv_unified"]["cross"]["numerical_rank"] == cached["iv_unified"]["cross"]["n_cols"]
        and cached["iv_unified"]["QtZ"]["numerical_rank"] == cached["iv_unified"]["QtZ"]["n_cols"]
    )
    flags["iv_unified"] = int((not np.isfinite(unified_gap)) or unified_gap > SOLVER_REL_FLAG or not unified_ok)
    flags["ols"] = 0
    return flags


def _fill_spectra(row, method, window, solver, cached):
    if method == "ols":
        kind = {"full": "ols_full", "iv": "ols_iv", "iv2": "ols_iv2"}[window]
        solve_name = "design"
        solve = cached[kind]["Z"]
        cross = cached[kind]["Z"]
        instruments = cached[kind]["Z"]
        design = cached[kind]["Z"]
    elif method == "iv":
        kind = "iv"
        solve_name = "Q_transpose_Z" if solver in {"qr", "trunc"} else "cross_moment"
        solve = cached[kind]["QtZ"] if solver in {"qr", "trunc"} else cached[kind]["cross"]
        cross = cached[kind]["cross"]
        instruments = cached[kind]["W"]
        design = cached[kind]["Z"]
    elif method == "iv_unified":
        kind = "iv_unified"
        solve_name = "Q_transpose_Z" if solver in {"qr", "trunc"} else "cross_moment"
        solve = cached[kind]["QtZ"] if solver in {"qr", "trunc"} else cached[kind]["cross"]
        cross = cached[kind]["cross"]
        instruments = cached[kind]["W"]
        design = cached[kind]["Z"]
    else:
        kind = "iv2"
        solve_name = "Zhat_qr" if solver in {"qr", "trunc"} else "Zhat_lstsq"
        solve = cached[kind]["Zhat_qr"] if solver in {"qr", "trunc"} else cached[kind]["Zhat"]
        cross = cached[kind]["cross"]
        instruments = cached[kind]["W"]
        design = cached[kind]["Z"]
    row.update({
        "n_rows": design["n_rows"],
        "n_cols_design": design["n_cols"],
        "n_cols_instrument": instruments["n_cols"],
        "solve_matrix": solve_name if method != "ols" or solver == "lstsq" else "design_truncated",
        "zhat_qr_vs_lstsq_rel": cached["zhat_gap"] if method == "iv2" else float("nan"),
    })
    _copy_spec(row, "W", instruments)
    _copy_spec(row, "Z", design)
    _copy_spec(row, "solve", solve)
    _copy_spec(row, "cross", cross)


def _copy_spec(row, prefix, spec):
    key = "solve" if prefix == "solve" else prefix
    row[f"rank_{key}"] = spec["numerical_rank"]
    row[f"cond_{key}"] = spec["cond"]
    row[f"cutoff_{key}"] = spec["svd_cutoff"]
    row[f"smin_{key}"] = spec["smin"]
    row[f"smax_{key}"] = spec["smax"]


def _fill_operator_gaps(row, estimate, formal, window_ols, rank, design, outputs):
    formal_records = {item["block"]: item for item in block_records(estimate - formal, formal, rank, REL_FLOOR)}
    window_records = {item["block"]: item for item in block_records(estimate - window_ols, window_ols, rank, REL_FLOOR)}
    for block in BLOCKS:
        row[f"gap_formal_{block}_abs"] = formal_records[block]["diff_abs"]
        row[f"gap_formal_{block}_rel"] = formal_records[block]["diff_rel"]
        row[f"gap_window_ols_{block}_abs"] = window_records[block]["diff_abs"]
        row[f"gap_window_ols_{block}_rel"] = window_records[block]["diff_rel"]
    formal_pred = prediction_gap(design, estimate, formal)
    window_pred = prediction_gap(design, estimate, window_ols)
    y_norm = float(np.linalg.norm(outputs))
    row["pred_gap_formal_abs"] = formal_pred
    row["pred_gap_formal_rel"] = formal_pred / y_norm if np.isfinite(formal_pred) and y_norm > 0.0 else float("nan")
    row["pred_gap_window_ols_abs"] = window_pred
    row["pred_gap_window_ols_rel"] = window_pred / y_norm if np.isfinite(window_pred) and y_norm > 0.0 else float("nan")
    row["fit_residual_rel"] = fit_residual_rel(design, estimate, outputs)


def _equivalence_for(method, solver, role, row) -> int:
    """IV 还要求正规方程矩阵满数值秩。IV2 只看 W 和实际的 Zhat。"""
    del solver
    if role != "QR_EQUIVALENT":
        return 0
    base = equivalence_reliable(
        int(row["rank_W"]), int(row["n_cols_instrument"]), int(row["rank_solve"]), int(row["n_cols_design"])
    )
    if method in {"iv", "iv_unified"} and int(row["rank_cross"]) != int(row["n_cols_design"]):
        return 0
    return base


def _summary_row(rank, kappa, saved, operators, sliced, identity_values, solver_rows, moment_rows, published):
    iv_full = block_records(saved["iv"] - saved["ols"], saved["ols"], rank, REL_FLOOR)
    iv_window = block_records(saved["iv"] - operators["ols_iv"], operators["ols_iv"], rank, REL_FLOOR)
    iv_shift = block_records(operators["ols_iv"] - saved["ols"], saved["ols"], rank, REL_FLOOR)
    iv2_full = block_records(saved["iv2"] - saved["ols"], saved["ols"], rank, REL_FLOOR)
    iv2_window = block_records(saved["iv2"] - operators["ols_iv2"], operators["ols_iv2"], rank, REL_FLOOR)
    iv2_shift = block_records(operators["ols_iv2"] - saved["ols"], saved["ols"], rank, REL_FLOOR)
    unified = block_records(operators["iv_unified_solve"] - operators["ols_iv2"], operators["ols_iv2"], rank, REL_FLOOR)
    by_block = lambda records, block, field: next(item[field] for item in records if item["block"] == block)
    qr_iv = _solver(solver_rows, "iv", "qr")
    qr_iv2 = _solver(solver_rows, "iv2", "qr")
    trunc_iv = _solver(solver_rows, "iv", "trunc")
    trunc_iv2 = _solver(solver_rows, "iv2", "trunc")
    formal_iv = _solver(solver_rows, "iv", "solve")
    formal_iv2 = _solver(solver_rows, "iv2", "lstsq")
    moment_iv = next(item for item in moment_rows if item["method"] == "iv")
    moment_iv2 = next(item for item in moment_rows if item["method"] == "iv2")
    row = {
        "r": rank,
        "kappa": kappa,
        "identity_max_abs": max(identity_values.values()),
        "iv_saved_vs_refit_rel": _relative(operators["iv_solve"], saved["iv"]),
        "iv2_saved_vs_refit_rel": _relative(operators["iv2_lstsq"], saved["iv2"]),
        "ols_saved_vs_refit_rel": _relative(operators["ols_full"], saved["ols"]),
        "iv_window_ols_vs_sliced_fit_rel": _relative(operators["ols_iv"], sliced["iv"]),
        "iv2_window_ols_vs_sliced_fit_rel": _relative(operators["ols_iv2"], sliced["iv2"]),
        "iv_equivalence_reliable": qr_iv["equivalence_reliable"],
        "iv2_equivalence_reliable": qr_iv2["equivalence_reliable"],
        "iv_solver_sensitive": qr_iv["solver_sensitive"],
        "iv2_solver_sensitive": qr_iv2["solver_sensitive"],
        "iv_solver_gap_rel": qr_iv["gap_formal_O_rel"],
        "iv2_solver_gap_rel": qr_iv2["gap_formal_O_rel"],
        "iv_solver_pred_rel": qr_iv["pred_gap_formal_rel"],
        "iv2_solver_pred_rel": qr_iv2["pred_gap_formal_rel"],
        "iv_solver_pred_label": qr_iv["pred_formal_label"],
        "iv2_solver_pred_label": qr_iv2["pred_formal_label"],
        "iv_backward_rel": formal_iv["backward_rel"],
        "iv2_backward_rel": formal_iv2["backward_rel"],
        "iv_qr_backward_rel": qr_iv["backward_rel"],
        "iv2_qr_backward_rel": qr_iv2["backward_rel"],
        "iv_trunc_gap_rel": trunc_iv["gap_formal_O_rel"],
        "iv2_trunc_gap_rel": trunc_iv2["gap_formal_O_rel"],
        "iv_trunc_pred_rel": trunc_iv["pred_gap_formal_rel"],
        "iv2_trunc_pred_rel": trunc_iv2["pred_gap_formal_rel"],
        "iv_cond_cross": formal_iv["cond_cross"],
        "iv_cond_QtZ": qr_iv["cond_solve"],
        "iv_rank_cross": formal_iv["rank_cross"],
        "iv_rank_QtZ": qr_iv["rank_solve"],
        "iv_rank_W": formal_iv["rank_W"],
        "iv_cols": formal_iv["n_cols_design"],
        "iv2_cond_cross": formal_iv2["cond_cross"],
        "iv2_cond_Zhat": formal_iv2["cond_solve"],
        "iv2_cond_Zhat_qr": qr_iv2["cond_solve"],
        "iv2_rank_Zhat": formal_iv2["rank_solve"],
        "iv2_rank_W": formal_iv2["rank_W"],
        "iv2_cols_W": formal_iv2["n_cols_instrument"],
        "iv2_cols": formal_iv2["n_cols_design"],
        "iv2_first_stage_r2_median": formal_iv2["first_stage_r2_median"],
        "iv2_zhat_qr_vs_lstsq_rel": formal_iv2["zhat_qr_vs_lstsq_rel"],
        "ols_full_cond_Z": _solver(solver_rows, "ols", "lstsq", "full")["cond_solve"],
        "ols_full_normal_rel": _solver(solver_rows, "ols", "lstsq", "full")["backward_rel"],
        "ols_iv_normal_rel": _solver(solver_rows, "ols", "lstsq", "iv")["backward_rel"],
        "ols_iv2_normal_rel": _solver(solver_rows, "ols", "lstsq", "iv2")["backward_rel"],
        "iv_cross_per_sample": moment_iv["cross_per_sample_frob"],
        "iv2_cross_per_sample": moment_iv2["cross_per_sample_frob"],
        "iv_cross_raw": moment_iv["cross_raw_frob"],
        "iv2_cross_raw": moment_iv2["cross_raw_frob"],
    }
    for block in BLOCKS:
        row[f"iv_full_{block}_abs"] = by_block(iv_full, block, "diff_abs")
        row[f"iv_full_{block}_rel"] = by_block(iv_full, block, "diff_rel")
        row[f"iv_window_{block}_abs"] = by_block(iv_window, block, "diff_abs")
        row[f"iv_window_{block}_rel"] = by_block(iv_window, block, "diff_rel")
        row[f"iv_shift_{block}_abs"] = by_block(iv_shift, block, "diff_abs")
        row[f"iv_shift_{block}_rel"] = by_block(iv_shift, block, "diff_rel")
        row[f"iv2_full_{block}_abs"] = by_block(iv2_full, block, "diff_abs")
        row[f"iv2_full_{block}_rel"] = by_block(iv2_full, block, "diff_rel")
        row[f"iv2_window_{block}_abs"] = by_block(iv2_window, block, "diff_abs")
        row[f"iv2_window_{block}_rel"] = by_block(iv2_window, block, "diff_rel")
        row[f"iv2_shift_{block}_abs"] = by_block(iv2_shift, block, "diff_abs")
        row[f"iv2_shift_{block}_rel"] = by_block(iv2_shift, block, "diff_rel")
        row[f"iv_unified_{block}_abs"] = by_block(unified, block, "diff_abs")
        row[f"iv_unified_{block}_rel"] = by_block(unified, block, "diff_rel")
    if published:
        row["part1_iv_full_rel"] = float(published["iv_dev_O_rel"])
        row["part1_iv2_full_rel"] = float(published["iv2_dev_O_rel"])
        row["part1_iv_window_rel"] = float(published["diagnostic_iv_vs_window_ols_rel"])
        row["part1_iv2_window_rel"] = float(published["diagnostic_iv2_vs_window_ols_rel"])
    return row


def _solver(rows, method, solver, window=None):
    for row in rows:
        if row["method"] == method and row["solver"] == solver and (window is None or row["window"] == window):
            return row
    raise KeyError((method, solver, window))


def _blank_solver_row() -> dict:
    row = {}
    for name in SOLVER_COLUMNS:
        row[name] = float("nan")
    return row


def _check_counts(spec, original, n_traj, problems) -> None:
    for kind, key in (
        ("ols_full", "ols_full"),
        ("iv", "iv"),
        ("iv2", "iv2"),
        ("iv_unified", "iv_unified"),
    ):
        expected = spec[kind if kind != "iv_unified" else "iv_unified"]["n_rows"] * n_traj
        actual = int(original[key][0].shape[0])
        if actual != expected:
            problems.append(f"{kind} rows {actual} != {expected}")


def _restore(operator: np.ndarray, rank: int, kappa: float) -> np.ndarray:
    if not np.isfinite(operator).all():
        return np.asarray(operator, dtype=np.float64)
    return _to_original(operator, rank, kappa)


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        return float("nan")
    denom = float(np.linalg.norm(right))
    if denom < REL_FLOOR:
        return float("nan")
    return float(np.linalg.norm(left - right) / denom)


def _lstsq(matrix: np.ndarray, outputs: np.ndarray):
    operator, _residual, rank, _singular = np.linalg.lstsq(matrix, outputs, rcond=None)
    error = "" if int(rank) >= matrix.shape[1] else "singular design"
    return operator, int(rank), error


def iv_linear(lhs, rhs, width, out_cols):
    try:
        operator = np.linalg.solve(lhs, rhs)
        error = ""
    except np.linalg.LinAlgError as exc:
        operator = np.full((width, out_cols), np.nan)
        error = str(exc)
    return operator, {"error": error, "backward_rel": equation_residual(lhs, operator, rhs)}


def _diag_ratio(r_factor: np.ndarray) -> float:
    diag = np.abs(np.diag(np.asarray(r_factor, dtype=np.float64)))
    if diag.size == 0 or float(diag.max()) == 0.0:
        return float("nan")
    return float(diag.min() / diag.max())


SOLVER_COLUMNS = (
    "r", "method", "window", "solver", "role", "kappa", "n_rows", "n_cols_design", "n_cols_instrument",
    "rank_W", "cond_W", "cutoff_W", "smin_W", "smax_W",
    "rank_Z", "cond_Z", "cutoff_Z", "smin_Z", "smax_Z",
    "rank_solve", "cond_solve", "cutoff_solve", "smin_solve", "smax_solve", "solve_matrix",
    "rank_cross", "cond_cross", "cutoff_cross", "smin_cross", "smax_cross",
    "backward_rel", "fit_residual_rel", "equivalence_reliable", "truncation_rel_cutoff", "truncation_kept", "lstsq_rank",
    "first_stage_r2_median", "zhat_qr_vs_lstsq_rel", "r_diag_ratio",
    "gap_formal_O_abs", "gap_formal_O_rel", "gap_formal_A_abs", "gap_formal_A_rel",
    "gap_formal_H_abs", "gap_formal_H_rel", "gap_formal_c_abs", "gap_formal_c_rel",
    "gap_window_ols_O_abs", "gap_window_ols_O_rel", "gap_window_ols_A_abs", "gap_window_ols_A_rel",
    "gap_window_ols_H_abs", "gap_window_ols_H_rel", "gap_window_ols_c_abs", "gap_window_ols_c_rel",
    "pred_gap_formal_abs", "pred_gap_formal_rel", "pred_gap_window_ols_abs", "pred_gap_window_ols_rel",
    "pred_formal_label", "solver_sensitive", "error",
)
