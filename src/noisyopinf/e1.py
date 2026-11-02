"""E1：S1 上 OLS 的 Monte Carlo 均值与 ols_limit 对照。"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
import yaml
from joblib import Parallel, delayed

from noisyopinf.estimators import fit, split_O
from noisyopinf.metrics import err_A_rel, err_H_rel, err_O_rel, err_c_abs, err_c_scaled, specrad_trio, state_rms
from noisyopinf.noise import add_white, sigma_from_rho
from noisyopinf.seeds import rng
from noisyopinf.systems import galerkin_burgers
from noisyopinf.systems.config import project_root
from noisyopinf.theory import ols_limit

RHO_GRID = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)
REQUIRED = {
    "system": "s1",
    "r": 8,
    "rho": [0.02, 0.05, 0.1, 0.2],
    "n_train": [10, 30, 100],
    "estimators": ["ols"],
    "replicates": 200,
    "experiment_id": 1,
}
BLOCKS = ("O", "A", "H", "c")
PILOT = {"rho": 0.05, "n_train": 30, "replicates": 10}


def load_config(path: Path | None = None) -> dict:
    """读取并核对 E1 网格。不允许改 rho、样本数或估计器。"""
    source = project_root() / "configs" / "e1.yaml" if path is None else Path(path)
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    for key, value in REQUIRED.items():
        if loaded.get(key) != value:
            raise ValueError(f"e1.yaml {key} must be {value}")
    if list(loaded.get("rho_grid", [])) != list(RHO_GRID):
        raise ValueError("e1.yaml rho_grid must be the master grid")
    return loaded


def expand_cases(config: dict) -> list[dict]:
    """按 rho、n_train 展开 12 个配置。"""
    cases = []
    for rho in config["rho"]:
        for n_train in config["n_train"]:
            cases.append({
                "rho": float(rho),
                "n_train": int(n_train),
                "rho_index": rho_index(rho),
                "estimator": "ols",
                "replicates": int(config["replicates"]),
            })
    return cases


def rho_index(rho: float) -> int:
    """噪声键里的 rho 下标，取 04 第 1.1 节的主网格。"""
    matches = [index for index, item in enumerate(RHO_GRID) if abs(item - float(rho)) <= 1e-15]
    if len(matches) != 1:
        raise ValueError(f"rho {rho} is outside the master grid")
    return matches[0]


def planned_runs(config: dict) -> int:
    """配置数乘以重复数。正式网格是 2400。"""
    return len(expand_cases(config)) * int(config["replicates"])


def nested_prefix(trajectories: list[np.ndarray], n_train: int) -> list[np.ndarray]:
    """较小的 n_train 必须是已生成序列的前缀。"""
    if n_train < 1 or n_train > len(trajectories):
        raise ValueError("n_train is outside the generated training sequence")
    return trajectories[: int(n_train)]


def noise_keys(rho_index_value: int, replicate: int, traj_index: int) -> tuple[int, ...]:
    """03 的噪声键：(system_id, 4, experiment_id, rho_index, replicate, traj_index)。"""
    return (1, 4, 1, int(rho_index_value), int(replicate), int(traj_index))


def add_training_noise(clean: list[np.ndarray], sigma: float, rho_index_value: int, replicate: int):
    """只给这一次重复的训练轨迹加白噪声。每条轨迹单独一个子流。"""
    return [
        add_white(traj, sigma, rng(*noise_keys(rho_index_value, replicate, index)))
        for index, traj in enumerate(clean)
    ]


def load_s1_train(count: int = 100) -> tuple[list[np.ndarray], list[float]]:
    """按 index 0..count-1 生成 S1 训练轨迹。超界时停止，不改用后续 index。"""
    produced = Parallel(n_jobs=_workers(), backend="loky")(
        delayed(galerkin_burgers.rollout)(index, 1) for index in range(int(count))
    )
    trajectories, peaks = [], []
    for index, (traj, peak) in enumerate(produced):
        if (not np.isfinite(traj).all()) or float(peak) >= 5.0:
            raise RuntimeError(f"S1 training index {index} failed the finite/peak check")
        trajectories.append(np.asarray(traj, dtype=np.float64))
        peaks.append(float(peak))
    return trajectories, peaks


def true_operator() -> np.ndarray:
    """S1 的已验收真算子。"""
    operator, _, _ = galerkin_burgers.true_operator()
    return np.asarray(operator, dtype=np.float64)


def sigma2_of(clean: list[np.ndarray], rho: float) -> float:
    """同一配置的 200 次重复共用这个 sigma²。"""
    return float(sigma_from_rho("s1", rho, clean) ** 2)


def fit_replicate(clean: list[np.ndarray], sigma: float, rho_index_value: int, replicate: int) -> dict:
    """一次 OLS 拟合。失败时保留算子和诊断，不重新抽样。"""
    noisy = add_training_noise(clean, sigma, rho_index_value, replicate)
    result = fit(noisy, "ols")
    finite = bool(np.isfinite(result.O).all())
    return {
        "replicate": int(replicate),
        "O": np.asarray(result.O, dtype=np.float64),
        "n_pairs": int(result.n_pairs),
        "fit_seconds": float(result.fit_seconds),
        "diagnostics": dict(result.diagnostics),
        "finite": finite,
    }


def fit_case(clean: list[np.ndarray], sigma: float, rho_index_value: int, replicates: int) -> list[dict]:
    """一个配置的全部重复。重复编号固定为 0..R-1。"""
    jobs = (
        delayed(fit_replicate)(clean, sigma, rho_index_value, replicate)
        for replicate in range(int(replicates))
    )
    results = Parallel(n_jobs=_workers(), backend="loky")(jobs)
    return sorted(results, key=lambda item: item["replicate"])


def limit_bundle(clean: list[np.ndarray], sigma2: float) -> tuple[np.ndarray, dict]:
    """理论算子只来自 ols_limit。诊断矩必须与它一致。"""
    official = np.asarray(ols_limit(clean, sigma2), dtype=np.float64)
    diagnostics = _gram_diagnostics(clean, sigma2, official)
    return official, diagnostics


def monte_carlo_moments(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """均值和逐元素标准误。任一重复非有限时，均值和标准误都是 NaN。"""
    failed = [index for index, operator in enumerate(stack) if not np.isfinite(operator).all()]
    shape = stack.shape[1:]
    if failed or stack.shape[0] < 2:
        return np.full(shape, np.nan), np.full(shape, np.nan), failed
    mean = np.mean(stack, axis=0)
    error = np.std(stack, axis=0, ddof=1) / np.sqrt(stack.shape[0])
    return mean, error, failed


def block_views(operator: np.ndarray) -> dict[str, np.ndarray]:
    """O 以及 A、H、c。"""
    rank = int(operator.shape[1])
    linear, quadratic, constant = split_O(operator, rank)
    return {"O": np.asarray(operator, dtype=np.float64), "A": linear, "H": quadratic, "c": constant}


def gap_terms(estimate: np.ndarray, limit: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """记录理论差的分子、两个分母和两个比值。分母为 0 时比值是 NaN。"""
    numer = float(np.linalg.norm(np.asarray(estimate) - np.asarray(limit)))
    denom_rel = float(np.linalg.norm(np.asarray(limit) - np.asarray(truth)))
    denom_abs = float(np.linalg.norm(np.asarray(truth)))
    return {
        "gap_numer": numer,
        "gap_denom_rel": denom_rel,
        "gap_denom_abs": denom_abs,
        "theory_gap_rel": _divide(numer, denom_rel),
        "theory_gap_abs": _divide(numer, denom_abs),
        "theory_bias_norm": denom_rel,
        "sim_bias_norm": float(np.linalg.norm(np.asarray(estimate) - np.asarray(truth))),
    }


def delta_method_se(mean: np.ndarray, truth: np.ndarray, se: np.ndarray) -> float:
    """偏差范数的标准误。只使用对角协方差近似，不用 ||SE||_F 代替。"""
    diff = np.asarray(mean, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    scale = float(np.linalg.norm(diff))
    if scale == 0.0 or not np.isfinite(scale) or not np.isfinite(se).all():
        return float("nan")
    return float(np.linalg.norm(np.asarray(se, dtype=np.float64) * diff / scale))


def summarize_case(case: dict, stack: np.ndarray, limit: np.ndarray, truth: np.ndarray, diag: dict) -> dict:
    """一个配置的 O、A、H、c 指标。"""
    mean, se, failed = monte_carlo_moments(stack)
    row = {
        "exp": "e1", "system": "s1", "rho": case["rho"], "rho_index": case["rho_index"],
        "n_train": case["n_train"], "r": 8, "replicates": int(stack.shape[0]),
        "sigma2": case["sigma2"], "estimator": "ols", "n_failed": len(failed),
        "failed_replicates": ",".join(str(item) for item in failed),
        "theory_gram_cond": diag["gram_cond"], "theory_finite": bool(np.isfinite(limit).all()),
        "theory_match_rel": diag["matches_ols_limit_rel"], "seed_pattern": "1,4,1,rho_index,replicate,traj",
    }
    if not np.isfinite(mean).all() or not np.isfinite(limit).all():
        row["status"] = "failed"
        return row
    _fill_blocks(row, mean, se, limit, truth)
    row["status"] = "ok"
    return row


def run_record(case: dict, result: dict, truth: np.ndarray, rms: float) -> dict:
    """一次拟合对应 runs.csv 的一行。预测列留空。"""
    from noisyopinf.io_utils import RUN_COLUMNS
    row = {column: None for column in RUN_COLUMNS}
    finite = result["finite"]
    row.update({
        "exp": "e1", "system": "s1", "rho": case["rho"], "sigma2_true": case["sigma2"],
        "sigma2_source": "na", "r": 8, "n_train": case["n_train"], "n_pairs": result["n_pairs"],
        "basis": "galerkin", "estimator": "ols", "replicate": result["replicate"],
        "seed_keys": f"1,4,1,{case['rho_index']},{result['replicate']},traj",
        "fit_seconds": result["fit_seconds"], "gram_cond": result["diagnostics"].get("gram_cond"),
        "n_clipped": result["diagnostics"].get("n_clipped"),
        "status": "ok" if finite else "failed",
        "error": None if finite else str(result["diagnostics"].get("error", "non-finite operator")),
    })
    _fill_errors(row, result["O"], truth, rms, finite)
    return row


def operator_key(kind: str, rho_index_value: int, n_train: int, replicate: int | None = None) -> str:
    """算子归档键。重复编号固定三位。"""
    stem = f"{kind}_rho{int(rho_index_value)}_n{int(n_train)}"
    if replicate is None:
        return stem
    return f"{stem}_r{int(replicate):03d}"


def extrapolate_seconds(pilot_fit_seconds: float, pilot_reps: int) -> float:
    """用 n_train=30 的拟合时间，按配对数比例估算 12 个配置。"""
    per_fit = float(pilot_fit_seconds) / float(pilot_reps)
    weight = 4 * 200 * (10 + 30 + 100) / 30.0
    return per_fit * weight


def _workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def _divide(numer: float, denom: float) -> float:
    if not np.isfinite(numer) or not np.isfinite(denom) or denom == 0.0:
        return float("nan")
    return float(numer / denom)


def _fill_blocks(row: dict, mean: np.ndarray, se: np.ndarray, limit: np.ndarray, truth: np.ndarray) -> None:
    estimated, limited, reference, errors = (block_views(item) for item in (mean, limit, truth, se))
    for name in BLOCKS:
        terms = gap_terms(estimated[name], limited[name], reference[name])
        for key, value in terms.items():
            row[f"{name}_{key}"] = value
        se_norm = float(np.linalg.norm(errors[name]))
        row[f"{name}_se_frobenius"] = se_norm
        row[f"{name}_mc_se_norm"] = _divide(se_norm, float(np.linalg.norm(reference[name])))
        if name in {"A", "H"}:
            row[f"{name}_bias_se_delta"] = delta_method_se(estimated[name], reference[name], errors[name])


def _fill_errors(row: dict, operator: np.ndarray, truth: np.ndarray, rms: float, finite: bool) -> None:
    if not finite:
        for name in ("err_O_rel", "err_A_rel", "err_H_rel", "err_c_abs", "err_c_scaled", "specrad_hat", "specrad_ref", "specrad_err"):
            row[name] = float("nan")
        return
    row["err_O_rel"] = err_O_rel(operator, truth)
    row["err_A_rel"] = err_A_rel(operator, truth)
    row["err_H_rel"] = err_H_rel(operator, truth)
    row["err_c_abs"] = err_c_abs(operator, truth)
    row["err_c_scaled"] = err_c_scaled(operator, truth, rms)
    row["specrad_hat"], row["specrad_ref"], row["specrad_err"] = specrad_trio(operator, truth)


def _gram_diagnostics(clean, sigma2: float, official: np.ndarray) -> dict:
    import noisyopinf.theory as theory
    from noisyopinf.features import diag_quad_rows, exponents, features
    trajs, _ = theory._validated_trajs(clean)
    variance, _ = theory._check_noise(sigma2, 0.0)
    kappa = theory._peak_abs(trajs)
    inputs, outputs = theory._pair_states([traj / kappa for traj in trajs])
    table = exponents(inputs.shape[1])
    alphas = (table[:, None, :] + table[None, :, :]).reshape(-1, table.shape[1])
    gram = theory._monomial_sums(inputs, variance / kappa**2, alphas).reshape(table.shape[0], -1)
    shift = np.zeros(table.shape[0], dtype=np.float64)
    shift[diag_quad_rows(inputs.shape[1])] = variance / kappa**2
    solved = _solve_or_none(gram, (features(inputs) + shift).T @ outputs)
    restored = None if solved is None else theory._to_original(solved, inputs.shape[1], kappa)
    return {
        "gram_cond": float(np.linalg.cond(gram)),
        "gram_finite": bool(np.isfinite(gram).all()),
        "solve_error": None if solved is not None else "singular Gram",
        "matches_ols_limit_rel": _match(restored, official),
        "kappa": float(kappa),
    }


def _solve_or_none(gram: np.ndarray, cross: np.ndarray):
    try:
        return np.linalg.solve(gram, cross)
    except np.linalg.LinAlgError:
        return None


def _match(restored, official: np.ndarray) -> float:
    if restored is None or not np.isfinite(official).all():
        return float("nan")
    denom = float(np.linalg.norm(official))
    numer = float(np.linalg.norm(restored - official))
    return numer if denom == 0.0 else numer / denom


def training_rms(clean: list[np.ndarray]) -> float:
    return state_rms(np.vstack(clean))


def refuse_existing(path: Path) -> None:
    """结果目录已经存在时停止，避免覆盖。"""
    if Path(path).exists():
        raise FileExistsError(path)
