"""OLS、ridge、MC、MC-PSD、IV、IV2 和 oracle。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from noisyopinf.features import (
    corrected_features,
    diag_quad_rows,
    exponents,
    features,
    hermite_monomial_sums,
)


@dataclass
class FitResult:
    """一次拟合的原坐标算子和诊断。"""

    O: np.ndarray
    method: str
    n_pairs: int
    kappa: float
    lam: float | None
    diagnostics: dict
    fit_seconds: float


def split_O(O: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 (p, r) 算子拆成 A、H、c。"""
    quad_count = r * (r + 1) // 2
    return O[:r].T.copy(), O[r:r + quad_count].T.copy(), O[-1].copy()


def fit(
    trajs: Sequence[np.ndarray],
    method: str,
    sigma2: float | None = None,
    lam: float | None = None,
    correct_intercept: bool = True,
) -> FitResult:
    """拟合一个估计器。求解失败时返回全 NaN，不抛出 LinAlgError。"""
    started = perf_counter()
    states, rank = _validated_trajs(trajs)
    kappa = _peak_abs(states)
    scaled = [traj / kappa for traj in states]
    variance = _scaled_variance(method, sigma2, kappa)
    operator, diagnostics, n_pairs = _dispatch(
        method, scaled, variance, lam, correct_intercept
    )
    if np.isfinite(operator).all():
        operator = _to_original(operator, rank, kappa)
    return FitResult(
        O=operator,
        method=method,
        n_pairs=n_pairs,
        kappa=kappa,
        lam=lam,
        diagnostics=diagnostics,
        fit_seconds=perf_counter() - started,
    )


def _validated_trajs(trajs: Sequence[np.ndarray]) -> tuple[list[np.ndarray], int]:
    if len(trajs) == 0:
        raise ValueError("trajs is empty")
    checked: list[np.ndarray] = []
    rank: int | None = None
    for traj in trajs:
        array = np.asarray(traj, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
            raise ValueError("each trajectory must have shape (T, r) with T >= 2")
        if not np.isfinite(array).all():
            raise ValueError("trajectory contains non-finite values")
        if rank is None:
            rank = int(array.shape[1])
        elif array.shape[1] != rank:
            raise ValueError("trajectories have different state dimensions")
        checked.append(array)
    return checked, int(rank)


def _peak_abs(trajs: Sequence[np.ndarray]) -> float:
    peak = max(float(np.max(np.abs(traj))) for traj in trajs)
    if peak == 0.0 or not np.isfinite(peak):
        return 1.0
    return peak


def _pair_states(trajs: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return np.vstack([traj[:-1] for traj in trajs]), np.vstack([traj[1:] for traj in trajs])


def _mc_moments(
    inputs: np.ndarray, outputs: np.ndarray, variance: float
) -> tuple[np.ndarray, np.ndarray]:
    encoded = corrected_features(inputs, variance)
    powers = exponents(inputs.shape[1])
    width = powers.shape[0]
    stacked = (powers[:, None, :] + powers[None, :, :]).reshape(width * width, -1)
    unique, inverse = np.unique(stacked, axis=0, return_inverse=True)
    sums = hermite_monomial_sums(inputs, variance, unique)
    gram = sums[inverse].reshape(width, width)
    return gram, encoded.T @ outputs


def _solve_moments(gram: np.ndarray, cross: np.ndarray) -> tuple[np.ndarray, dict]:
    eigenvalues = np.linalg.eigvalsh(0.5 * (gram + gram.T))
    diagnostics: dict = {
        "gram_min_eig": float(eigenvalues[0]),
        "gram_cond": float(np.linalg.cond(gram)),
        "n_clipped": 0,
    }
    try:
        return np.linalg.solve(gram, cross), diagnostics
    except np.linalg.LinAlgError as exc:
        diagnostics["error"] = str(exc)
        return np.full_like(cross, np.nan), diagnostics


def _to_original(operator: np.ndarray, rank: int, kappa: float) -> np.ndarray:
    restored = np.array(operator, dtype=np.float64, copy=True)
    quad_count = rank * (rank + 1) // 2
    restored[rank:rank + quad_count] /= kappa
    restored[-1] *= kappa
    return restored


def _scaled_variance(method: str, sigma2: float | None, kappa: float) -> float:
    """需要 σ² 的估计器在缩放坐标中使用 σ²/κ²。"""
    if method in {"mc", "mc_psd", "iv", "iv2"} and sigma2 is None:
        raise ValueError(f"{method} requires sigma2")
    if sigma2 is None:
        return 0.0
    variance = float(sigma2)
    if variance < 0.0 or not np.isfinite(variance):
        raise ValueError("sigma2 must be finite and non-negative")
    return variance / kappa**2


def _dispatch(method, trajs, variance, lam, correct_intercept):
    if method in {"ols", "oracle"}:
        return _fit_ols(trajs)
    if method == "ridge":
        return _fit_ridge(trajs, lam)
    if method == "mc":
        return _fit_mc(trajs, variance, psd=False)
    if method == "mc_psd":
        return _fit_mc(trajs, variance, psd=True)
    if method in {"iv", "iv2"}:
        return _fit_iv(trajs, variance, method, correct_intercept)
    raise ValueError(f"unknown method {method}")


def _fit_ols(trajs: Sequence[np.ndarray]):
    inputs, outputs = _pair_states(trajs)
    design = features(inputs)
    operator, _, rank, _ = np.linalg.lstsq(design, outputs, rcond=None)
    diagnostics: dict = {"n_clipped": 0, "gram_cond": float(np.linalg.cond(design.T @ design))}
    if rank < design.shape[1]:
        diagnostics["error"] = "singular design"
        operator = np.full((design.shape[1], outputs.shape[1]), np.nan)
    return operator, diagnostics, int(inputs.shape[0])


def _fit_ridge(trajs: Sequence[np.ndarray], lam: float | None):
    if lam is None:
        raise ValueError("ridge requires lam")
    inputs, outputs = _pair_states(trajs)
    design = features(inputs)
    width = design.shape[1]
    penalty = np.full(width, np.sqrt(float(lam)), dtype=np.float64)
    penalty[-1] = 0.0
    augmented_z = np.vstack([design, np.diag(penalty)])
    augmented_y = np.vstack([outputs, np.zeros((width, outputs.shape[1]))])
    operator, _, rank, _ = np.linalg.lstsq(augmented_z, augmented_y, rcond=None)
    diagnostics: dict = {"n_clipped": 0, "lam": float(lam)}
    if rank < width:
        diagnostics["error"] = "singular design"
        operator = np.full((width, outputs.shape[1]), np.nan)
    return operator, diagnostics, int(inputs.shape[0])


def _fit_mc(trajs: Sequence[np.ndarray], variance: float, psd: bool):
    inputs, outputs = _pair_states(trajs)
    gram, cross = _mc_moments(inputs, outputs, variance)
    if psd:
        operator, diagnostics = _solve_psd(gram, cross)
    else:
        operator, diagnostics = _solve_moments(gram, cross)
    return operator, diagnostics, int(inputs.shape[0])


def _solve_psd(gram: np.ndarray, cross: np.ndarray):
    """把 Gram 特征值截断到 1e-8 λ_max 后再求解。"""
    eigenvalues, vectors = np.linalg.eigh(0.5 * (gram + gram.T))
    peak = float(eigenvalues[-1])
    diagnostics: dict = {
        "gram_min_eig": float(eigenvalues[0]),
        "gram_cond": float(np.linalg.cond(gram)),
        "n_clipped": 0,
    }
    if peak <= 0.0:
        diagnostics["error"] = "nonpositive spectrum"
        return np.full_like(cross, np.nan), diagnostics
    floor = 1e-8 * peak
    diagnostics["n_clipped"] = int(np.sum(eigenvalues < floor))
    repaired = (vectors * np.maximum(eigenvalues, floor)) @ vectors.T
    try:
        return np.linalg.solve(repaired, cross), diagnostics
    except np.linalg.LinAlgError as exc:
        diagnostics["error"] = str(exc)
        return np.full_like(cross, np.nan), diagnostics


def _fit_iv(trajs, variance: float, method: str, correct_intercept: bool):
    design, instruments, outputs = _instrument_rows(trajs, method)
    diagnostics = {
        "n_clipped": 0,
        "instrument_cond": float(np.linalg.cond(instruments.T @ design)),
        "first_stage_r2_median": _median_r2(design, _project(instruments, design)),
    }
    operator = _iv_operator(design, instruments, outputs, method)
    if operator is None:
        diagnostics["error"] = "singular instruments"
        operator = np.full((design.shape[1], outputs.shape[1]), np.nan)
    elif correct_intercept:
        operator = _correct_intercept(operator, variance)
    return operator, diagnostics, int(design.shape[0])


def _instrument_rows(trajs, method: str):
    rows = [_one_instrument(traj, method) for traj in trajs]
    rows = [item for item in rows if item[0].shape[0] > 0]
    if not rows:
        raise ValueError("no instrument rows")
    return (
        np.vstack([item[0] for item in rows]),
        np.vstack([item[1] for item in rows]),
        np.vstack([item[2] for item in rows]),
    )


def _one_instrument(traj: np.ndarray, method: str):
    if method == "iv" and traj.shape[0] >= 3:
        return features(traj[1:-1]), features(traj[:-2]), traj[2:]
    if method == "iv2" and traj.shape[0] >= 4:
        current = features(traj[2:-1])
        lag1 = features(traj[1:-2])[:, :-1]
        lag2 = features(traj[:-3])[:, :-1]
        ones = np.ones((current.shape[0], 1), dtype=np.float64)
        return current, np.hstack([lag1, lag2, ones]), traj[3:]
    width = features(traj[:1]).shape[1]
    return np.zeros((0, width)), np.zeros((0, 1)), np.zeros((0, traj.shape[1]))


def _iv_operator(design, instruments, outputs, method: str):
    try:
        if method == "iv":
            return np.linalg.solve(instruments.T @ design, instruments.T @ outputs)
        projected = _project(instruments, design)
        operator, _, rank, _ = np.linalg.lstsq(projected, outputs, rcond=None)
        if rank < projected.shape[1]:
            return None
        return operator
    except np.linalg.LinAlgError:
        return None


def _project(instruments: np.ndarray, design: np.ndarray) -> np.ndarray:
    coeff, _, _, _ = np.linalg.lstsq(instruments, design, rcond=None)
    return instruments @ coeff


def _median_r2(design: np.ndarray, predicted: np.ndarray) -> float:
    residual = np.sum((design - predicted) ** 2, axis=0)
    centered = design - design.mean(axis=0, keepdims=True)
    total = np.sum(centered**2, axis=0)
    scores = np.ones(design.shape[1], dtype=np.float64)
    usable = total > 0.0
    scores[usable] = 1.0 - residual[usable] / total[usable]
    return float(np.median(scores))


def _correct_intercept(operator: np.ndarray, variance: float) -> np.ndarray:
    """ĉ = ĉ' + σ² Σ_i Ô'[row(i,i), :]。"""
    corrected = np.array(operator, dtype=np.float64, copy=True)
    rank = _rank_from_width(corrected.shape[0])
    corrected[-1] = corrected[-1] + float(variance) * corrected[diag_quad_rows(rank)].sum(axis=0)
    return corrected


def _rank_from_width(width: int) -> int:
    """由 p = r + r(r+1)/2 + 1 反解 r。"""
    for rank in range(1, width):
        if rank + rank * (rank + 1) // 2 + 1 == width:
            return rank
    raise ValueError("operator width is not a quadratic dictionary")
