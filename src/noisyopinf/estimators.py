"""Hermite 矩修正估计器。本文件目前只实现 mc。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from noisyopinf.features import corrected_features, exponents, hermite_monomial_sums


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
) -> FitResult:
    """拟合 mc。奇异 Gram 返回全 NaN，不抛出 LinAlgError。"""
    started = perf_counter()
    if method != "mc":
        raise ValueError("only method='mc' is implemented")
    if sigma2 is None:
        raise ValueError("mc requires sigma2")
    states, rank = _validated_trajs(trajs)
    variance = float(sigma2)
    if variance < 0.0 or not np.isfinite(variance):
        raise ValueError("sigma2 must be finite and non-negative")
    kappa = _peak_abs(states)
    scaled = [traj / kappa for traj in states]
    inputs, outputs = _pair_states(scaled)
    gram, cross = _mc_moments(inputs, outputs, variance / kappa**2)
    operator, diagnostics = _solve_moments(gram, cross)
    if np.isfinite(operator).all():
        operator = _to_original(operator, rank, kappa)
    return FitResult(
        O=operator,
        method="mc",
        n_pairs=int(inputs.shape[0]),
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
