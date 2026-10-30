"""POD 基、投影，以及 σ² 的两种估计。"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy import integrate, optimize


def snapshot_matrix(snapshots_list: list[np.ndarray]) -> np.ndarray:
    """每条轨迹形状 (K+1, N)，按列拼成 (N, K_tot)。不中心化。"""
    columns = []
    for item in snapshots_list:
        state = np.asarray(item, dtype=np.float64)
        if state.ndim != 2:
            raise ValueError("each trajectory must have shape (K+1, N)")
        columns.append(state.T)
    return np.hstack(columns)


def pod_basis(snapshots_list: list[np.ndarray], r: int) -> tuple[np.ndarray, np.ndarray]:
    """返回 V (N, r) 和全部奇异值。"""
    matrix = snapshot_matrix(snapshots_list)
    if r < 1 or r > min(matrix.shape):
        raise ValueError("r is outside the available rank")
    left, values, _ = np.linalg.svd(matrix, full_matrices=False)
    return left[:, : int(r)], values


def project(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """(K+1, N) 投影到 (K+1, r)。"""
    return np.asarray(U, dtype=np.float64) @ np.asarray(V, dtype=np.float64)


def mp_median(beta: float) -> float:
    """Marchenko–Pastur 密度的中位数。β = N/K_tot，且 β < 1。"""
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must lie in (0, 1)")
    root = float(np.sqrt(beta))
    low = (1.0 - root) ** 2
    high = (1.0 + root) ** 2

    def density(lam: float) -> float:
        span = (high - lam) * (lam - low)
        if span <= 0.0:
            return 0.0
        return float(np.sqrt(span) / (2.0 * np.pi * beta * lam))

    def centered(point: float) -> float:
        value, _ = integrate.quad(density, low, point, epsabs=1e-12, limit=200)
        return value - 0.5

    left = low + (high - low) * 1e-10
    right = high - (high - low) * 1e-10
    return float(optimize.brentq(centered, left, right))


def _covariance_eigs(matrix: np.ndarray) -> np.ndarray:
    """(1/K) S S^T 的特征值，从大到小。"""
    singular = np.linalg.svd(matrix, compute_uv=False)
    return singular**2 / float(matrix.shape[1])


def _sigma2_from(matrix: np.ndarray, method: str) -> float:
    n, k_tot = matrix.shape
    if method == "trailing_median":
        return float(np.median(_covariance_eigs(matrix)[n // 2 :]))
    if method == "gavish_donoho":
        beta = n / float(k_tot)
        sigma = np.median(np.linalg.svd(matrix, compute_uv=False))
        return float((sigma / np.sqrt(k_tot * mp_median(beta))) ** 2)
    raise ValueError(method)


def estimate_sigma2(noisy_snapshots_list, method: str = "trailing_median", clean_snapshots_list=None):
    """估计 σ²。要求 K_tot >= 2N。干净谱的诊断只在传入干净快照时计算。"""
    matrix = snapshot_matrix(list(noisy_snapshots_list))
    n, k_tot = matrix.shape
    if k_tot < 2 * n:
        raise ValueError("K_tot must be at least 2N")
    sigma2 = _sigma2_from(matrix, method)
    diagnostics: dict = {
        "method": method,
        "N": int(n),
        "K_tot": int(k_tot),
        "beta": float(n / k_tot),
        "clean_half_eigenvalue_over_sigma2": None,
    }
    if clean_snapshots_list is not None:
        clean = _covariance_eigs(snapshot_matrix(list(clean_snapshots_list)))
        diagnostics["clean_half_eigenvalue_over_sigma2"] = float(clean[n // 2 - 1] / sigma2)
    return sigma2, diagnostics
