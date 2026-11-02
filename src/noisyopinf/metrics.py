"""02 第 7 节的算子误差、预测误差和重复汇总。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from noisyopinf.estimators import split_O
from noisyopinf.features import features


def err_O_rel(O_hat: np.ndarray, O_ref: np.ndarray) -> float:
    """Frobenius 相对误差。分母为 0 时返回 NaN。"""
    return _relative_frobenius(np.asarray(O_hat, dtype=np.float64), np.asarray(O_ref, dtype=np.float64))


def err_A_rel(O_hat: np.ndarray, O_ref: np.ndarray) -> float:
    """线性块的 Frobenius 相对误差。"""
    rank = _rank(O_ref)
    return _relative_frobenius(split_O(O_hat, rank)[0], split_O(O_ref, rank)[0])


def err_H_rel(O_hat: np.ndarray, O_ref: np.ndarray) -> float:
    """二次块的 Frobenius 相对误差。"""
    rank = _rank(O_ref)
    return _relative_frobenius(split_O(O_hat, rank)[1], split_O(O_ref, rank)[1])


def err_c_abs(O_hat: np.ndarray, O_ref: np.ndarray) -> float:
    """常数项的绝对 2 范数误差。"""
    rank = _rank(O_ref)
    return float(np.linalg.norm(split_O(O_hat, rank)[2] - split_O(O_ref, rank)[2]))


def err_c_scaled(O_hat: np.ndarray, O_ref: np.ndarray, rms: float) -> float:
    """常数项绝对误差除以 RMS(q)。RMS 为 0 时返回 NaN。"""
    scale = float(rms)
    if scale == 0.0 or not np.isfinite(scale):
        return float("nan")
    return err_c_abs(O_hat, O_ref) / scale


def state_rms(snapshots: np.ndarray) -> float:
    """RMS(q) = sqrt(mean ||q_k||^2 / r)。snapshots: (n, r)。"""
    states = np.asarray(snapshots, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(states * states, axis=1) / states.shape[1])))


def spectral_radius(A: np.ndarray) -> float:
    """矩阵谱半径。"""
    return float(np.max(np.abs(np.linalg.eigvals(np.asarray(A, dtype=np.float64)))))


def specrad_trio(O_hat: np.ndarray, O_ref: np.ndarray) -> tuple[float, float, float]:
    """返回 specrad_hat、specrad_ref 和带符号的差。"""
    rank = _rank(O_ref)
    hat = spectral_radius(split_O(O_hat, rank)[0])
    ref = spectral_radius(split_O(O_ref, rank)[0])
    return hat, ref, hat - ref


def ep_violation(O_hat: np.ndarray, snapshots: np.ndarray) -> float:
    """mean |q^T H s(q)| / mean(||q|| ||H s(q)||)。"""
    states = np.asarray(snapshots, dtype=np.float64)
    rank = states.shape[1]
    quadratic = split_O(O_hat, rank)[1]
    encoded = features(states)
    width = rank * (rank + 1) // 2
    forcing = encoded[:, rank:rank + width] @ quadratic.T
    numerator = np.mean(np.abs(np.sum(states * forcing, axis=1)))
    denominator = np.mean(np.linalg.norm(states, axis=1) * np.linalg.norm(forcing, axis=1))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("nan")
    return float(numerator / denominator)


def rollout_err_h(predicted: np.ndarray, truth: np.ndarray, h: int, V: np.ndarray | None = None) -> float:
    """视界 h 含下标 0 到 h。窗口内出现非有限值时返回 inf。

    V 为 None 时比较降阶状态。给出 V (N, r) 时，truth 是全阶 (T, N)。
    """
    window = slice(0, int(h) + 1)
    pred = np.asarray(predicted, dtype=np.float64)[window]
    ref = np.asarray(truth, dtype=np.float64)[window]
    if not np.isfinite(pred).all() or not np.isfinite(ref).all():
        return float("inf")
    if V is not None:
        pred = pred @ np.asarray(V, dtype=np.float64).T
    return _relative_frobenius(pred, ref)


def rollout_aggregate(errors: Sequence[float], diverged: Sequence[bool]) -> dict[str, float]:
    """中位数保留 inf；均值只含未发散轨迹；发散比例用全部轨迹。"""
    values = np.array([float("inf") if flag else float(err) for err, flag in zip(errors, diverged, strict=True)])
    finite = values[np.isfinite(values)]
    mean_finite = float(np.mean(finite)) if finite.size else float("nan")
    return {
        "rollout_med": float(np.median(values)),
        "rollout_mean_finite": mean_finite,
        "diverged_frac": float(np.mean(np.asarray(diverged, dtype=np.float64))),
    }


def horizon_medians(per_horizon: dict[int, Sequence[float]]) -> dict[str, float]:
    """把视界 10 和 50 的逐轨迹误差收成 rollout_med_h10、rollout_med_h50。"""
    named = {}
    for horizon, values in per_horizon.items():
        named[f"rollout_med_h{horizon}"] = float(np.median(np.asarray(values, dtype=np.float64)))
    return named


def bias_rel(estimates: Sequence[np.ndarray], reference: np.ndarray) -> float:
    """||mean O - O_ref||_F / ||O_ref||_F。"""
    stack = np.stack([np.asarray(item, dtype=np.float64) for item in estimates])
    return _relative_frobenius(np.mean(stack, axis=0), np.asarray(reference, dtype=np.float64))


def sd_rel(estimates: Sequence[np.ndarray], reference: np.ndarray) -> float:
    """样本标准差除以 ||O_ref||_F。重复数小于 2 时返回 NaN。"""
    stack = np.stack([np.asarray(item, dtype=np.float64) for item in estimates])
    if stack.shape[0] < 2:
        return float("nan")
    center = np.mean(stack, axis=0)
    spread = np.sqrt(np.sum(np.linalg.norm(stack - center, axis=(1, 2)) ** 2) / (stack.shape[0] - 1))
    denom = float(np.linalg.norm(np.asarray(reference, dtype=np.float64)))
    if denom == 0.0:
        return float("nan")
    return float(spread / denom)


def block_bias_sd(estimates: Sequence[np.ndarray], reference: np.ndarray) -> dict[str, float]:
    """A、H 两块的 bias_rel 与 sd_rel。"""
    rank = _rank(reference)
    hat_a = [split_O(item, rank)[0] for item in estimates]
    hat_h = [split_O(item, rank)[1] for item in estimates]
    ref_a, ref_h, _ = split_O(reference, rank)
    return {
        "bias_A_rel": bias_rel(hat_a, ref_a),
        "sd_A_rel": sd_rel(hat_a, ref_a),
        "bias_H_rel": bias_rel(hat_h, ref_h),
        "sd_H_rel": sd_rel(hat_h, ref_h),
    }


def iqr(values: Sequence[float]) -> float:
    """四分位距。inf 参与分位数计算。"""
    data = np.asarray(values, dtype=np.float64)
    return float(np.quantile(data, 0.75) - np.quantile(data, 0.25))


def proj_floor(V: np.ndarray, snapshots: np.ndarray) -> float:
    """||V V^T u - u||_F / ||u||_F。snapshots: (T, N)，V: (N, r)。"""
    basis = np.asarray(V, dtype=np.float64)
    truth = np.asarray(snapshots, dtype=np.float64)
    reconstructed = truth @ basis @ basis.T
    return _relative_frobenius(reconstructed, truth)


def paired_median(errors: Sequence[float], baseline: Sequence[float]) -> float:
    """同一重复上 err(估计器) - err(ols) 的中位数。"""
    gap = np.asarray(errors, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    return float(np.median(gap))


def bootstrap_interval(values: Sequence[float], gen: np.random.Generator, n_boot: int = 2000) -> tuple[float, float]:
    """按重复重抽样的 2.5% 与 97.5% 分位。"""
    data = np.asarray(values, dtype=np.float64)
    draws = gen.integers(0, data.shape[0], size=(int(n_boot), data.shape[0]))
    means = np.median(data[draws], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _rank(O_ref: np.ndarray) -> int:
    return int(np.asarray(O_ref).shape[1])


def _relative_frobenius(estimate: np.ndarray, reference: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference))
    if denom == 0.0:
        return float("nan")
    return float(np.linalg.norm(estimate - reference) / denom)
