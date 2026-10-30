"""白噪声和 AR(1) 测量噪声。每条轨迹单独调用一次。"""

from __future__ import annotations

import numpy as np


def add_white(X: np.ndarray, sigma: float, gen: np.random.Generator) -> np.ndarray:
    """给一条轨迹加独立 N(0, sigma**2) 噪声。X: (T, d)。"""
    states = np.asarray(X, dtype=np.float64)
    scale = float(sigma)
    if scale < 0.0 or not np.isfinite(scale):
        raise ValueError("sigma must be finite and non-negative")
    return states + gen.normal(0.0, scale, size=states.shape)


def add_ar1(
    X: np.ndarray, sigma: float, phi: float, gen: np.random.Generator
) -> np.ndarray:
    """给一条轨迹加边际方差为 sigma**2 的 AR(1) 噪声。X: (T, d)。"""
    states = np.asarray(X, dtype=np.float64)
    scale = float(sigma)
    lag = float(phi)
    if scale < 0.0 or abs(lag) > 1.0:
        raise ValueError("sigma must be non-negative and phi must lie in [-1, 1]")
    shocks = gen.normal(0.0, scale, size=states.shape)
    noise = np.empty_like(shocks)
    noise[0] = shocks[0]
    innov = float(np.sqrt(1.0 - lag * lag))
    for time in range(1, states.shape[0]):
        noise[time] = lag * noise[time - 1] + innov * shocks[time]
    return states + noise


def sigma_from_rho(system: str, rho: float, clean_train) -> float:
    """由相对噪声水平 rho 得到 σ。S1 用坐标 RMS，S2/S3 用全阶 RMS。"""
    level = float(rho)
    if level < 0.0 or not np.isfinite(level):
        raise ValueError("rho must be finite and non-negative")
    stacked = np.vstack([np.asarray(item, dtype=np.float64) for item in clean_train])
    if system == "s1":
        return level * float(np.sqrt(np.mean(np.sum(stacked**2, axis=1) / stacked.shape[1])))
    if system in {"s2", "s3"}:
        return level * float(np.sqrt(np.mean(stacked**2)))
    raise KeyError(system)
