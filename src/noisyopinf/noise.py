"""AR(1) 测量噪声。每条轨迹单独调用一次。"""

from __future__ import annotations

import numpy as np


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
