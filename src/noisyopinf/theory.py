"""AR(1) 噪声下 MC 的有限样本期望之比。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from noisyopinf.features import features, quad_index


def mc_limit_ar1(
    clean_trajs: Sequence[np.ndarray], s2: float, phi: float
) -> np.ndarray:
    """返回 O_lim = Gbar_0^{-1}(C_0+Delta)，形状 (p, r)。

    这是 E[Ghat]^{-1} E[Chat]，不是 E[Ghat^{-1} Chat]。
    """
    trajs, rank = _validated_trajs(clean_trajs)
    variance, lag = _check_noise(s2, phi)
    kappa = _peak_abs(trajs)
    scaled = [traj / kappa for traj in trajs]
    solved = _scaled_limit(scaled, variance / kappa**2, lag)
    if not np.isfinite(solved).all():
        return solved
    return _to_original(solved, rank, kappa)


def _validated_trajs(clean_trajs: Sequence[np.ndarray]) -> tuple[list[np.ndarray], int]:
    if len(clean_trajs) == 0:
        raise ValueError("clean_trajs is empty")
    checked: list[np.ndarray] = []
    rank: int | None = None
    for traj in clean_trajs:
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


def _check_noise(s2: float, phi: float) -> tuple[float, float]:
    variance = float(s2)
    lag = float(phi)
    if not np.isfinite(variance) or not np.isfinite(lag):
        raise ValueError("s2 and phi must be finite")
    if variance < 0.0:
        raise ValueError("s2 must be non-negative")
    if abs(lag) > 1.0:
        raise ValueError("phi must lie in [-1, 1]")
    return variance, lag


def _peak_abs(trajs: Sequence[np.ndarray]) -> float:
    peak = max(float(np.max(np.abs(traj))) for traj in trajs)
    if peak == 0.0 or not np.isfinite(peak):
        return 1.0
    return peak


def _pair_states(trajs: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    inputs = [traj[:-1] for traj in trajs]
    outputs = [traj[1:] for traj in trajs]
    return np.vstack(inputs), np.vstack(outputs)


def _delta(inputs: np.ndarray, variance: float, lag: float) -> np.ndarray:
    rank = inputs.shape[1]
    pairs = quad_index(rank)
    width = rank + len(pairs) + 1
    delta = np.zeros((width, rank), dtype=np.float64)
    factor = lag * variance
    delta[:rank] = inputs.shape[0] * factor * np.eye(rank)
    totals = inputs.sum(axis=0)
    for pos, (left, right) in enumerate(pairs):
        row = np.zeros(rank, dtype=np.float64)
        row[right] += totals[left]
        row[left] += totals[right]
        delta[rank + pos] = factor * row
    return delta


def _scaled_limit(trajs: Sequence[np.ndarray], variance: float, lag: float) -> np.ndarray:
    inputs, outputs = _pair_states(trajs)
    encoded = features(inputs)
    gram = encoded.T @ encoded
    cross = encoded.T @ outputs
    try:
        return np.linalg.solve(gram, cross + _delta(inputs, variance, lag))
    except np.linalg.LinAlgError:
        return np.full((encoded.shape[1], inputs.shape[1]), np.nan)


def _to_original(operator: np.ndarray, rank: int, kappa: float) -> np.ndarray:
    restored = np.array(operator, dtype=np.float64, copy=True)
    quad_count = rank * (rank + 1) // 2
    restored[rank:rank + quad_count] /= kappa
    restored[-1] *= kappa
    return restored
