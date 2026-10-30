"""AR(1) 噪声下 MC 的有限样本期望之比。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from noisyopinf.features import diag_quad_rows, exponents, features, quad_index


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


def expected_monomial(q: np.ndarray, alpha: np.ndarray, s2: float) -> float:
    """一个干净状态上 E[q̃^α]。次数最高到 4。"""
    state = np.asarray(q, dtype=np.float64)
    powers = np.asarray(alpha, dtype=int)
    acc = 1.0
    for coord, degree in enumerate(powers):
        acc *= float(_raw_power(state[coord], int(degree), float(s2)))
    return acc


def expected_monomial_sums(clean_trajs: Sequence[np.ndarray], s2: float, alphas: np.ndarray) -> np.ndarray:
    """配对输入上 E[q̃^α] 的和。不含末端快照，不跨轨迹。"""
    trajs, _ = _validated_trajs(clean_trajs)
    variance = _check_noise(s2, 0.0)[0]
    inputs, _ = _pair_states(trajs)
    return _monomial_sums(inputs, variance, np.asarray(alphas, dtype=int))


def ols_limit(clean_trajs: Sequence[np.ndarray], s2: float, *, dictionary: str = "full") -> np.ndarray:
    """O_lim = Ḡ(σ²)^{-1} C̄(σ²)。线性字典返回 (r, r) 的回归系数。"""
    trajs, rank = _validated_trajs(clean_trajs)
    variance = _check_noise(s2, 0.0)[0]
    kappa = _peak_abs(trajs)
    scaled = [traj / kappa for traj in trajs]
    solved = _ols_limit_scaled(scaled, variance / kappa**2, dictionary)
    if dictionary == "linear" or not np.isfinite(solved).all():
        return solved
    return _to_original(solved, rank, kappa)


def linear_attenuation_limit(A: np.ndarray, M: np.ndarray, s2: float) -> np.ndarray:
    """纯线性回归稀释：A M (M+σ² I)^{-1}。"""
    matrix = np.asarray(M, dtype=np.float64)
    eye = np.eye(matrix.shape[0], dtype=np.float64)
    return np.asarray(A, dtype=np.float64) @ matrix @ np.linalg.inv(matrix + float(s2) * eye)


def _ols_limit_scaled(trajs: Sequence[np.ndarray], variance: float, dictionary: str) -> np.ndarray:
    inputs, outputs = _pair_states(trajs)
    if dictionary == "linear":
        return _linear_limit(inputs, outputs, variance)
    if dictionary != "full":
        raise ValueError("dictionary must be 'full' or 'linear'")
    return _full_limit(inputs, outputs, variance)


def _linear_limit(inputs: np.ndarray, outputs: np.ndarray, variance: float) -> np.ndarray:
    gram = inputs.T @ inputs + inputs.shape[0] * variance * np.eye(inputs.shape[1])
    try:
        return np.linalg.solve(gram, inputs.T @ outputs)
    except np.linalg.LinAlgError:
        return np.full((inputs.shape[1], outputs.shape[1]), np.nan)


def _full_limit(inputs: np.ndarray, outputs: np.ndarray, variance: float) -> np.ndarray:
    table = exponents(inputs.shape[1])
    width = table.shape[0]
    alphas = (table[:, None, :] + table[None, :, :]).reshape(-1, table.shape[1])
    gram = _monomial_sums(inputs, variance, alphas).reshape(width, width)
    shift = np.zeros(width, dtype=np.float64)
    shift[diag_quad_rows(inputs.shape[1])] = variance
    cross = (features(inputs) + shift).T @ outputs
    try:
        return np.linalg.solve(gram, cross)
    except np.linalg.LinAlgError:
        return np.full((width, outputs.shape[1]), np.nan)


def _raw_power(value: float, degree: int, variance: float) -> float:
    if degree == 0:
        return 1.0
    if degree == 1:
        return float(value)
    square = float(value) * float(value)
    if degree == 2:
        return square + variance
    if degree == 3:
        return square * float(value) + 3.0 * variance * float(value)
    if degree == 4:
        return square * square + 6.0 * variance * square + 3.0 * variance * variance
    raise ValueError("moment degree must be at most 4")


def _monomial_sums(states: np.ndarray, variance: float, alphas: np.ndarray) -> np.ndarray:
    acc = np.ones((alphas.shape[0], states.shape[0]), dtype=np.float64)
    for coord in range(states.shape[1]):
        column = np.empty((alphas.shape[0], states.shape[0]), dtype=np.float64)
        for degree in range(5):
            mask = alphas[:, coord] == degree
            if np.any(mask):
                column[mask] = _power_row(states[:, coord], degree, variance)
        acc *= column
    return acc.sum(axis=1)


def _power_row(values: np.ndarray, degree: int, variance: float) -> np.ndarray:
    if degree == 0:
        return np.ones_like(values)
    if degree == 1:
        return values
    square = values * values
    if degree == 2:
        return square + variance
    if degree == 3:
        return values * square + 3.0 * variance * values
    return square * square + 6.0 * variance * square + 3.0 * variance * variance
