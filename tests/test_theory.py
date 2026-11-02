"""mc_limit_ar1 的公式测试和 05 规定的 Monte Carlo 对照。"""

from __future__ import annotations

import numpy as np

from noisyopinf.estimators import fit
from noisyopinf.features import features, quad_index
from noisyopinf.noise import add_ar1
from noisyopinf.seeds import rng
from noisyopinf.theory import mc_limit_ar1

ORDER_R3 = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


def _reference_limit(trajs: list[np.ndarray], s2: float, phi: float) -> np.ndarray:
    """在原坐标手写 02 第 6 节，不经过被测函数的缩放。"""
    inputs = np.vstack([traj[:-1] for traj in trajs])
    outputs = np.vstack([traj[1:] for traj in trajs])
    rank = inputs.shape[1]
    pairs = [(i, j) for i in range(rank) for j in range(i, rank)]
    width = rank + len(pairs) + 1
    delta = np.zeros((width, rank))
    delta[:rank] = inputs.shape[0] * phi * s2 * np.eye(rank)
    totals = inputs.sum(axis=0)
    for pos, (left, right) in enumerate(pairs):
        row = np.zeros(rank)
        row[right] += totals[left]
        row[left] += totals[right]
        delta[rank + pos] = phi * s2 * row
    encoded = np.hstack(
        [
            inputs,
            np.column_stack([inputs[:, i] * inputs[:, j] for i, j in pairs]),
            np.ones((inputs.shape[0], 1)),
        ]
    )
    return np.linalg.solve(encoded.T @ encoded, encoded.T @ outputs + delta)


def _recovered_delta(trajs: list[np.ndarray], s2: float, phi: float) -> np.ndarray:
    inputs = np.vstack([np.asarray(traj[:-1], dtype=np.float64) for traj in trajs])
    outputs = np.vstack([np.asarray(traj[1:], dtype=np.float64) for traj in trajs])
    encoded = features(inputs)
    operator = mc_limit_ar1(trajs, s2, phi)
    return encoded.T @ encoded @ operator - encoded.T @ outputs


def _input_peak(trajs: list[np.ndarray]) -> float:
    return float(np.max(np.abs(np.vstack([traj[:-1] for traj in trajs]))))


def _data_peak(trajs: list[np.ndarray]) -> float:
    return float(max(np.max(np.abs(traj)) for traj in trajs))


def _replace_terminals(trajs: list[np.ndarray], value: float) -> list[np.ndarray]:
    edited = [np.array(traj, dtype=np.float64, copy=True) for traj in trajs]
    for traj in edited:
        traj[-1] = value
    return edited


def _three_block_delta(n: int, totals: np.ndarray, s2: float, phi: float) -> np.ndarray:
    """手工三块 Δ。totals 是调用者选定的求和，n 是配对数。"""
    rank = int(totals.shape[0])
    pairs = [(i, j) for i in range(rank) for j in range(i, rank)]
    delta = np.zeros((rank + len(pairs) + 1, rank))
    delta[:rank] = n * phi * s2 * np.eye(rank)
    for pos, (left, right) in enumerate(pairs):
        row = np.zeros(rank)
        row[right] += totals[left]
        row[left] += totals[right]
        delta[rank + pos] = phi * s2 * row
    return delta


def _diverse_trajs() -> list[np.ndarray]:
    gen = rng(7, 3)
    first = gen.normal(0.8, 1.0, size=(18, 2))
    second = gen.normal(0.8, 1.0, size=(16, 2))
    peak = float(np.max(np.abs(np.vstack([first[:-1], second[:-1]]))))
    first[-1] = 0.25 * peak
    second[-1] = 0.5 * peak
    return [first, second]


def test_dictionary_order_and_shape() -> None:
    assert quad_index(3) == ORDER_R3
    operator = mc_limit_ar1(_diverse_trajs(), 0.05, 0.4)
    assert operator.shape == (2 + 3 + 1, 2)


def test_blocks_use_pair_inputs_only() -> None:
    trajs = _diverse_trajs()
    phi, s2 = 0.4, 0.05
    delta = _recovered_delta(trajs, s2, phi)
    inputs = np.vstack([traj[:-1] for traj in trajs])
    totals = inputs.sum(axis=0)
    assert delta.shape == (6, 2)
    np.testing.assert_allclose(delta[:2], inputs.shape[0] * phi * s2 * np.eye(2), atol=1e-8)
    np.testing.assert_allclose(delta[2], 2.0 * phi * s2 * totals[0] * np.array([1.0, 0.0]), atol=1e-8)
    np.testing.assert_allclose(
        delta[3], phi * s2 * np.array([totals[1], totals[0]]), atol=1e-8
    )
    np.testing.assert_allclose(delta[4], 2.0 * phi * s2 * totals[1] * np.array([0.0, 1.0]), atol=1e-8)
    np.testing.assert_allclose(delta[5], 0.0, atol=1e-8)
    assert inputs.shape[0] == 18 - 1 + 16 - 1


def test_terminal_snapshots_do_not_affect_delta() -> None:
    gen = rng(7, 3)
    base = [gen.normal(0.8, 1.0, size=(18, 2)), gen.normal(0.8, 1.0, size=(16, 2))]
    peak = _input_peak(base)
    low = _replace_terminals(base, 0.2 * peak)
    high = _replace_terminals(base, 0.7 * peak)
    assert peak > 0.0
    assert _data_peak(low) == peak
    assert _data_peak(high) == peak
    for left, right in zip(low, high):
        np.testing.assert_array_equal(left[:-1], right[:-1])
    phi, s2 = 0.4, 0.05
    delta_low = _recovered_delta(low, s2, phi)
    delta_high = _recovered_delta(high, s2, phi)
    np.testing.assert_allclose(delta_low, delta_high, atol=1e-8)
    inputs = np.vstack([traj[:-1] for traj in low])
    expected = _three_block_delta(inputs.shape[0], inputs.sum(axis=0), s2, phi)
    np.testing.assert_allclose(delta_low, expected, atol=1e-8)
    wrong_low = _three_block_delta(inputs.shape[0], np.vstack(low).sum(axis=0), s2, phi)
    wrong_high = _three_block_delta(inputs.shape[0], np.vstack(high).sum(axis=0), s2, phi)
    assert np.max(np.abs(wrong_low - wrong_high)) > 1e-4
    assert np.max(np.abs(wrong_low - expected)) > 1e-4


def test_quadratic_omission_is_visible_at_nonzero_mean() -> None:
    trajs = _diverse_trajs()
    phi, s2 = 0.4, 0.05
    found = mc_limit_ar1(trajs, s2, phi)
    inputs = np.vstack([traj[:-1] for traj in trajs])
    outputs = np.vstack([traj[1:] for traj in trajs])
    encoded = features(inputs)
    gram = encoded.T @ encoded
    cross = encoded.T @ outputs
    linear_only = np.zeros_like(cross)
    linear_only[:2] = inputs.shape[0] * phi * s2 * np.eye(2)
    omitted = np.linalg.solve(gram, cross + linear_only)
    clean = np.linalg.solve(gram, cross)
    ratio = np.linalg.norm(found - omitted) / np.linalg.norm(found - clean)
    assert ratio > 0.2


def test_phi_zero_removes_delta() -> None:
    trajs = _diverse_trajs()
    found = mc_limit_ar1(trajs, 0.05, 0.0)
    inputs = np.vstack([traj[:-1] for traj in trajs])
    outputs = np.vstack([traj[1:] for traj in trajs])
    encoded = features(inputs)
    clean = np.linalg.solve(encoded.T @ encoded, encoded.T @ outputs)
    np.testing.assert_allclose(found, clean, atol=1e-10)
    np.testing.assert_allclose(_recovered_delta(trajs, 0.05, 0.0), 0.0, atol=1e-8)


def test_kappa_scaling_matches_original_formula() -> None:
    base = _diverse_trajs()
    found = mc_limit_ar1(base, 0.05, 0.4)
    np.testing.assert_allclose(found, _reference_limit(base, 0.05, 0.4), rtol=1e-10, atol=1e-10)
    grown = [50.0 * traj for traj in base]
    scaled = mc_limit_ar1(grown, 0.05 * 50.0**2, 0.4)
    restored = scaled.copy()
    restored[2:5] *= 50.0
    restored[-1] /= 50.0
    np.testing.assert_allclose(restored, found, rtol=1e-8, atol=1e-8)


def test_singular_limit_is_nan() -> None:
    operator = mc_limit_ar1([np.ones((2, 1))], 0.1, 0.3)
    assert operator.shape == (3, 1)
    assert np.isnan(operator).all()


def test_mc_limit_ar1() -> None:
    """05：φ=0.5，n=20000，σ=0.05·RMS，200 次，差小于偏差的 10%。"""
    clean, truth = _quadratic_sample(n_traj=200, steps=100)
    sigma = 0.05 * _rms(clean)
    limit = mc_limit_ar1(clean, sigma**2, 0.5)
    accumulated = np.zeros_like(limit)
    for rep in range(200):
        noisy = [
            add_ar1(traj, sigma, 0.5, rng(1, 4, 8, 0, rep, index))
            for index, traj in enumerate(clean)
        ]
        accumulated += fit(noisy, "mc", sigma2=sigma**2).O
    mean = accumulated / 200.0
    gap = np.linalg.norm(mean - limit)
    bias = np.linalg.norm(limit - truth)
    assert bias > 0.0
    assert gap < 0.1 * bias


def _quadratic_sample(n_traj: int, steps: int) -> tuple[list[np.ndarray], np.ndarray]:
    gen = rng(7, 1)
    rank = 3
    matrix = gen.normal(size=(rank, rank))
    matrix *= 0.9 / np.max(np.abs(np.linalg.eigvals(matrix)))
    coupling = 0.02 * gen.normal(size=(rank, 6))
    shift = 0.05 * gen.normal(size=rank)
    trajs = []
    for index in range(n_traj):
        state = 0.5 * rng(7, 2, index).normal(size=rank)
        rows = [state]
        for _ in range(steps):
            state = _quadratic_step(state, matrix, coupling, shift)
            rows.append(state)
        trajs.append(np.vstack(rows))
    stacked = np.vstack(trajs)
    assert np.isfinite(stacked).all()
    assert np.max(np.abs(stacked)) < 20.0
    truth = np.vstack([matrix.T, coupling.T, shift])
    return trajs, truth


def _quadratic_step(
    state: np.ndarray, matrix: np.ndarray, coupling: np.ndarray, shift: np.ndarray
) -> np.ndarray:
    pairs = quad_index(state.shape[0])
    monomials = np.array([state[i] * state[j] for i, j in pairs])
    return matrix @ state + coupling @ monomials + shift


def _rms(trajs: list[np.ndarray]) -> float:
    stacked = np.vstack(trajs)
    return float(np.sqrt(np.mean(np.sum(stacked * stacked, axis=1) / stacked.shape[1])))


def test_ols_limit_zero_noise():
    from noisyopinf.theory import ols_limit
    from tests.quadratic_system import quadratic_operator, relative_error, rollout

    operator = quadratic_operator((11, 80))
    trajs = rollout(operator, 8, 40, (11, 81))
    assert relative_error(ols_limit(trajs, 0.0), fit(trajs, "ols").O) < 1e-10


def test_linear_attenuation():
    from noisyopinf.theory import linear_attenuation_limit, ols_limit

    matrix = rng(11, 82).normal(size=(3, 3))
    matrix *= 0.8 / np.max(np.abs(np.linalg.eigvals(matrix)))
    trajs = []
    gen = rng(11, 83)
    for _ in range(5):
        state = gen.normal(size=3)
        traj = np.empty((21, 3))
        traj[0] = state
        for time in range(20):
            state = matrix @ state
            traj[time + 1] = state
        trajs.append(traj)
    inputs = np.vstack([traj[:-1] for traj in trajs])
    moment = inputs.T @ inputs / inputs.shape[0]
    variance = 0.04
    solved = ols_limit(trajs, variance, dictionary="linear")
    expected = linear_attenuation_limit(matrix, moment, variance)
    gap = np.linalg.norm(solved.T - expected) / np.linalg.norm(expected)
    assert gap < 1e-10


def test_ols_limit_matches_mc():
    from noisyopinf.noise import add_white
    from noisyopinf.theory import ols_limit
    from tests.quadratic_system import quadratic_operator, rollout

    operator = quadratic_operator((11, 84))
    trajs = rollout(operator, 200, 100, (11, 85))
    sigma = 0.05 * _rms(trajs)
    limit = ols_limit(trajs, sigma**2)
    accumulated = np.zeros_like(limit)
    for rep in range(200):
        gen = rng(11, 86, rep)
        noisy = [add_white(traj, sigma, gen) for traj in trajs]
        accumulated += fit(noisy, "ols").O
    mean = accumulated / 200.0
    bias = np.linalg.norm(limit - operator)
    assert bias > 0.0
    assert np.linalg.norm(mean - limit) < 0.1 * bias


def _clean_plus_delta(trajs: list[np.ndarray], delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inputs = np.vstack([traj[:-1] for traj in trajs])
    outputs = np.vstack([traj[1:] for traj in trajs])
    encoded = features(inputs)
    gram = encoded.T @ encoded
    cross = encoded.T @ outputs
    clean = np.linalg.solve(gram, cross)
    return clean, clean + np.linalg.solve(gram, delta)


def test_one_dimensional_delta_blocks() -> None:
    """1 维玩具：线性块 nφσ²，对角二次块系数为 2，常数行为 0。"""
    trajs = [np.array([[1.0], [2.0], [4.0], [7.0]])]
    variance, lag = 0.2, 0.5
    delta = np.zeros((3, 1))
    delta[0, 0] = 3 * lag * variance
    delta[1, 0] = 2 * lag * variance * 7.0
    clean, shifted = _clean_plus_delta(trajs, delta)
    np.testing.assert_allclose(mc_limit_ar1(trajs, variance, 0.0), clean, atol=1e-12)
    np.testing.assert_allclose(mc_limit_ar1(trajs, variance, lag), shifted, atol=1e-10)
    assert delta[2, 0] == 0.0


def test_quadratic_cross_term_uses_both_sums() -> None:
    """2 维玩具：S=(7, 7)，交叉项 φσ²(δ_jb S_i + δ_ib S_j)。"""
    states = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0], [1.0, 2.0], [2.0, 1.0], [5.0, 5.0]]
    )
    trajs = [states[:5], states[4:]]
    variance, lag = 0.5, 0.4
    factor = lag * variance
    delta = np.zeros((6, 2))
    delta[0, 0] = 7 * factor
    delta[1, 1] = 7 * factor
    delta[2, 0] = 2 * factor * 7.0
    delta[3] = factor * np.array([7.0, 7.0])
    delta[4, 1] = 2 * factor * 7.0
    clean, shifted = _clean_plus_delta(trajs, delta)
    np.testing.assert_allclose(mc_limit_ar1(trajs, variance, 0.0), clean, atol=1e-12)
    np.testing.assert_allclose(mc_limit_ar1(trajs, variance, lag), shifted, atol=1e-10)
    np.testing.assert_allclose(delta[5], 0.0)
