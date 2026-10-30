"""05 第 3 节：估计器。"""

from __future__ import annotations

import numpy as np

from noisyopinf.estimators import fit
from noisyopinf.features import diag_quad_rows, features
from noisyopinf.noise import add_white
from noisyopinf.seeds import rng
from tests.quadratic_system import quadratic_operator, relative_error, rollout

OPERATOR = quadratic_operator()
SHORT = rollout(OPERATOR, 20, 100, (11, 1))


def _noisy(trajs, sigma, key):
    gen = rng(*key)
    return [add_white(traj, sigma, gen) for traj in trajs]


def _rms(trajs) -> float:
    stacked = np.vstack(trajs)
    return float(np.sqrt(np.mean(stacked**2)))


def test_short_system_stays_bounded():
    assert max(np.max(np.abs(traj)) for traj in SHORT) < 20.0


def test_ols_exact_recovery():
    estimate = fit(SHORT, "ols").O
    assert relative_error(estimate, OPERATOR) < 1e-10


def test_all_methods_noise_free():
    for method in ("mc", "mc_psd", "iv", "iv2"):
        estimate = fit(SHORT, method, sigma2=0.0).O
        assert relative_error(estimate, OPERATOR) < 1e-8


def test_scaling_invariance():
    plain = fit(SHORT, "ols").O
    enlarged = [traj * 1000.0 for traj in SHORT]
    scaled = fit(enlarged, "ols").O
    expected = plain.copy()
    rank = plain.shape[1]
    quad = rank * (rank + 1) // 2
    expected[rank:rank + quad] /= 1000.0
    expected[-1] *= 1000.0
    assert relative_error(scaled, expected) < 1e-8


def _median_error(trajs, method, sigma, reps, tag):
    errors = []
    for rep in range(reps):
        noisy = _noisy(trajs, sigma, (11, tag, rep))
        estimate = fit(noisy, method, sigma2=sigma**2).O
        errors.append(relative_error(estimate, OPERATOR))
    return float(np.median(errors))


def test_mc_consistency():
    small = rollout(OPERATOR, 20, 100, (11, 2))
    large = rollout(OPERATOR, 500, 100, (11, 3))
    small_error = _median_error(small, "mc", 0.05 * _rms(small), 10, 70)
    large_sigma = 0.05 * _rms(large)
    large_error = _median_error(large, "mc", large_sigma, 10, 71)
    ols_error = _median_error(large, "ols", large_sigma, 10, 72)
    assert large_error < small_error
    assert large_error < 0.5 * ols_error


def test_iv_consistency():
    small = rollout(OPERATOR, 20, 100, (11, 4))
    large = rollout(OPERATOR, 500, 100, (11, 5))
    for method, tag in (("iv", 80), ("iv2", 90)):
        small_error = _median_error(small, method, 0.05 * _rms(small), 10, tag)
        large_error = _median_error(large, method, 0.05 * _rms(large), 10, tag + 1)
        assert large_error < small_error


def _population_iv(trajs, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """干净配对上的总体 IV 矩解。mu 只加在对角二次特征上。"""
    width = features(trajs[0][:1]).shape[1]
    mu = np.zeros(width, dtype=np.float64)
    mu[diag_quad_rows(3)] = sigma**2
    instruments, design, outputs = [], [], []
    for traj in trajs:
        instruments.append(features(traj[:-2]) + mu)
        design.append(features(traj[1:-1]) + mu)
        outputs.append(traj[2:])
    w_bar = np.vstack(instruments)
    z_bar = np.vstack(design)
    moment = w_bar.T @ z_bar
    return np.linalg.solve(moment, w_bar.T @ np.vstack(outputs)), mu


def _assert_population_intercept(trajs, sigma: float) -> None:
    operator, mu = _population_iv(trajs, sigma)
    target = -mu @ OPERATOR
    gap = operator[-1] - OPERATOR[-1]
    assert np.linalg.norm(gap - target) / np.linalg.norm(target) < 0.1
    corrected = np.array(operator, dtype=np.float64, copy=True)
    corrected[-1] += (sigma**2) * operator[diag_quad_rows(3)].sum(axis=0)
    assert np.linalg.norm(corrected[-1] - OPERATOR[-1]) < 0.1 * np.linalg.norm(target)


def _assert_fit_correction(trajs, sigma: float) -> None:
    noisy = _noisy(trajs, sigma, (11, 61))
    for method in ("iv", "iv2"):
        raw = fit(noisy, method, sigma2=sigma**2, correct_intercept=False).O
        corrected = fit(noisy, method, sigma2=sigma**2).O
        expected = (sigma**2) * raw[diag_quad_rows(3)].sum(axis=0)
        assert np.allclose(corrected[-1] - raw[-1], expected, rtol=1e-10, atol=1e-12)
        assert np.allclose(corrected[:-1], raw[:-1], rtol=1e-10, atol=1e-12)


def test_iv_intercept_correction():
    trajs = rollout(OPERATOR, 250, 200, (11, 6))
    sigma = 0.05 * _rms(trajs)
    _assert_population_intercept(trajs, sigma)
    _assert_fit_correction(trajs, sigma)


def test_ridge_lambda_zero():
    ols = fit(SHORT, "ols").O
    ridge = fit(SHORT, "ridge", lam=1e-12).O
    assert relative_error(ridge, ols) < 1e-6


def test_singular_returns_nan():
    singular = [np.zeros((2, 3)), np.zeros((2, 3))]
    result = fit(singular, "ols")
    assert np.isnan(result.O).all()
    assert result.diagnostics["error"]


def test_pairs_do_not_cross_trajectories():
    short_first = rng(11, 20).normal(size=(4, 3))
    short_second = rng(11, 21).normal(size=(5, 3))
    short_first[-1] = 50.0
    short_second[0] = -50.0
    counted = fit([short_first, short_second], "ols")
    assert counted.n_pairs == 3 + 4
    assert counted.n_pairs != short_first.shape[0] + short_second.shape[0] - 1
    first = rng(11, 22).normal(size=(12, 3))
    second = rng(11, 23).normal(size=(12, 3))
    first[-1] = 50.0
    second[0] = -50.0
    result = fit([first, second], "ols")
    assert result.n_pairs == 11 + 11
    kappa = max(np.max(np.abs(first)), np.max(np.abs(second)))
    inputs = np.vstack([(first / kappa)[:-1], (second / kappa)[:-1]])
    outputs = np.vstack([(first / kappa)[1:], (second / kappa)[1:]])
    manual, _, rank, _ = np.linalg.lstsq(features(inputs), outputs, rcond=None)
    assert rank == manual.shape[0]
    quad = 3 * 4 // 2
    manual = manual.copy()
    manual[3:3 + quad] /= kappa
    manual[-1] *= kappa
    assert relative_error(result.O, manual) < 1e-10
    assert first.shape[0] + second.shape[0] - 1 != result.n_pairs
