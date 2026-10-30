"""05 第 6 节：系统和 ETDRK4。"""

from __future__ import annotations

import numpy as np

from noisyopinf.seeds import rng
from noisyopinf.systems import burgers, galerkin_burgers, kdv_burgers
from noisyopinf.systems.spectral import solve, wavenumbers


def test_galerkin_orthonormal():
    phi, _ = galerkin_burgers.basis(4, 64)
    gram = phi.T @ phi / phi.shape[0]
    assert np.max(np.abs(gram - np.eye(gram.shape[0]))) < 1e-14


def test_galerkin_energy_conserving():
    phi, phi_x = galerkin_burgers.basis(4, 64)
    q = rng(8, 1).normal(size=phi.shape[1])
    flux = galerkin_burgers.nonlinear(q, phi, phi_x)
    assert abs(float(q @ flux)) < 1e-12 * np.linalg.norm(q) ** 3


def test_galerkin_exact_quadratic():
    _, residual, constant = galerkin_burgers.true_operator()
    assert np.max(np.abs(residual)) < 1e-11
    assert np.linalg.norm(constant) < 1e-12


def test_etdrk4_linear_exact():
    n, mode = 64, 3
    x = np.arange(n, dtype=np.float64) / n
    u0 = np.cos(2.0 * np.pi * mode * x)
    lk = -0.3 * wavenumbers(n) ** 2
    dt, n_sub, n_obs = 1e-3, 10, 20
    final = solve(u0, lk, dt, n_sub, n_obs, nonlinear=False)[-1]
    expected = np.exp(lk[mode] * n_obs * n_sub * dt) * u0
    assert np.max(np.abs(final - expected)) < 1e-10


def _final(u0, lk, dt, t_end):
    steps = int(round(t_end / dt))
    if abs(steps * dt - t_end) > 1e-12:
        raise AssertionError("dt does not land on the final time")
    return solve(u0, lk, dt, 1, steps)[-1]


def _relative(left, right):
    return float(np.linalg.norm(left - right) / np.linalg.norm(right))


def _order_errors(u0, lk, dts, t_end):
    reference = _final(u0, lk, dts[-1] / 32.0, t_end)
    return [_relative(_final(u0, lk, dt, t_end), reference) for dt in dts]


def test_etdrk4_order():
    n = 256
    u0 = burgers.sine_initial(n, 2, 1, 0, 0.5)
    lk = -0.01 * wavenumbers(n) ** 2
    dts = [1.0e-3, 5.0e-4, 2.5e-4]
    errors = _order_errors(u0, lk, dts, 0.2)
    if min(errors) < 1.0e-11:
        dts = [4.0e-3, 2.0e-3, 1.0e-3]
        errors = _order_errors(u0, lk, dts, 0.2)
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(10.0 <= ratio <= 22.0 for ratio in ratios), (dts, errors, ratios)


def test_burgers_mean_conserved():
    traj = burgers.simulate(0, 1)
    means = traj.mean(axis=1)
    assert np.max(np.abs(means - means[0])) < 1e-12


def test_kdv_symbol_matches_formula():
    k = wavenumbers(16)
    symbol = kdv_burgers.linear_symbol(16, 0.005, 1.0e-5)
    expected = -0.005 * k**2 + 1j * 1.0e-5 * k**3
    assert np.allclose(symbol, expected)
