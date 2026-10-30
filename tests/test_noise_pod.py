"""05 第 5 节：噪声，以及 σ 的定义。"""

from __future__ import annotations

import numpy as np

from noisyopinf.noise import add_ar1, add_white, sigma_from_rho
from noisyopinf.seeds import rng


def test_white_noise_stats():
    sigma = 0.2
    sample = add_white(np.zeros((20000, 4)), sigma, rng(11, 30))
    assert abs(sample.var() - sigma**2) / sigma**2 < 0.01


def test_ar1_stats():
    sigma, phi = 0.25, 0.6
    sample = add_ar1(np.zeros((80000, 3)), sigma, phi, rng(11, 31))
    assert abs(sample.var() - sigma**2) / sigma**2 < 0.01
    centered = sample - sample.mean(axis=0)
    lag = np.mean(centered[1:] * centered[:-1], axis=0)
    scale = np.mean(centered**2, axis=0)
    assert np.max(np.abs(lag / scale - phi)) < 0.02


def test_projected_noise_isotropic():
    sigma = 0.5
    basis, _ = np.linalg.qr(rng(11, 32).normal(size=(30, 4)))
    noise = add_white(np.zeros((20000, 30)), sigma, rng(11, 33))
    projected = noise @ basis
    covariance = projected.T @ projected / projected.shape[0]
    gap = covariance - sigma**2 * np.eye(4)
    assert np.linalg.norm(gap) / sigma**2 < 0.05


def test_sigma_from_rho_s1_and_s2():
    states = [np.array([[3.0, 0.0], [0.0, 4.0]])]
    assert abs(sigma_from_rho("s1", 0.1, states) - 0.1 * np.sqrt(np.mean([4.5, 8.0]))) < 1e-12
    field = [np.array([[1.0, -1.0], [1.0, -1.0]])]
    assert abs(sigma_from_rho("s2", 0.2, field) - 0.2) < 1e-12
