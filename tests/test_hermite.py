"""05 第 2 节：Hermite 与期望单项式。"""

from __future__ import annotations

from itertools import product

import numpy as np

from noisyopinf.features import exponents, features, hermite_monomial_sums, hermite_table
from noisyopinf.seeds import rng
from noisyopinf.theory import expected_monomial


def _within_five_se(samples: np.ndarray, truth: float) -> bool:
    center = float(samples.mean())
    error = float(samples.std(ddof=1) / np.sqrt(samples.shape[0]))
    gap = abs(center - truth)
    if error == 0.0:
        return gap == 0.0
    return gap < 5.0 * error


def test_hermite_unbiased_1d():
    variance = 0.3
    noise = rng(11, 5).normal(0.0, np.sqrt(variance), size=10**6)
    for state in (-1.3, 0.0, 0.7):
        table = hermite_table((state + noise)[:, None], variance)[:, :, 0]
        for degree, truth in enumerate([1.0, state, state**2, state**3, state**4]):
            assert _within_five_se(table[degree], truth)


def test_hermite_unbiased_multivariate():
    variance = 0.3
    clean = rng(11, 6).normal(size=3)
    noise = rng(11, 7).normal(0.0, np.sqrt(variance), size=(10**6, 3))
    noisy = clean + noise
    table = hermite_table(noisy, variance)
    for alpha in product(range(5), repeat=3):
        if sum(alpha) > 4:
            continue
        sample = np.ones(noisy.shape[0])
        for coord, degree in enumerate(alpha):
            sample *= table[degree, :, coord]
        assert _within_five_se(sample, float(np.prod(clean**np.array(alpha))))


def test_corrected_gram_unbiased():
    clean = rng(11, 8).normal(size=(50, 3))
    variance = 0.05
    truth = features(clean).T @ features(clean)
    table = exponents(3)
    alphas = (table[:, None, :] + table[None, :, :]).reshape(-1, 3)
    accumulated = np.zeros_like(truth)
    for rep in range(2000):
        noisy = clean + rng(11, 9, rep).normal(0.0, np.sqrt(variance), size=clean.shape)
        sums = hermite_monomial_sums(noisy, variance, alphas)
        accumulated += sums.reshape(truth.shape)
    average = accumulated / 2000.0
    gap = np.linalg.norm(average - truth) / np.linalg.norm(truth)
    assert gap < 0.01


def test_corrected_cross_unbiased():
    clean = rng(11, 10).normal(size=(50, 3))
    future = rng(11, 11).normal(size=(50, 3))
    variance = 0.05
    truth = features(clean).T @ future
    rows = [3 + offset for offset, (i, j) in enumerate([(a, b) for a in range(3) for b in range(a, 3)]) if i == j]
    accumulated = np.zeros_like(truth)
    for rep in range(2000):
        gen = rng(11, 12, rep)
        noisy = clean + gen.normal(0.0, np.sqrt(variance), size=clean.shape)
        noisy_future = future + gen.normal(0.0, np.sqrt(variance), size=future.shape)
        encoded = features(noisy)
        encoded[:, rows] -= variance
        accumulated += encoded.T @ noisy_future
    average = accumulated / 2000.0
    gap = np.linalg.norm(average - truth) / np.linalg.norm(truth)
    assert gap < 0.01


def test_expected_monomial():
    clean = rng(11, 13).normal(size=3)
    variance = 0.3
    noise = rng(11, 14).normal(0.0, np.sqrt(variance), size=(10**6, 3))
    for alpha in product(range(5), repeat=3):
        if sum(alpha) > 4:
            continue
        powers = np.array(alpha)
        sample = np.prod((clean + noise) ** powers, axis=1)
        assert _within_five_se(sample, expected_monomial(clean, powers, variance))
