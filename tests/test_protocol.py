"""重复噪声的独立性和 ridge λ 的并列规则。"""

from __future__ import annotations

import numpy as np

from noisyopinf.protocol import LAMBDAS, larger_lambda_index, noise_replicate, select_ridge_lambda
from noisyopinf.seeds import rng


def _clean():
    train = [rng(13, 3).normal(size=(6, 2)), rng(13, 4).normal(size=(5, 2))]
    val = [rng(13, 5).normal(size=(6, 2))]
    test = [rng(13, 6).normal(size=(7, 2))]
    return train, val, test


def test_noise_is_independent_and_test_stays_clean():
    train, val, test = _clean()
    saved_train = [item.copy() for item in train]
    saved = [item.copy() for item in test]
    noisy_train, noisy_val, clean_test = noise_replicate(train, val, test, 0.1, 4)
    again_train, _, _ = noise_replicate(train, val, test, 0.1, 4)
    other_train, _, _ = noise_replicate(train, val, test, 0.1, 5)
    np.testing.assert_allclose(noisy_train[0], again_train[0])
    assert not np.allclose(noisy_train[0], other_train[0])
    assert not np.allclose(noisy_train[0], noisy_val[0])
    np.testing.assert_array_equal(clean_test[0], saved[0])
    np.testing.assert_array_equal(train[0], saved_train[0])
    assert len(noisy_train) == len(train)


def test_larger_lambda_wins_ties_and_selection_ignores_test():
    assert LAMBDAS[larger_lambda_index([1.0, 1.0, 2.0])] == LAMBDAS[1]
    assert LAMBDAS[larger_lambda_index([2.0, 1.0, 1.0])] == LAMBDAS[2]
    train, val, test = _clean()
    saved = [item.copy() for item in test]
    chosen = select_ridge_lambda(train, val, 1.0e6)
    assert chosen in LAMBDAS
    np.testing.assert_array_equal(test[0], saved[0])
