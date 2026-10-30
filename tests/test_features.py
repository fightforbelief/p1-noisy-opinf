"""05 第 1 节：特征。"""

from __future__ import annotations

import numpy as np

from noisyopinf.features import diag_quad_rows, exponents, features, quad_index
from noisyopinf.seeds import rng


def test_quad_index_order():
    assert quad_index(3) == [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


def test_feature_values():
    state = rng(11, 2).normal(size=(1, 4))
    encoded = features(state)[0]
    pairs = quad_index(4)
    assert np.allclose(encoded[:4], state[0])
    manual = [state[0, i] * state[0, j] for i, j in pairs]
    assert np.allclose(encoded[4:-1], manual)
    assert encoded[-1] == 1.0


def test_exponents_match_features():
    state = rng(11, 3).normal(size=(1, 4))
    encoded = features(state)[0]
    table = exponents(4)
    for row, powers in enumerate(table):
        value = np.prod(state[0] ** powers)
        assert abs(encoded[row] - value) < 1e-12


def test_diag_rows():
    state = rng(11, 4).normal(size=(2, 3))
    encoded = features(state)
    rows = diag_quad_rows(3)
    assert np.allclose(encoded[:, rows], state**2)
