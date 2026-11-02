"""rollout 的步长、视界和发散。"""

from __future__ import annotations

import numpy as np

from noisyopinf.features import features
from noisyopinf.rollout import rollout, threshold_from_train
from noisyopinf.seeds import rng


def _linear(scale: float, rank: int = 2) -> np.ndarray:
    quad = rank * (rank + 1) // 2
    operator = np.zeros((rank + quad + 1, rank))
    operator[:rank] = scale * np.eye(rank)
    return operator


def test_one_step_matches_features():
    operator = rng(13, 1).normal(size=(6, 2))
    state = np.array([0.3, -0.4])
    original = state.copy()
    saved = operator.copy()
    predicted, diverged = rollout(operator, state, 1, 1.0e6)
    expected = features(state[None, :])[0] @ saved
    np.testing.assert_allclose(predicted[1], expected)
    assert predicted.shape == (2, 2)
    assert diverged is False
    np.testing.assert_array_equal(state, original)
    np.testing.assert_array_equal(operator, saved)


def test_multistep_linear_and_horizon():
    operator = _linear(0.5)
    state = np.array([1.0, 2.0])
    predicted, diverged = rollout(operator, state, 4, 1.0e6)
    assert predicted.shape == (5, 2)
    assert diverged is False
    for step in range(5):
        np.testing.assert_allclose(predicted[step], state * 0.5**step)


def test_threshold_keeps_blowup_then_nan():
    operator = _linear(10.0)
    predicted, diverged = rollout(operator, np.array([1.0, 1.0]), 3, 50.0)
    assert diverged is True
    np.testing.assert_allclose(predicted[1], [10.0, 10.0])
    np.testing.assert_allclose(predicted[2], [100.0, 100.0])
    assert np.isnan(predicted[3]).all()


def test_nonfinite_fills_from_the_bad_state():
    operator = _linear(1.0)
    operator[0, 0] = np.nan
    predicted, diverged = rollout(operator, np.array([1.0, 1.0]), 2, 1.0e6)
    assert diverged is True
    np.testing.assert_allclose(predicted[0], [1.0, 1.0])
    assert np.isnan(predicted[1:]).all()


def test_infinite_initial_state_stops_immediately():
    predicted, diverged = rollout(_linear(1.0), np.array([np.inf, 0.0]), 2, 1.0e6)
    assert diverged is True
    assert np.isposinf(predicted[0, 0])
    assert np.isnan(predicted[1:]).all()


def test_threshold_from_train_uses_specified_factor():
    train = [np.array([[0.0, 2.0], [1.0, -4.0]])]
    assert threshold_from_train(train) == 4.0e3
