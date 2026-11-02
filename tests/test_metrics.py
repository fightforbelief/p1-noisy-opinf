"""手算得到的算子误差、视界和汇总。"""

from __future__ import annotations

import numpy as np

from noisyopinf.metrics import (
    bias_rel,
    block_bias_sd,
    ep_violation,
    err_A_rel,
    err_H_rel,
    err_O_rel,
    err_c_abs,
    err_c_scaled,
    horizon_medians,
    iqr,
    proj_floor,
    rollout_aggregate,
    rollout_err_h,
    sd_rel,
    specrad_trio,
    state_rms,
)
from noisyopinf.seeds import rng


def _linear(scale: float) -> np.ndarray:
    operator = np.zeros((6, 2))
    operator[:2] = scale * np.eye(2)
    return operator


def test_identical_operator_has_zero_error():
    operator = _linear(1.0)
    assert err_O_rel(operator, operator) == 0.0
    assert err_A_rel(operator, operator) == 0.0
    assert err_c_abs(operator, operator) == 0.0


def test_hand_operator_error_and_blocks():
    reference = _linear(1.0)
    estimate = reference.copy()
    estimate[0, 0] += 1.0
    assert err_O_rel(estimate, reference) == 1.0 / np.sqrt(2.0)
    assert err_A_rel(estimate, reference) == 1.0 / np.sqrt(2.0)
    hat, ref, gap = specrad_trio(estimate, reference)
    assert hat == 2.0 and ref == 1.0 and gap == 1.0
    reference[2, 0] = 2.0
    reference[-1] = np.array([3.0, 4.0])
    estimate = reference.copy()
    estimate[2, 0] = 4.0
    estimate[-1, 1] = 6.0
    assert err_H_rel(estimate, reference) == 1.0
    assert err_c_abs(estimate, reference) == 2.0
    assert err_c_scaled(estimate, reference, state_rms(np.ones((4, 2)))) == 2.0


def test_bias_and_sd_match_two_point_ensemble():
    reference = _linear(1.0)
    bumped = reference.copy()
    bumped[0, 0] += 2.0
    estimates = [reference, bumped]
    assert bias_rel(estimates, reference) == 1.0 / np.sqrt(2.0)
    assert sd_rel(estimates, reference) == np.sqrt(2.0) / np.sqrt(2.0)
    blocks = block_bias_sd(estimates, reference)
    assert blocks["bias_A_rel"] == 1.0 / np.sqrt(2.0)
    assert blocks["sd_A_rel"] == 1.0
    assert np.isnan(blocks["bias_H_rel"])
    assert np.isnan(blocks["sd_H_rel"])
    assert np.isnan(sd_rel([reference], reference))


def test_horizon_median_and_failure_values():
    truth = np.ones((11, 1))
    predicted = np.full((11, 1), 2.0)
    assert rollout_err_h(predicted, truth, 10) == 1.0
    longer = np.vstack([predicted, [[9.0]]])
    assert rollout_err_h(longer, np.vstack([truth, [[1.0]]]), 10) == 1.0
    broken = predicted.copy()
    broken[5, 0] = np.nan
    assert rollout_err_h(broken, truth, 10) == np.inf
    summary = rollout_aggregate([1.0, 3.0, 9.0], [False, False, True])
    assert summary["rollout_med"] == 3.0
    assert summary["rollout_mean_finite"] == 2.0
    assert summary["diverged_frac"] == 1.0 / 3.0
    named = horizon_medians({10: [1.0, 3.0], 50: [4.0, np.inf]})
    assert named["rollout_med_h10"] == 2.0
    assert named["rollout_med_h50"] == np.inf


def test_short_horizon_is_rejected_and_h50_uses_51_states():
    import pytest
    prefix = np.ones((52, 1))
    predicted = prefix.copy()
    predicted[-1, 0] = 0.0
    assert rollout_err_h(predicted, prefix, 50) == 0.0
    changed = prefix.copy()
    changed[50, 0] = 3.0
    assert rollout_err_h(changed, prefix, 50) == 2.0 / np.sqrt(51.0)
    with pytest.raises(ValueError, match="50"):
        rollout_err_h(np.ones((50, 1)), np.ones((50, 1)), 50)
    with pytest.raises(ValueError, match="10"):
        rollout_err_h(np.ones((10, 1)), np.ones((10, 1)), 10)


def test_full_order_window_uses_basis():
    basis = np.array([[1.0], [0.0]])
    predicted = np.ones((3, 1))
    truth = np.tile(np.array([[1.0, 0.0]]), (3, 1))
    assert rollout_err_h(predicted, truth, 2, basis) == 0.0


def test_projection_floor_energy_and_iqr():
    basis = np.array([[1.0], [0.0]])
    snapshots = np.array([[3.0, 4.0]])
    assert proj_floor(basis, snapshots) == 4.0 / 5.0
    operator = _linear(1.0)
    assert ep_violation(operator, rng(13, 2).normal(size=(5, 2))) == 0.0
    assert iqr([1.0, 2.0, 3.0, 4.0]) == 1.5
