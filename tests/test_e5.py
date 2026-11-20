"""E5 第 1 部分的样本对齐、零噪声目标和指标复算。"""

from __future__ import annotations

import numpy as np

from noisyopinf.estimators import _instrument_rows, _one_instrument, fit
from noisyopinf.e5 import closure_rel, deviation_record, load_config, residual_cross, scaled_matrices, window_counts
from noisyopinf.features import features
from noisyopinf.pod import pod_basis
from noisyopinf.seeds import rng
from tests.quadratic_system import quadratic_operator, relative_error, rollout

OPERATOR = quadratic_operator()
CONFIG = load_config("configs/e5_part1.yaml")


def test_config_keeps_part1_grid():
    assert CONFIG["r"] == (4, 6, 8, 10, 12, 16, 20)
    assert CONFIG["n_train"] == 100 and CONFIG["n_val"] == 0 and CONFIG["n_test"] == 50
    assert CONFIG["correct_intercept"] is False and CONFIG["sigma2"] == 0


def test_windows_match_estimator_rows():
    traj = rng(51, 1).normal(size=(11, 3))
    spec = window_counts(traj.shape[0])
    design, instruments, outputs = _one_instrument(traj, "iv")
    assert design.shape[0] == spec["iv"]["n_rows"] == 9
    assert np.allclose(design, features(traj[1:-1]))
    assert np.allclose(instruments, features(traj[:-2]))
    assert np.allclose(outputs, traj[2:])
    current, lagged, later = _one_instrument(traj, "iv2")
    assert current.shape[0] == spec["iv2"]["n_rows"] == 8
    assert np.allclose(later, traj[3:])
    assert np.allclose(current, features(traj[2:-1]))
    assert lagged.shape[1] == 2 * (features(traj[:1]).shape[1] - 1) + 1


def test_instrument_rows_do_not_cross_trajectories():
    first = rng(51, 2).normal(size=(7, 3))
    second = rng(51, 3).normal(size=(8, 3))
    second[0] = 80.0
    design, _instruments, outputs = _instrument_rows([first, second], "iv")
    assert design.shape[0] == (7 - 2) + (8 - 2)
    assert np.allclose(outputs, np.vstack([first[2:], second[2:]]))
    assert not np.any(np.all(np.isclose(outputs, second[0]), axis=1))
    iv2_design, _lagged, iv2_outputs = _instrument_rows([first, second], "iv2")
    assert iv2_design.shape[0] == (7 - 3) + (8 - 3)
    assert np.allclose(iv2_outputs, np.vstack([first[3:], second[3:]]))


def test_zero_noise_mc_matches_same_sample_ols():
    base = rollout(OPERATOR, 6, 40, (51, 4))
    gen = rng(51, 5)
    perturbed = [traj + 1.0e-3 * gen.normal(size=traj.shape) for traj in base]
    ols = fit(perturbed, "ols").O
    mc = fit(perturbed, "mc", sigma2=0.0).O
    assert relative_error(mc, ols) < 1e-8
    assert relative_error(ols, OPERATOR) > 1e-5


def test_exact_system_iv_targets_match_clean_ols():
    clean = rollout(OPERATOR, 20, 100, (51, 6))
    reference = fit(clean, "ols").O
    for method in ("iv", "iv2"):
        estimate = fit(clean, method, sigma2=0.0, correct_intercept=False).O
        assert relative_error(estimate, OPERATOR) < 1e-8
        assert relative_error(estimate, reference) < 1e-7


def test_closure_and_small_reference_are_recomputable():
    clean = rollout(OPERATOR, 4, 30, (51, 7))
    operator = fit(clean, "ols").O
    inputs = np.vstack([traj[:-1] for traj in clean])
    outputs = np.vstack([traj[1:] for traj in clean])
    manual = float(np.linalg.norm(outputs - features(inputs) @ operator) / np.linalg.norm(outputs))
    assert abs(closure_rel(clean, operator) - manual) < 1e-12
    record = deviation_record(operator, operator, 1.0e-12)
    assert record["dev_O_abs"] == 0.0 and record["rel_O_status"] == "OK"
    zero = np.zeros_like(operator)
    blocked = deviation_record(operator, zero, 1.0e-12)
    assert np.isnan(blocked["dev_O_rel"]) and blocked["rel_O_status"] == "SMALL_REFERENCE"
    assert blocked["dev_O_abs"] > 0.0


def test_ols_residual_is_orthogonal_to_its_features():
    clean = rollout(OPERATOR, 4, 25, (51, 9))
    operator = fit(clean, "ols").O
    cross = residual_cross(clean, operator, "ols")
    assert cross.shape == operator.shape
    assert np.linalg.norm(cross) < 1e-8 * max(1.0, np.linalg.norm(operator))


def test_ols_spectrum_uses_features_not_raw_states():
    clean = rollout(OPERATOR, 2, 12, (51, 10))
    design, instruments, _kappa = scaled_matrices(clean, "ols")
    width = features(clean[0][:1]).shape[1]
    assert design.shape[1] == width
    assert instruments.shape == design.shape
    assert width > clean[0].shape[1]


def test_basis_uses_only_the_supplied_train_list():
    gen = rng(51, 8)
    train = [gen.normal(size=(12, 6)) for _ in range(3)]
    test = [40.0 * gen.normal(size=(12, 6)) for _ in range(2)]
    basis, _values = pod_basis(train, 2)
    mixed, _mixed_values = pod_basis(train + test, 2)
    assert not np.allclose(np.abs(basis), np.abs(mixed))
