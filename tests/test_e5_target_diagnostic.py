"""共同窗口、差分恒等式，以及满秩良态下的求解等价。"""

from __future__ import annotations

import numpy as np

from noisyopinf.e5 import spectrum_record, window_counts
from noisyopinf.e5_target_diagnostic import (
    PRED_REL_OBVIOUS,
    PRED_REL_TINY,
    REL_FLOOR,
    SOLVER_REL_FLAG,
    TRUNCATION_REL,
    diagnose_rank,
    equivalence_reliable,
    identity_residual,
    iv2_lstsq_operator,
    iv2_qr_operator,
    iv_qr_operator,
    iv_solve_operator,
    ols_operator,
    per_trajectory_rows,
    stack_rows,
    svd_truncated_solve,
    three_gaps,
    window_index_spec,
)
from noisyopinf.estimators import _one_instrument, fit
from noisyopinf.features import features
from noisyopinf.seeds import rng
from tests.quadratic_system import quadratic_operator, rollout

OPERATOR = quadratic_operator()


def test_thresholds_stay_at_the_prerun_values():
    assert TRUNCATION_REL == 1.0e-8
    assert SOLVER_REL_FLAG == 1.0e-8
    assert PRED_REL_TINY == 1.0e-8
    assert PRED_REL_OBVIOUS == 1.0e-4
    assert REL_FLOOR == 1.0e-12


def test_formal_windows_match_estimator_slices():
    traj = rng(90, 7).normal(size=(11, 3))
    design, instruments, outputs = per_trajectory_rows(traj, "iv")
    expected = _one_instrument(traj, "iv")
    assert np.allclose(design, expected[0])
    assert np.allclose(instruments, expected[1])
    assert np.allclose(outputs, expected[2])
    design2, instruments2, outputs2 = per_trajectory_rows(traj, "iv2")
    expected2 = _one_instrument(traj, "iv2")
    assert np.allclose(design2, expected2[0])
    assert np.allclose(instruments2, expected2[1])
    assert np.allclose(outputs2, expected2[2])
    spec = window_index_spec(11)
    counts = window_counts(11)
    assert spec["ols_full"]["input"] == counts["ols"]["input"]
    assert spec["iv"]["input"] == counts["iv"]["input"]
    assert spec["iv"]["instrument"] == counts["iv"]["instrument"]
    assert spec["iv2"]["lag2"] == counts["iv2"]["lag2"]
    assert spec["iv_unified"]["input"] == (2, 9)
    assert spec["iv_unified"]["n_rows"] == spec["iv"]["n_rows"] - 1


def test_unified_iv_keeps_constant_and_does_not_cross():
    first = rng(90, 5).normal(size=(9, 2))
    second = rng(90, 6).normal(size=(9, 2))
    second[0] = 50.0
    design, instruments, outputs = stack_rows([first, second], "iv_unified")
    assert np.allclose(outputs, np.vstack([first[3:], second[3:]]))
    assert design.shape[0] == 6 * 2
    assert np.allclose(instruments[:, -1], 1.0)
    assert not np.any(np.all(np.isclose(outputs, second[0]), axis=1))
    _design, lagged, _outputs = per_trajectory_rows(first, "iv2")
    assert lagged.shape[1] == 2 * (instruments.shape[1] - 1) + 1
    assert lagged.shape[1] != instruments.shape[1]


def test_three_matrix_gaps_add_without_using_norms():
    gen = rng(90, 4)
    method = gen.normal(size=(10, 3))
    window = gen.normal(size=(10, 3))
    full = gen.normal(size=(10, 3))
    gaps = three_gaps(method, window, full)
    residual = identity_residual(gaps)
    assert float(np.max(np.abs(residual))) < 1.0e-12


def test_well_conditioned_iv_solve_matches_qr():
    gen = rng(90, 1)
    n_rows, width, outputs_width = 60, 6, 2
    orthogonal, _ = np.linalg.qr(gen.normal(size=(n_rows, width)))
    instruments = orthogonal[:, :width] * np.linspace(1.0, 2.0, width)
    design = orthogonal[:, :width] * np.linspace(1.5, 2.5, width)
    truth = gen.normal(size=(width, outputs_width))
    outputs = design @ truth
    solved, _info = iv_solve_operator(design, instruments, outputs)
    qr_operator, qr_info = iv_qr_operator(design, instruments, outputs)
    assert np.linalg.norm(solved - truth) / np.linalg.norm(truth) < 1.0e-10
    assert np.linalg.norm(qr_operator - solved) / np.linalg.norm(solved) < 1.0e-10
    rank_w = spectrum_record(instruments)["numerical_rank"]
    rank_solve = spectrum_record(qr_info["lhs"])["numerical_rank"]
    assert equivalence_reliable(rank_w, width, rank_solve, width) == 1


def test_well_conditioned_iv2_lstsq_matches_qr_projection():
    gen = rng(90, 2)
    n_rows, width, instrument_width, outputs_width = 80, 5, 9, 2
    orthogonal, _ = np.linalg.qr(gen.normal(size=(n_rows, instrument_width)))
    instruments = orthogonal[:, :instrument_width] * np.linspace(1.0, 1.8, instrument_width)
    design = orthogonal[:, :width] * np.linspace(1.2, 2.0, width)
    truth = gen.normal(size=(width, outputs_width))
    outputs = design @ truth
    lstsq_operator, projected, _info = iv2_lstsq_operator(design, instruments, outputs)
    qr_operator, qr_projected, _qr_info = iv2_qr_operator(design, instruments, outputs)
    assert np.linalg.norm(lstsq_operator - truth) / np.linalg.norm(truth) < 1.0e-10
    assert np.linalg.norm(qr_operator - lstsq_operator) / np.linalg.norm(lstsq_operator) < 1.0e-10
    assert np.linalg.norm(qr_projected - projected) / np.linalg.norm(projected) < 1.0e-10
    rank_w = spectrum_record(instruments)["numerical_rank"]
    rank_zhat = spectrum_record(qr_projected)["numerical_rank"]
    assert equivalence_reliable(rank_w, instrument_width, rank_zhat, width) == 1


def test_rank_deficient_instruments_are_not_called_equivalent():
    gen = rng(90, 3)
    instruments = gen.normal(size=(40, 5))
    instruments[:, 4] = instruments[:, 1]
    design = gen.normal(size=(40, 5))
    outputs = gen.normal(size=(40, 2))
    _operator, info = iv_qr_operator(design, instruments, outputs)
    rank_w = spectrum_record(instruments)["numerical_rank"]
    rank_solve = spectrum_record(info["lhs"])["numerical_rank"]
    assert rank_w < 5
    assert equivalence_reliable(rank_w, 5, rank_solve, 5) == 0


def test_ols_normal_residual_is_small_when_the_design_is_benign():
    gen = rng(90, 8)
    design = gen.normal(size=(40, 5))
    truth = gen.normal(size=(5, 2))
    outputs = design @ truth
    operator, info = ols_operator(design, outputs)
    assert info["backward_rel"] < 1.0e-10
    assert np.linalg.norm(operator - truth) / np.linalg.norm(truth) < 1.0e-10


def test_truncation_keeps_the_prespecified_cutoff():
    matrix = np.diag([1.0, 1.0e-4, 1.0e-10])
    rhs = np.ones((3, 1))
    _operator, info = svd_truncated_solve(matrix, rhs, TRUNCATION_REL)
    assert info["cutoff"] == 1.0e-8
    assert info["kept"] == 2


def test_short_exact_system_keeps_the_gap_identity():
    clean = rollout(OPERATOR, 4, 12, (90, 9))
    saved = {
        "ols": fit(clean, "ols").O,
        "iv": fit(clean, "iv", sigma2=0.0, correct_intercept=False).O,
        "iv2": fit(clean, "iv2", sigma2=0.0, correct_intercept=False).O,
    }
    payload = diagnose_rank(clean[0].shape[1], clean, saved, None)
    assert payload["problems"] == []
    assert payload["summary"]["identity_max_abs"] < 1.0e-8
    assert np.isfinite(payload["summary"]["ols_full_normal_rel"])
    assert payload["summary"]["ols_full_normal_rel"] < 1.0e-8
