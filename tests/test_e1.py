"""E1 的网格、种子、汇总和失败保留。"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from noisyopinf.e1 import (
    add_training_noise,
    delta_method_se,
    expand_cases,
    fit_replicate,
    gap_terms,
    limit_bundle,
    load_config,
    monte_carlo_moments,
    nested_prefix,
    planned_runs,
    refuse_existing,
    rho_index,
    run_record,
    summarize_case,
)
from noisyopinf.io_utils import RUN_COLUMNS
from noisyopinf.seeds import rng


def test_e1_grid_has_2400_ols_runs():
    config = load_config()
    cases = expand_cases(config)
    assert len(cases) == 12
    assert planned_runs(config) == 2400
    assert [case["estimator"] for case in cases] == ["ols"] * 12
    assert [case["rho_index"] for case in cases if case["n_train"] == 10] == [2, 3, 4, 5]


def test_training_prefixes_are_nested():
    trajectories = [np.full((3, 2), index, dtype=np.float64) for index in range(6)]
    short = nested_prefix(trajectories, 2)
    middle = nested_prefix(trajectories, 5)
    assert all(item is origin for item, origin in zip(short, trajectories[:2], strict=True))
    assert all(item is origin for item, origin in zip(middle[:2], short, strict=True))


def test_noise_seed_is_reproducible_per_replicate():
    clean = [rng(11, 210).normal(size=(6, 2)) for _ in range(3)]
    first = add_training_noise(clean, 0.02, rho_index(0.05), 4)
    second = add_training_noise(clean, 0.02, rho_index(0.05), 4)
    other = add_training_noise(clean, 0.02, rho_index(0.05), 5)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_allclose(left, right)
    assert any(not np.allclose(left, right) for left, right in zip(first, other, strict=True))


def test_pairs_stay_inside_each_trajectory():
    clean = [rng(11, 211, index).normal(size=(8, 2)) for index in range(3)]
    result = fit_replicate(clean, 0.01, rho_index(0.02), 0)
    assert result["finite"]
    assert result["n_pairs"] == 3 * 7


def test_failed_replicate_stays_in_the_mean():
    stack = np.ones((5, 6, 2), dtype=np.float64)
    stack[3] = np.nan
    mean, error, failed = monte_carlo_moments(stack)
    assert failed == [3]
    assert np.isnan(mean).all() and np.isnan(error).all()


def test_zero_denominator_stays_nan():
    block = np.zeros((2, 2))
    terms = gap_terms(block, block, block)
    assert terms["gap_numer"] == 0.0
    assert math.isnan(terms["theory_gap_rel"])
    assert math.isnan(terms["theory_gap_abs"])


def test_bias_norm_se_is_not_the_frobenius_se():
    mean = np.array([[2.0, 0.0]])
    truth = np.array([[0.0, 0.0]])
    error = np.array([[0.3, 4.0]])
    assert delta_method_se(mean, truth, error) == pytest.approx(0.3)
    assert float(np.linalg.norm(error)) > 4.0


def test_summary_matches_an_independent_norm():
    generator = np.random.default_rng(0)
    truth = generator.normal(size=(6, 2))
    limit = truth + 0.2
    stack = np.stack([limit + 0.01 * generator.normal(size=limit.shape) for _ in range(25)])
    case = {"rho": 0.05, "rho_index": 3, "n_train": 4, "sigma2": 1.0e-4}
    row = summarize_case(case, stack, limit, truth, {"gram_cond": 3.0, "matches_ols_limit_rel": 0.0})
    mean = np.mean(stack, axis=0)
    manual = float(np.linalg.norm(mean - limit) / np.linalg.norm(limit - truth))
    assert row["status"] == "ok"
    assert row["O_theory_gap_rel"] == pytest.approx(manual)
    assert "A_bias_se_delta" in row and "H_bias_se_delta" in row


def test_run_row_uses_only_common_columns():
    truth = np.eye(6, 2)
    result = {
        "replicate": 4, "O": np.full((6, 2), np.nan), "n_pairs": 12, "fit_seconds": 0.2,
        "diagnostics": {"error": "singular design", "gram_cond": float("nan"), "n_clipped": 0},
        "finite": False,
    }
    case = {"rho": 0.2, "rho_index": 5, "n_train": 10, "sigma2": 0.04}
    row = run_record(case, result, truth, 1.0)
    assert set(row) == set(RUN_COLUMNS)
    assert "theory_gap_rel" not in row
    assert row["status"] == "failed" and row["replicate"] == 4
    assert math.isnan(row["err_O_rel"])
    assert row["rollout_med"] is None


def test_limit_diagnostics_match_the_accepted_function():
    clean = [rng(11, 212, index).normal(size=(15, 3)) for index in range(5)]
    official, diag = limit_bundle(clean, 0.02)
    assert np.isfinite(official).all()
    assert diag["matches_ols_limit_rel"] < 1e-8
    assert math.isfinite(diag["gram_cond"])


def test_existing_output_is_not_replaced(tmp_path):
    target = tmp_path / "e1"
    target.mkdir()
    with pytest.raises(FileExistsError):
        refuse_existing(target)


def test_summary_csv_quotes_the_seed_pattern(tmp_path):
    module = _script()
    row = {
        "exp": "e1", "system": "s1", "rho": 0.05, "n_train": 10, "status": "ok",
        "seed_pattern": "1,4,1,rho_index,replicate,traj",
    }
    row["O_theory_gap_rel"] = 0.125
    module._write_summary(tmp_path / "summary.csv", [row])
    loaded = module._read_summary(tmp_path / "summary.csv")
    assert loaded[0]["seed_pattern"] == row["seed_pattern"]
    assert loaded[0]["O_theory_gap_rel"] == pytest.approx(0.125)
    assert loaded[0]["status"] == "ok"


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "e1_ols_bias.py"
    spec = importlib.util.spec_from_file_location("e1_ols_bias_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
