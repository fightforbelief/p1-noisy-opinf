"""E2 的网格、共同阈值、配对和失败保留。"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from noisyopinf.e1 import refuse_existing
from noisyopinf.e2 import (
    ESTIMATORS,
    MEASURES,
    _quartiles,
    add_split_noise,
    csv_row,
    evaluate_arrays,
    expand_cases,
    load_config,
    load_s1_split,
    noise_keys,
    pairs_per_estimator,
    planned_runs,
    shared_threshold,
    summary_rows,
)
from noisyopinf.estimators import fit
from noisyopinf.io_utils import RUN_COLUMNS
from noisyopinf.metrics import err_O_rel, rollout_aggregate
from noisyopinf.protocol import select_ridge_lambda
from noisyopinf.rollout import threshold_from_train
from noisyopinf.seeds import rng


def test_e2_grid_has_12600_rows():
    config = load_config()
    cases = expand_cases(config)
    assert len(cases) == 18
    assert planned_runs(config) == 12600
    assert cases[0]["rho_index"] == 0
    assert [case["rho_index"] for case in cases if case["n_train"] == 10] == [0, 1, 2, 3, 4, 5]


def test_shared_threshold_uses_only_the_clean_prefix():
    train = [np.full((2, 2), index + 1.0) for index in range(4)]
    short = shared_threshold(train[:2])
    longer = shared_threshold(train)
    assert short == pytest.approx(2.0e3)
    assert longer == pytest.approx(4.0e3)
    assert short == pytest.approx(threshold_from_train(train[:2]))
    assert shared_threshold(train[:2]) != threshold_from_train([train[0] + 10.0])


def test_validation_noise_is_offset_and_reproducible():
    clean = [rng(4, 1, index).normal(size=(6, 2)) for index in range(2)]
    first = add_split_noise(clean, 0.1, 3, 4, False)
    again = add_split_noise(clean, 0.1, 3, 4, False)
    val = add_split_noise(clean, 0.1, 3, 4, True)
    np.testing.assert_allclose(first[0], again[0])
    assert noise_keys(3, 4, 0, True)[-1] == 100
    assert not np.allclose(first[0], val[0])


def test_zero_noise_keeps_oracle_and_noisy_estimators_apart():
    clean = [rng(4, 2, index).normal(size=(51, 2)) for index in range(3)]
    task = _task(clean, 0.0, 0)
    out = evaluate_arrays(clean, clean[:2], clean[:2], task)
    assert [row["estimator"] for row in out["rows"]] == list(ESTIMATORS)
    assert np.allclose(out["operators"]["oracle"], fit(clean, "oracle").O)
    noisy = add_split_noise(clean, 0.0, 0, 0, False)
    assert np.allclose(noisy[0], clean[0])
    assert np.allclose(out["operators"]["ols"], fit(noisy, "ols").O)
    assert err_O_rel(out["operators"]["mc"], out["operators"]["oracle"]) < 1e-8
    assert err_O_rel(out["operators"]["mc_psd"], out["operators"]["oracle"]) < 1e-8
    assert {row["blowup_threshold"] for row in out["rows"]} == {task["threshold"]}


def test_ridge_choice_matches_the_existing_selector_and_ignores_test():
    clean = [rng(4, 3, index).normal(size=(51, 2)) for index in range(3)]
    task = _task(clean, 0.05, 1)
    out = evaluate_arrays(clean, clean[:2], clean[:2], task)
    sigma = float(np.sqrt(out["sigma2"]))
    noisy_train = add_split_noise(clean, sigma, 1, 1, False)
    noisy_val = add_split_noise(clean[:2], sigma, 1, 1, True)
    chosen = select_ridge_lambda(noisy_train, noisy_val, task["threshold"])
    assert out["ridge"]["lam"] == pytest.approx(chosen)
    assert "test" not in inspect.signature(select_ridge_lambda).parameters
    assert len(out["ridge"]["scores"]) == 13


def test_s1_zero_noise_matches_oracle_within_acceptance():
    clean = load_s1_split(1, 10)
    oracle = fit(clean, "oracle").O
    noisy = add_split_noise(clean, 0.0, 0, 0, False)
    assert err_O_rel(fit(noisy, "ols").O, oracle) < 1e-8
    assert err_O_rel(fit(noisy, "mc", sigma2=0.0).O, oracle) < 1e-8
    assert err_O_rel(fit(noisy, "mc_psd", sigma2=0.0).O, oracle) < 1e-8
    assert err_O_rel(fit(noisy, "iv", sigma2=0.0).O, oracle) < 1e-8
    assert err_O_rel(fit(noisy, "iv2", sigma2=0.0).O, oracle) < 1e-8


def test_pairs_stay_inside_trajectories_and_csv_columns_stay_common():
    clean = [rng(4, 4, index).normal(size=(51, 2)) for index in range(3)]
    out = evaluate_arrays(clean, clean[:2], clean[:2], _task(clean, 0.02, 2))
    expected = {pairs_per_estimator(method, 3, 51) for method in ESTIMATORS}
    assert {row["n_pairs"] for row in out["rows"]} == expected
    assert 3 * 51 - 1 not in expected
    exported = csv_row(out["rows"][0])
    assert set(exported) == set(RUN_COLUMNS)
    assert "blowup_threshold" not in exported
    assert "theory_gap_rel" not in exported


def test_paired_gap_follows_replicate_not_row_order():
    case = {"rho": 0.05, "n_train": 4, "rho_index": 3, "replicates": 2}
    rows, operators = _paired_fixture(case)
    rows = list(reversed(rows))
    summary = summary_rows([case], rows, operators, np.zeros((6, 2)))
    found = {(row["estimator"]): row["paired_median"] for row in summary}
    assert found["ols"] == pytest.approx(0.0)
    assert found["mc"] == pytest.approx(2.0)
    again = summary_rows([case], rows, operators, np.zeros((6, 2)))
    assert again[1]["boot_low"] == pytest.approx(summary[1]["boot_low"])
    assert again[1]["boot_high"] == pytest.approx(summary[1]["boot_high"])


def test_failed_fit_stays_nan_and_inf_median_stays_inf():
    case = {"rho": 0.2, "n_train": 4, "rho_index": 5, "replicates": 2}
    rows, operators = _paired_fixture(case)
    target = next(row for row in rows if row["estimator"] == "ols" and row["replicate"] == 0)
    target["status"] = "failed"
    target["err_O_rel"] = float("nan")
    operators[f"rep_rho{case['rho_index']}_n{case['n_train']}_ols_r000"] = np.full((6, 2), np.nan)
    summary = summary_rows([case], rows, operators, np.zeros((6, 2)))
    failed = next(row for row in summary if row["estimator"] == "ols")
    assert failed["n_failed"] == 1
    assert np.isnan(failed["err_O_rel_median"])
    assert np.isnan(failed["bias_O_rel"])
    aggregated = rollout_aggregate([1.0, 2.0], [False, True])
    assert aggregated["rollout_med"] == float("inf")
    assert aggregated["rollout_mean_finite"] == pytest.approx(1.0)
    assert aggregated["diverged_frac"] == pytest.approx(0.5)


def test_quartile_keeps_infinity_and_matches_linear_quantiles():
    finite = np.linspace(0.0, 1.0, 100)
    q25, median, q75 = _quartiles(finite)
    assert median == pytest.approx(np.median(finite))
    assert q25 == pytest.approx(np.quantile(finite, 0.25))
    assert q75 == pytest.approx(np.quantile(finite, 0.75))
    _, inf_median, inf_q75 = _quartiles([1.0] * 75 + [float("inf")] * 25)
    assert inf_median == 1.0
    assert inf_q75 == float("inf")
    exact = _quartiles([1.0, float("inf"), float("inf"), float("inf"), float("inf")])
    assert exact[0] == float("inf")
    assert exact[1] == float("inf")
    assert all(np.isnan(item) for item in _quartiles([1.0, float("nan")]))


def test_existing_e2_directory_is_refused(tmp_path):
    target = tmp_path / "e2"
    target.mkdir()
    with pytest.raises(FileExistsError):
        refuse_existing(target)


def test_smoke_still_thresholds_each_estimator_from_its_own_training_data():
    from noisyopinf import pipeline
    source = inspect.getsource(pipeline._one_row)
    assert "threshold_from_train(train_q)" in source


def _task(clean, rho: float, replicate: int) -> dict:
    truth = fit(clean, "oracle").O
    return {
        "rho": rho, "rho_index": 1, "n_train": len(clean), "replicate": replicate,
        "threshold": 1.0e6, "truth": truth,
    }


def _paired_fixture(case: dict):
    rows, operators = [], {}
    errors = {"ols": (1.0, 10.0), "mc": (4.0, 11.0)}
    for method in ESTIMATORS:
        for replicate, fallback in ((0, 0.2), (1, 0.4)):
            value = errors.get(method, (fallback, fallback + 1.0))[replicate]
            row = {name: 0.0 for name in MEASURES}
            row.update({
                "rho": case["rho"], "n_train": case["n_train"], "estimator": method, "replicate": replicate,
                "status": "ok", "sigma2_true": 0.01, "sigma2_source": "na", "err_O_rel": value,
                "blowup_threshold": 30.0,
            })
            rows.append(row)
            key = f"rep_rho{case['rho_index']}_n{case['n_train']}_{method}_r{replicate:03d}"
            operators[key] = np.full((6, 2), value)
    return rows, operators
