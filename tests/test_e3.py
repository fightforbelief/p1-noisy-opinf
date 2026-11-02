"""E3 的网格、POD 参考、全阶噪声和逐轨迹汇总。"""

from __future__ import annotations

import numpy as np
import pytest

from noisyopinf.data import parameter_hash
from noisyopinf.e1 import refuse_existing
from noisyopinf.e3 import (
    _one_row,
    _quartiles,
    add_full_order_noise,
    evaluate_replicate,
    expand_cases,
    load_config,
    noise_keys,
    parameter_digest,
    planned_runs,
    row_plan,
    same_reference,
    sigma2_for_source,
    summary_rows,
)
from noisyopinf.estimators import fit
from noisyopinf.metrics import proj_floor
from noisyopinf.pod import pod_basis, project


def test_e3_grid_has_3600_rows():
    config = load_config()
    cases = expand_cases(config)
    plan = row_plan()
    assert config["experiment_id"] == 3
    assert len(cases) == 12
    assert len(plan) == 15
    assert planned_runs(config) == 3600
    assert sum(source == "na" for _method, source in plan) == 3
    assert sum(source != "na" for _method, source in plan) == 12
    assert cases[0]["rho_index"] == 0
    assert {case["basis"] for case in cases} == {"clean", "noisy"}


def test_sigma_source_does_not_replace_missing_estimate():
    with pytest.raises(ValueError):
        sigma2_for_source("est_trailing", 0.25, {"est_trailing": {"value": float("nan")}})
    assert sigma2_for_source("true", 0.0, {}) == 0.0
    assert sigma2_for_source("na", 1.0, {}) is None


def test_quartiles_keep_inf_and_nan():
    _low, median, high = _quartiles([1.0, np.inf, np.inf])
    assert median == np.inf
    assert high == np.inf
    assert _quartiles([1.0, 2.0, np.inf])[2] == np.inf
    assert all(np.isnan(value) for value in _quartiles([1.0, np.nan]))


def test_existing_result_directory_is_refused():
    from noisyopinf.systems.config import project_root

    target = project_root() / "results" / "_e3_refuse_probe"
    target.mkdir(parents=True, exist_ok=True)
    try:
        with pytest.raises(FileExistsError):
            refuse_existing(target)
    finally:
        target.rmdir()


def test_noise_keys_use_experiment_3():
    assert noise_keys(1, 2, 3, False) == (2, 4, 3, 1, 2, 3)
    assert noise_keys(1, 2, 3, True) == (2, 4, 3, 1, 2, 103)
    assert parameter_digest() == parameter_hash("s2")


def test_failed_estimate_keeps_the_row():
    train, _val, test = _lifted()
    basis, _ = pod_basis(train, 3)
    clean_q = [project(traj, basis) for traj in train]
    reference = fit(clean_q, "ols").O
    estimates = {"est_trailing": {"value": float("nan")}, "est_gd": {"value": float("nan")}}
    row, operator, details = _one_row(
        "mc", "est_trailing", clean_q, clean_q, test, basis, reference, 0.1, 10.0,
        0.0, 0, 0, "clean", 0.0, estimates, None,
    )
    assert row["status"] == "failed"
    assert row["sigma2_used"] is None
    assert len(details) == len(test)
    assert np.isnan(operator).all()


def test_zero_noise_bases_agree_and_estimates_are_not_forced():
    train, val, test = _lifted()
    quiet = evaluate_replicate(train, val, test, 0.0, 0, 0, rank=3)
    clean = quiet["bases"]["clean"]
    noisy = quiet["bases"]["noisy"]
    assert np.allclose(clean["basis"], noisy["basis"], atol=1e-8, rtol=1e-8)
    assert np.allclose(clean["reference"], noisy["reference"], atol=1e-8, rtol=1e-8)
    assert abs(clean["threshold"] - noisy["threshold"]) <= 1e-8
    estimate = quiet["estimates"]["est_trailing"]["value"]
    row = _row(clean, "mc", "est_trailing")
    truth = _row(clean, "mc", "true")
    assert truth["sigma2_used"] == 0.0
    if np.isfinite(estimate):
        assert row["sigma2_used"] == estimate
    else:
        assert row["status"] == "failed"
        assert row["sigma2_used"] is None


def test_noise_basis_reference_and_shared_sigma(bundle):
    train, _val, test = _lifted()
    loud, other = bundle
    clean = loud["bases"]["clean"]
    noisy = loud["bases"]["noisy"]
    assert len(clean["rows"]) == 15
    assert [row["sigma2_source"] for row in clean["rows"]].count("na") == 3
    assert clean["reference_hash"] != noisy["reference_hash"]
    assert not same_reference([clean["reference"], noisy["reference"]])
    assert _row(clean, "oracle", "na")["err_O_rel"] < 1e-8
    assert _row(noisy, "oracle", "na")["err_O_rel"] < 1e-8
    assert clean["orthogonality"] < 1e-10
    assert noisy["orthogonality"] < 1e-10
    _assert_threshold(train, clean)
    _assert_threshold(train, noisy)
    assert abs(clean["proj_floor"] - proj_floor(clean["basis"], np.vstack(test))) <= 1e-10
    residual = np.vstack(add_full_order_noise(train, loud["sigma"], 4, 0, False)) - np.vstack(train)
    basis = clean["basis"]
    outside = residual - residual @ basis @ basis.T
    assert np.linalg.norm(outside) > 0.5 * np.linalg.norm(residual)
    assert abs(loud["full_order_noise_rms"] - loud["sigma"]) / loud["sigma"] < 0.05
    for method, source in row_plan():
        if source == "na":
            continue
        assert _row(clean, method, source)["sigma2_used"] == _row(noisy, method, source)["sigma2_used"]
    assert clean["ridge"]["lam"] == other["bases"]["clean"]["ridge"]["lam"]
    assert clean["ridge"]["scores"] == other["bases"]["clean"]["ridge"]["scores"]
    assert np.array_equal(clean["reference"], other["bases"]["clean"]["reference"])
    assert clean["rows"][0]["seed_keys"] == "2,4,3,4,0,traj|val+100"
    length = train[0].shape[0]
    ols = _row(clean, "ols", "na")
    assert ols["status"] == "ok"
    assert ols["n_pairs"] == len(train) * (length - 1)
    _assert_details(clean)
    _assert_details(noisy)


def test_noisy_basis_bias_is_not_pooled_across_references(bundle):
    loud, _other = bundle
    second = evaluate_replicate(*_lifted(), 0.2, 4, 1, rank=3)
    rows, operators = [], {}
    for item in (loud, second):
        for payload in item["bases"].values():
            rows.extend(payload["rows"])
            operators.update(payload["operators"])
    cases = [
        {"rho": 0.2, "basis": "clean", "rho_index": 4, "replicates": 2},
        {"rho": 0.2, "basis": "noisy", "rho_index": 4, "replicates": 2},
    ]
    first = summary_rows(cases, rows, operators)
    second_pass = summary_rows(cases, rows, operators)
    assert first[0]["boot_low"] == second_pass[0]["boot_low"]
    clean = next(item for item in first if item["basis"] == "clean" and item["estimator"] == "ols")
    noisy = next(item for item in first if item["basis"] == "noisy" and item["estimator"] == "ols")
    assert np.isfinite(clean["bias_O_rel"])
    assert np.isnan(noisy["bias_O_rel"])
    assert np.isnan(noisy["sd_O_rel"])


@pytest.fixture(scope="module")
def bundle():
    train, val, test = _lifted()
    loud = evaluate_replicate(train, val, test, 0.2, 4, 0, rank=3)
    changed = [traj + 0.03 for traj in test]
    other = evaluate_replicate(train, val, changed, 0.2, 4, 0, rank=3)
    return loud, other


def _lifted():
    generator = np.random.default_rng(11)
    dimension, rank, length = 16, 3, 60
    basis, _ = np.linalg.qr(generator.normal(size=(dimension, rank)))
    linear = np.diag([0.9, 0.85, 0.8])

    def group(seed: int, count: int):
        local = np.random.default_rng(seed)
        rows = []
        for _ in range(count):
            reduced = np.zeros((length, rank))
            reduced[0] = 0.2 * local.normal(size=rank)
            for step in range(length - 1):
                reduced[step + 1] = linear @ reduced[step]
            rows.append(reduced @ basis.T)
        return rows

    return group(1, 6), group(2, 4), group(3, 4)


def _row(payload, method, source):
    return next(row for row in payload["rows"] if row["estimator"] == method and row["sigma2_source"] == source)


def _assert_threshold(train, payload):
    projected = [project(traj, payload["basis"]) for traj in train]
    limit = 1000.0 * max(float(np.max(np.abs(traj))) for traj in projected)
    assert abs(limit - payload["threshold"]) <= 1e-8
    assert {item["threshold"] for item in payload["details"]} == {payload["threshold"]}


def _assert_details(payload):
    grouped = {}
    for item in payload["details"]:
        grouped.setdefault((item["estimator"], item["sigma2_source"]), []).append(item)
    for row in payload["rows"]:
        if row["status"] != "ok":
            continue
        chosen = sorted(grouped[(row["estimator"], row["sigma2_source"])], key=lambda item: item["traj"])
        values = np.asarray([np.inf if item["diverged"] else item["err_full"] for item in chosen])
        finite = values[np.isfinite(values)]
        assert _same(row["rollout_med"], float(np.median(values)))
        assert _same(row["diverged_frac"], float(np.mean([item["diverged"] for item in chosen])))
        if finite.size:
            assert _same(row["rollout_mean_finite"], float(np.mean(finite)))
        assert _same(row["rollout_med_h10"], float(np.median([item["err_h10"] for item in chosen])))


def _same(left, right) -> bool:
    if np.isnan(left) and np.isnan(right):
        return True
    if np.isposinf(left) and np.isposinf(right):
        return True
    return abs(float(left) - float(right)) <= 1e-12
