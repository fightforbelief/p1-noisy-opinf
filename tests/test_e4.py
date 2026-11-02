"""E4 的网格、共用噪声、选秩和 SNR 规则。"""

from __future__ import annotations

import numpy as np
import pytest

from noisyopinf.e1 import refuse_existing
from noisyopinf.e4 import (
    ESTIMATORS,
    RANKS,
    aggregate_snr,
    choose_rank,
    evaluate_replicate,
    load_config,
    mode_energies,
    noise_keys,
    planned_runs,
    select_from_curves,
    snr_rank,
)
from noisyopinf.pod import pod_basis
from noisyopinf.systems.config import project_root


def test_e4_grid_has_1200_rows():
    config = load_config()
    assert config["experiment_id"] == 4
    assert config["tie_rule"] == "smaller_r"
    assert planned_runs(config) == 1200
    assert list(config["r"]) == list(RANKS)
    assert config["estimators"] == list(ESTIMATORS)


def test_noise_keys_use_experiment_4():
    assert noise_keys(2, 3, 4, False) == (2, 4, 4, 2, 3, 4)
    assert noise_keys(2, 3, 4, True) == (2, 4, 4, 2, 3, 104)


def test_snr_uses_max_index_and_keeps_an_empty_set():
    energies = np.array([5.0, 0.2, 4.0, 0.1])
    chosen, passed = snr_rank(energies, 1.0, 1.0)
    assert chosen == 3
    assert passed == [1, 3]
    empty, nothing = snr_rank(energies, 10.0, 1.0)
    assert empty is None
    assert nothing == []
    assert aggregate_snr([None, 6, 4, 8]) == (6, 1)
    assert aggregate_snr([5, 7]) == (5, 0)
    assert aggregate_snr([None, None]) == (None, 2)


def test_rank_ties_take_the_smaller_rank_and_inf_loses():
    chosen, score, ties = choose_rank({16: 1.0, 4: 1.0, 8: 2.0})
    assert chosen == 4
    assert score == 1.0
    assert ties == 2
    chosen, score, _ties = choose_rank({4: np.inf, 8: 0.2, 12: np.nan})
    assert chosen == 8
    assert score == 0.2


def test_selection_uses_only_its_own_curve():
    curves = []
    for replicate in (0, 1):
        for rank, clean, noisy, test in ((4, 0.5, 0.2, 0.9), (8, 0.1, 0.4, 0.3), (16, 0.4, 0.1, 0.2)):
            curves.append({"rho": 0.05, "r": rank, "estimator": "ols", "replicate": replicate, "clean_val_med": clean, "noisy_val_med": noisy, "test_med": test})
    selected = {item["criterion"]: item["r_selected"] for item in select_from_curves(curves)}
    assert selected == {"val": 8, "noisyval": 16, "test": 16}


def test_prefix_of_one_svd_matches_each_truncated_basis():
    matrix = [np.random.default_rng(3).normal(size=(30, 20)) for _ in range(4)]
    full, _values = pod_basis(matrix, 16)
    for rank in (4, 8, 16):
        fresh, _singular = pod_basis(matrix, rank)
        assert np.allclose(full[:, :rank], fresh)


def test_existing_result_directory_is_refused():
    target = project_root() / "results" / "_e4_refuse_probe"
    target.mkdir(parents=True, exist_ok=True)
    try:
        with pytest.raises(FileExistsError):
            refuse_existing(target)
    finally:
        target.rmdir()


def test_shared_noise_sigma_and_validation_paths(bundle):
    result, other = bundle
    assert result["truncation_gap"] < 1e-8
    assert len(result["energies"]) == 16
    assert abs(result["full_order_noise_rms"] - result["sigma"]) / result["sigma"] < 0.15
    used = {row["sigma2_used"] for payload in result["ranks"].values() for row in payload["rows"] if row["estimator"] in {"mc", "iv2"} and row["status"] == "ok"}
    assert len(used) == 1
    hashes = {payload["reference_hash"] for payload in result["ranks"].values()}
    assert len(hashes) == len(result["ranks"])
    for payload in result["ranks"].values():
        ols = next(row for row in payload["rows"] if row["estimator"] == "ols")
        ridge = next(row for row in payload["rows"] if row["estimator"] == "ridge")
        assert ols["sigma2_source"] == "na" and ols["sigma2_used"] is None
        assert ridge["sigma2_source"] == "na"
        assert {detail["threshold"] for detail in payload["details"]} == {payload["threshold"]}
    same = result["ranks"][4]
    changed = other["ranks"][4]
    assert same["ridge"]["scores"] == changed["ridge"]["scores"]
    assert same["reference_hash"] == changed["reference_hash"]
    assert any(row["clean_val_med"] != row["noisy_val_med"] for row in same["rows"] if row["status"] == "ok")


@pytest.fixture(scope="module")
def bundle():
    train, val, test = _lifted()
    original = evaluate_replicate(train, val, test, 0.2, 5, 0)
    changed = [traj + 0.05 for traj in test]
    other = evaluate_replicate(train, val, changed, 0.2, 5, 0)
    return original, other


def _lifted():
    generator = np.random.default_rng(5)
    dimension, rank, length = 20, 16, 60
    basis, _ = np.linalg.qr(generator.normal(size=(dimension, rank)))
    linear = np.diag(np.linspace(0.9, 0.6, rank))

    def group(seed: int, count: int):
        local = np.random.default_rng(seed)
        rows = []
        for _ in range(count):
            reduced = np.zeros((length, rank))
            reduced[0] = 0.15 * local.normal(size=rank)
            for step in range(length - 1):
                reduced[step + 1] = linear @ reduced[step]
            rows.append(reduced @ basis.T)
        return rows

    return group(1, 8), group(2, 4), group(3, 4)
