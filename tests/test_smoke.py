"""05 第 7 节的小规模 S2 流程。输出只进临时目录。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from noisyopinf.io_utils import RUN_COLUMNS, read_runs, write_runs
from noisyopinf.pipeline import ESTIMATORS, N_TEST, N_VAL, judgement, prepare_case, evaluate_case
from noisyopinf.protocol import LAMBDAS, noise_replicate

SPEC_COLUMNS = (
    "exp", "system", "rho", "sigma2_true", "sigma2_used", "sigma2_source", "r",
    "n_train", "n_pairs", "basis", "estimator", "lam", "replicate", "seed_keys",
    "err_O_rel", "err_A_rel", "err_H_rel", "err_c_abs", "err_c_scaled",
    "specrad_hat", "specrad_ref", "specrad_err", "ep_violation", "ep_violation_ref",
    "rollout_med", "rollout_mean_finite", "diverged_frac", "proj_floor",
    "gram_min_eig", "gram_cond", "n_clipped", "fit_seconds", "status", "error",
)
PAIRS = {"oracle": 120, "ols": 120, "ridge": 120, "mc": 120, "mc_psd": 120, "iv": 117, "iv2": 114}


def test_undefined_metric_is_not_success():
    row = {"err_O_rel": float("nan"), "err_A_rel": 0.1}
    status, reason = judgement(row, ("err_O_rel", "err_A_rel"))
    assert status == "undefined"
    assert "err_O_rel" in reason
    missing, why = judgement({"err_O_rel": None}, ("err_O_rel",))
    assert missing == "failed" and "err_O_rel" in why
    empty_mean = {"rollout_mean_finite": float("nan"), "diverged_frac": 1.0}
    assert judgement(empty_mean, ("rollout_mean_finite",)) == ("ok", None)


def test_s2_smoke_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("NOISYOPINF_DATA", str(tmp_path / "unused-cache"))
    case = prepare_case(rank=4, n_train=3, n_obs=40, rho=0.05, replicate=0)
    rows = evaluate_case(case)
    path = tmp_path / "runs.csv"
    write_runs(path, rows)
    loaded = read_runs(path)
    _assert_schema(loaded)
    _assert_protocol(case)
    _assert_values(loaded, case)
    official = Path(__file__).resolve().parents[1] / "results" / "e3" / "runs.csv"
    assert not official.exists()


def _assert_schema(loaded: list[dict]) -> None:
    assert len(loaded) == len(ESTIMATORS)
    assert [row["estimator"] for row in loaded] == list(ESTIMATORS)
    assert list(loaded[0]) == list(RUN_COLUMNS)
    for column in SPEC_COLUMNS:
        assert column in RUN_COLUMNS
    assert "rollout_med_h10" in RUN_COLUMNS and "rollout_med_h50" in RUN_COLUMNS


def _assert_protocol(case) -> None:
    again, _, clean = noise_replicate(case.train_u, case.train_u, case.test_u, case.sigma, case.replicate)
    np.testing.assert_allclose(again[0], case.noisy_train_u[0])
    np.testing.assert_array_equal(clean[0], case.test_u[0])
    np.testing.assert_array_equal(case.clean_test_u[0], case.test_u[0])
    assert len(case.noisy_val_u) == N_VAL and len(case.clean_test_u) == N_TEST
    assert not np.allclose(case.noisy_train_u[0], case.train_u[0])
    assert not np.allclose(case.noisy_train_u[0], case.noisy_val_u[0])


def _assert_values(loaded: list[dict], case) -> None:
    for row in loaded:
        method = row["estimator"]
        assert row["exp"] == "smoke" and row["system"] == "s2" and row["basis"] == "clean"
        assert row["rho"] == 0.05 and row["r"] == 4 and row["n_train"] == 3
        assert row["replicate"] == 0 and row["seed_keys"] == "0,1|0,2"
        assert row["n_pairs"] == PAIRS[method]
        assert row["sigma2_true"] == case.sigma2
        assert row["rollout_med_h50"] is None
        if method in {"mc", "mc_psd", "iv", "iv2"}:
            assert row["sigma2_source"] == "true" and row["sigma2_used"] == case.sigma2
        else:
            assert row["sigma2_source"] == "na" and row["sigma2_used"] is None
        if method == "ridge":
            assert row["lam"] in LAMBDAS
        else:
            assert row["lam"] is None
        assert row["status"] == "ok", (method, row["status"], row["error"])
        assert row["error"] is None
        for column in SPEC_COLUMNS:
            if column in {"lam", "sigma2_used", "gram_min_eig", "gram_cond", "error"}:
                continue
            assert row[column] is not None
            if column == "rollout_mean_finite" and row["diverged_frac"] == 1.0:
                assert np.isnan(row[column])
                continue
            if isinstance(row[column], float):
                assert not np.isnan(row[column])
