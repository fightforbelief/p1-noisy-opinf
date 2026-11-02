"""结果文件的列、空值和覆盖规则。"""

from __future__ import annotations

import numpy as np

from noisyopinf.io_utils import (
    RUN_COLUMNS,
    read_meta,
    read_operators,
    read_runs,
    write_meta,
    write_operators,
    write_runs,
    runtime_facts,
)


def _row():
    row = {column: None for column in RUN_COLUMNS}
    row.update({
        "exp": "e2",
        "system": "s1",
        "rho": 0.1,
        "estimator": "ols",
        "replicate": 3,
        "seed_keys": "2-3",
        "err_O_rel": 0.25,
        "rollout_med": np.inf,
        "rollout_mean_finite": np.nan,
        "status": "failed",
        "error": "singular design",
    })
    return row


def test_csv_round_trip_keeps_blanks_and_inf(tmp_path):
    path = tmp_path / "results" / "e2" / "runs.csv"
    write_runs(path, [_row()])
    loaded = read_runs(path)[0]
    assert list(loaded) == list(RUN_COLUMNS)
    assert loaded["sigma2_true"] is None
    assert np.isnan(loaded["rollout_mean_finite"])
    assert loaded["rollout_med"] == np.inf
    assert loaded["status"] == "failed"
    assert loaded["err_O_rel"] == 0.25
    assert "0" not in path.read_text(encoding="utf-8").splitlines()[1].split(",")[2:4]


def test_unknown_field_is_rejected(tmp_path):
    path = tmp_path / "runs.csv"
    row = _row()
    row["theory_gap_rel"] = 0.5
    try:
        write_runs(path, [row])
    except ValueError as exc:
        assert "theory_gap_rel" in str(exc)
        assert not path.exists()
    else:
        raise AssertionError("unknown field was dropped")


def test_existing_file_is_not_overwritten(tmp_path):
    path = tmp_path / "runs.csv"
    write_runs(path, [_row()])
    original = path.read_text(encoding="utf-8")
    try:
        write_runs(path, [_row()])
    except FileExistsError:
        assert path.read_text(encoding="utf-8") == original
        assert not path.with_suffix(".csv.tmp").exists()
    else:
        raise AssertionError("overwrite was allowed")


def test_meta_and_operator_round_trip(tmp_path):
    meta_path = tmp_path / "meta.json"
    meta = {"git_commit": "abc", "python": "3.12.10", "cpu_count": 8, "elapsed_seconds": 1.5}
    write_meta(meta_path, meta)
    assert read_meta(meta_path) == meta
    operator_path = tmp_path / "operators.npz"
    arrays = {"ols": np.arange(4.0).reshape(2, 2), "mc": np.eye(2)}
    write_operators(operator_path, arrays)
    loaded = read_operators(operator_path)
    np.testing.assert_array_equal(loaded["ols"], arrays["ols"])
    assert not operator_path.with_suffix(".npz.tmp").exists()


def test_runtime_facts_exclude_numeric_results():
    facts = runtime_facts()
    assert facts["python"]
    assert "numpy" in facts["dependencies"]
    assert "err_O_rel" not in facts
