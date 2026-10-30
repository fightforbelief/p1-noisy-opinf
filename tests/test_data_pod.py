"""数据和 POD 的小规模测试。"""

from __future__ import annotations

import numpy as np

from noisyopinf.data import get_splits
from noisyopinf.pod import estimate_sigma2, mp_median, pod_basis, project, snapshot_matrix
from noisyopinf.seeds import rng


def test_s1_train_prefixes_are_nested(tmp_path, monkeypatch):
    monkeypatch.setenv("NOISYOPINF_DATA", str(tmp_path))
    short = get_splits("s1", 2, 1, 1)
    long = get_splits("s1", 3, 1, 1)
    assert short["train"].indices == [0, 1]
    assert long["train"].indices[:2] == short["train"].indices
    assert short["train"].U is None
    assert short["train"].Q[0].shape == (301, 8)
    assert np.array_equal(short["train"].Q[0], long["train"].Q[0])
    assert np.array_equal(short["train"].Q[1], long["train"].Q[1])
    assert short["val"].Q[0].shape == (301, 8)
    assert short["train"].skipped == []


def test_s1_cache_rejects_stale_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("NOISYOPINF_DATA", str(tmp_path))
    first = get_splits("s1", 1, 0, 0)
    path = tmp_path / "s1_train_0.npz"
    np.savez(path, state=np.zeros((301, 8)), param_hash=np.array("stale"))
    second = get_splits("s1", 1, 0, 0)
    assert np.array_equal(second["train"].Q[0], first["train"].Q[0])


def test_s2_and_s3_shapes(tmp_path, monkeypatch):
    monkeypatch.setenv("NOISYOPINF_DATA", str(tmp_path / "s2"))
    s2 = get_splits("s2", 1, 0, 0)
    assert s2["train"].Q is None
    assert s2["train"].U[0].shape == (201, 256)
    assert np.isfinite(s2["train"].U[0]).all()
    monkeypatch.setenv("NOISYOPINF_DATA", str(tmp_path / "s3"))
    s3 = get_splits("s3", 1, 0, 0)
    assert s3["train"].U[0].shape == (201, 512)
    assert np.isfinite(s3["train"].U[0]).all()


def test_pod_basis_reproduces_rank():
    gen = rng(8, 4)
    axes = np.linalg.qr(gen.normal(size=(20, 3)))[0]
    snaps = (axes @ gen.normal(size=(3, 30))).T
    basis, values = pod_basis([snaps], 3)
    rebuilt = project(snaps, basis) @ basis.T
    assert values.shape == (20,)
    assert values[3] <= 1e-8 * values[0]
    assert np.linalg.norm(rebuilt - snaps) / np.linalg.norm(snaps) < 1e-8


def test_trailing_median_matches_formula():
    snaps = rng(8, 5).normal(size=(16, 8))
    estimate, diagnostics = estimate_sigma2([snaps], method="trailing_median")
    eigs = np.linalg.svd(snapshot_matrix([snaps]), compute_uv=False) ** 2 / 16.0
    assert abs(estimate - float(np.median(eigs[4:]))) < 1e-12
    assert diagnostics["K_tot"] == 16


def test_mp_median_lies_in_bulk():
    beta = 0.25
    median = mp_median(beta)
    root = np.sqrt(beta)
    assert (1.0 - root) ** 2 < median < (1.0 + root) ** 2


def test_sigma2_trailing():
    n, k_tot, rank, sigma = 128, 2000, 5, 0.3
    factors = rng(8, 2).normal(size=(n, rank))
    coeff = rng(8, 2).normal(size=(rank, k_tot))
    clean = (factors @ coeff).T
    noisy = clean + rng(8, 3).normal(scale=sigma, size=clean.shape)
    estimate, diagnostics = estimate_sigma2(
        [noisy], method="trailing_median", clean_snapshots_list=[clean]
    )
    assert abs(estimate - sigma**2) / sigma**2 < 0.05
    assert diagnostics["clean_half_eigenvalue_over_sigma2"] < 0.01
