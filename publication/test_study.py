import numpy as np
from study import allen_cahn, fit_geometry, rows


def test_nested_spaces_and_direct_2sls():
    gen = np.random.default_rng(304)
    z, y = gen.normal(size=(120, 7)), gen.normal(size=(120, 2))
    u, tri = np.linalg.qr(z)
    ols = np.linalg.lstsq(z, y, rcond=None)[0]
    residual = y-z@ols
    w = z + gen.normal(size=z.shape)
    previous = np.inf
    for instruments in [w, np.column_stack([w, gen.normal(size=(120, 7))]), np.column_stack([w, z])]:
        delta, shift, bound, _ = fit_geometry(u, tri, y, ols, residual, instruments)
        q, _ = np.linalg.qr(instruments)
        direct = np.linalg.lstsq(q.T@z, q.T@y, rcond=None)[0]
        np.testing.assert_allclose(direct, ols+delta, atol=1e-12)
        assert shift < bound+1e-12 and bound <= previous+1e-12
        previous = bound
    assert bound < 1e-12


def test_rows_and_closed_response():
    x = np.arange(2*15).reshape(2, 15, 1)
    assert rows(x, 10, 2).ravel().tolist() == [8, 9, 10, 11, 23, 24, 25, 26]
    z = np.random.default_rng(22).normal(size=(70, 4))
    u, tri = np.linalg.qr(z)
    coef = np.ones((4, 2))
    delta, _, _, _ = fit_geometry(u, tri, z@coef, coef, np.full((70, 2), 1e-15), z)
    assert np.linalg.norm(delta) < 1e-13


def test_allen_cahn_refinement():
    for index in (91, 92, 93):
        coarse = allen_cahn(index, 1)
        fine = allen_cahn(index, 1, refinement=2)[:, ::2]
        assert np.max(np.linalg.norm(coarse-fine, axis=1)/np.linalg.norm(fine, axis=1)) < 1e-4
