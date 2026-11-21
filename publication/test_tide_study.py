import numpy as np
from tide_study import anchors, energy_coefficients, factor, fitted


def test_resolved_periodic_tide_has_exact_single_lag_shift_and_zero_two_lag_shift():
    t = np.arange(1600)
    omega = 2*np.pi/120
    x = 1.4 + np.cos(omega*t)
    ix = np.arange(120, 1320)
    h, tau = 30, 60
    _, _, ols, iv, _ = fitted(x, ix, 1, h, tau)
    np.testing.assert_allclose(iv[1]-ols[1], -np.tan(omega*5)*np.sin(omega*10), atol=1e-12)
    assert np.linalg.norm(fitted(x, ix, 2, h, tau)[-1]) < 1e-11
    a, b = energy_coefficients(x, ix, 60)
    np.testing.assert_allclose(a, (10*omega)**2, rtol=.001)
    assert b < 1e-20
    assert factor(2, 60, 60) == 4*factor(2, 60, 60, True)


def test_gaps_never_enter_calibration_or_the_scored_windows():
    x = np.arange(500, dtype=float)
    x[200] = np.nan
    for calibration in (True,False):
        ix = anchors(x, calibration)
        assert 200 not in ix and 230 not in ix and 185 not in ix
        assert (201 in ix) != calibration
