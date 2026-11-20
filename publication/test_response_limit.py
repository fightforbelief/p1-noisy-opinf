"""Check the response-specific coefficient, including its sign, by direct refitting."""
import math
import numpy as np
import pytest


@pytest.mark.parametrize("m", [1, 2, 3])
def test_quartic_response_leading_matrix(m):
    t = np.linspace(-1, 1, 81)
    z = np.column_stack([np.ones_like(t), t**4])
    u, r = np.linalg.qr(z)
    y = np.column_stack([t, t**2, t**3])
    ols = np.linalg.lstsq(z, y, rcond=None)[0]
    e = y-z@ols
    derivatives = [math.factorial(4)/math.factorial(4-j)*t**(4-j) for j in range(m+1)]
    jet, _ = np.linalg.qr(np.column_stack([z[:, 0], *derivatives[:-1]]))
    last = np.column_stack([np.zeros_like(t), derivatives[-1]])
    normal = (last-jet@(jet.T@last))@np.linalg.inv(r)
    leading = -u@(normal.T@e)
    errors = []
    for h in (0.02, 0.01, 0.005):
        past = [(t-j*h)**4 for j in range(1, m+1)]
        w, _ = np.linalg.qr(np.column_stack([z[:, 0], *past]))
        iv = np.linalg.lstsq(w.T@z, w.T@y, rcond=None)[0]
        scaled = z@(iv-ols)/h**m
        errors.append(np.linalg.norm(scaled-leading)/np.linalg.norm(leading))
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] < 0.10
