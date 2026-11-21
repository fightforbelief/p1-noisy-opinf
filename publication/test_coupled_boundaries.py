"""Analytic controls outside the equally spaced one-step PDE grid."""
import math

import numpy as np
import pytest

from derivative_study import observed


@pytest.mark.parametrize("nodes", [(0.5,), (0.5, 2.5), (0.5, 1.5, 3.0)])
def test_irregular_nodes_and_unequal_response_horizon(nodes):
    m = len(nodes)
    t = np.linspace(-1, 1, 91)
    q = t[:, None]**4
    z = np.column_stack([np.ones(len(t)), q])
    u, r = np.linalg.qr(z)
    derivatives = [math.factorial(4)/math.factorial(4-i)*t**(4-i) for i in range(m+1)]
    j, _ = np.linalg.qr(np.column_stack([np.ones(len(t)), *derivatives[:-1]]))
    g = derivatives[-1]-j@(j.T@derivatives[-1])
    ell = np.linalg.solve(r.T, np.column_stack([np.zeros(len(t)), g]).T).T
    c = 1.75
    alpha = np.prod(nodes)/math.factorial(m)
    beta = np.prod(c+np.array(nodes))/math.factorial(m)
    leading = -alpha*(beta-alpha)*ell.T@g[:, None]
    errors = []
    for h in (0.01, 0.005, 0.0025):
        w = np.column_stack([np.ones(len(t)), *((t-a*h)**4 for a in nodes)])
        actual, *_ = observed(u, q, ((t+c*h)**4)[:,None], w)
        errors.append(np.linalg.norm(actual/h**(2*m)-leading)/np.linalg.norm(leading))
    assert errors[-1] < errors[1] < errors[0]
    assert errors[-1] < 0.1


def test_closed_quadratic_vector_field_has_cubic_single_lag_displacement():
    # q'=-q^2 is in the dictionary. Its exact flow is q/(1+s*q).
    q = np.linspace(0.1,1.1,101)[:,None]
    def design(x):
        return np.column_stack([x, x*x, np.ones(len(x))])
    u, r = np.linalg.qr(design(q))
    g = q**3-u@(u.T@(q**3))
    derivative = np.column_stack([-q*q,-2*q**3,np.zeros(len(q))])
    ell = np.linalg.solve(r.T, (derivative-u@(u.T@derivative)).T).T
    leading = -ell.T@g
    errors = []
    for h in (0.005,0.0025,0.00125):
        actual, *_ = observed(u, q, q/(1+h*q), design(q/(1-h*q)))
        errors.append(np.linalg.norm(actual/h**3-leading)/np.linalg.norm(leading))
    assert errors[-1] < errors[1] < errors[0] and errors[-1] < 0.025
