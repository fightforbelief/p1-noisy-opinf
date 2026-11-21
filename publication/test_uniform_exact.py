"""Independent rational-arithmetic checks of the uniform signed law."""
from fractions import Fraction as Q
from math import factorial, prod

import numpy as np
import pytest


def solve(a, b):
    """Gauss-Jordan solve over rationals for the small Gram systems below."""
    n = len(a)
    aug = np.column_stack([a, b]).copy()
    for k in range(n):
        pivot = next(i for i in range(k, n) if aug[i, k])
        aug[[k, pivot]] = aug[[pivot, k]]
        aug[k] = aug[k] / aug[k, k]
        for i in range(n):
            if i != k:
                aug[i] -= aug[i, k] * aug[k]
    return aug[:, n:]


def project(x, y):
    return x @ solve(x.T @ x, x.T @ y)


def displacement(z, w, y):
    projected = project(w, z)
    return z @ solve(z.T @ projected, projected.T @ y) - project(z, y)


def anchors():
    return np.array([Q(i, 9) for i in range(1, 12)])[:, None]


@pytest.mark.parametrize("nodes", [(Q(1, 2),), (Q(1, 2), Q(2)), (Q(1, 2), Q(2), Q(3))])
@pytest.mark.parametrize("path", ["short-horizon", "long-horizon"])
def test_independent_scales_against_exact_fits(nodes, path):
    t = anchors()
    one = np.full(t.shape, Q(1))
    m = len(nodes)
    z = np.column_stack([one, t**4])
    jet = np.column_stack([one, *(t**(4-j) for j in range(m))])
    derivative = (factorial(4) // factorial(4-m)) * t**(4-m)
    g = derivative - project(jet, derivative)
    moment = np.array([[Q(0)], [(g.T @ g).item()]])
    direction = -z @ solve(z.T @ z, moment)
    errors = []
    for denominator in [128, 256, 512, 1024]:
        eps = Q(1, denominator)
        h, tau = (eps, eps**2) if path == "short-horizon" else (eps**2, eps)
        w = np.column_stack([one, *((t-a*h)**4 for a in nodes)])
        actual = displacement(z, w, (t+tau)**4)
        polynomial = prod(tau+a*h for a in nodes) - prod(a*h for a in nodes)
        scalar = prod(nodes)*h**m*polynomial/factorial(m)**2
        diff = actual - scalar*direction
        errors.append(((diff.T @ diff) / (actual.T @ actual)).item())
    # Exact squared relative errors. At least first-order decay up to a slack
    # factor between successive halvings, including ratios 1/1024 and 1024.
    assert all(0 < fine < coarse * Q(1, 2) for coarse, fine in zip(errors, errors[1:]))
    assert errors[-1] < Q(1, 1000)


def test_rank_deficient_jet_can_miss_a_limiting_direction():
    t = anchors()
    one = np.full(t.shape, Q(1))
    z = np.column_stack([one, t**4, t**3])
    jet_basis = np.column_stack([one, t**4, t**3, t**2])
    assert np.any(6*t - project(jet_basis, 6*t))
    h, tau = Q(1, 16), Q(1, 32)
    w = np.column_stack([one, (t-h)**4, (t-h)**3, (t-2*h)**4, (t-2*h)**3])
    assert np.array_equal(project(w, z), z)
    assert not np.any(displacement(z, w, (t+tau)**3))
