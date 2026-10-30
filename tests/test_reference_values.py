"""02 第 3.6 节的预验证数值。"""

from __future__ import annotations

import numpy as np

from noisyopinf.estimators import fit
from noisyopinf.features import features, quad_index
from noisyopinf.noise import add_white
from noisyopinf.seeds import rng
from noisyopinf.systems.galerkin_burgers import basis, linear_diag, step_map


def _operator():
    phi, phi_x = basis(2, 64)
    diag = linear_diag(0.02, 4)
    rank = 4
    width = rank + len(quad_index(rank)) + 1
    points = rng(12, 9).standard_normal(size=(5 * width, rank))
    images = step_map(points, phi, phi_x, diag, 0.005)
    operator, _, _, _ = np.linalg.lstsq(features(points), images, rcond=None)
    return operator, phi, phi_x, diag


def _trajectories(operator_parts):
    _, phi, phi_x, diag = operator_parts
    trajs = []
    for index in range(30):
        gen = rng(12, 1, index)
        state = np.zeros(4, dtype=np.float64)
        for wave in range(1, 3):
            scale = 0.5 / wave
            state[2 * (wave - 1)] = gen.normal(0.0, scale)
            state[2 * (wave - 1) + 1] = gen.normal(0.0, scale)
        traj = np.empty((301, 4), dtype=np.float64)
        traj[0] = state
        for time in range(300):
            state = step_map(state, phi, phi_x, diag, 0.005)
            traj[time + 1] = state
        trajs.append(traj)
    return trajs


def _bias(mean: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(mean - truth) / np.linalg.norm(truth))


def test_reference_values():
    parts = _operator()
    truth = parts[0]
    trajs = _trajectories(parts)
    sigma = 0.1 * float(np.sqrt(np.mean(np.vstack(trajs) ** 2)))
    totals = {name: np.zeros_like(truth) for name in ("ols", "mc", "iv")}
    for rep in range(200):
        gen = rng(12, 2, rep)
        noisy = [add_white(traj, sigma, gen) for traj in trajs]
        for name in totals:
            totals[name] += fit(noisy, name, sigma2=sigma**2).O
    ols_bias = _bias(totals["ols"] / 200.0, truth)
    assert abs(ols_bias - 0.0683) <= 0.3 * 0.0683
    assert _bias(totals["mc"] / 200.0, truth) < 3.0 * 0.0006
    assert _bias(totals["iv"] / 200.0, truth) < 3.0 * 0.0006
