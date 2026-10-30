"""估计器和理论测试共用的精确二次系统。"""

from __future__ import annotations

import numpy as np

from noisyopinf.features import features, quad_index
from noisyopinf.seeds import rng


def quadratic_operator(key: tuple[int, ...] = (11, 0), h_scale: float = 0.02) -> np.ndarray:
    """r=3，A 的谱半径为 0.9，H 和 c 使用固定缩放。"""
    gen = rng(*key)
    rank = 3
    raw = gen.normal(size=(rank, rank))
    radius = float(np.max(np.abs(np.linalg.eigvals(raw))))
    matrix = raw * (0.9 / radius)
    pairs = quad_index(rank)
    operator = np.zeros((rank + len(pairs) + 1, rank), dtype=np.float64)
    operator[:rank] = matrix.T
    operator[rank:rank + len(pairs)] = (h_scale * gen.normal(size=(rank, len(pairs)))).T
    operator[-1] = 0.01 * gen.normal(size=rank)
    return operator


def rollout(operator: np.ndarray, n_traj: int, steps: int, key: tuple[int, ...]) -> list[np.ndarray]:
    """从 N(0, 0.3^2) 初值生成精确二次轨迹。"""
    gen = rng(*key)
    rank = operator.shape[1]
    trajs = []
    for _ in range(n_traj):
        state = 0.3 * gen.normal(size=rank)
        traj = np.empty((steps + 1, rank), dtype=np.float64)
        traj[0] = state
        for time in range(steps):
            state = features(state[None, :])[0] @ operator
            traj[time + 1] = state
        trajs.append(traj)
    return trajs


def relative_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth))
