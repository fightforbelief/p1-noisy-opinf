"""原坐标下的离散二次模型迭代。"""

from __future__ import annotations

import numpy as np

from noisyopinf.features import features

BLOWUP_FACTOR = 1.0e3


def threshold_from_train(train_trajs) -> float:
    """发散阈值：10^3 乘训练状态的最大无穷范数。"""
    peak = max(float(np.max(np.abs(np.asarray(traj)))) for traj in train_trajs)
    return BLOWUP_FACTOR * peak


def rollout(O, q0, n_steps, blowup_threshold) -> tuple[np.ndarray, bool]:
    """从 q0 迭代 n_steps 步。返回形状 (n_steps+1, r) 的轨迹和是否发散。

    第 0 行是初值。非有限的新状态记为 NaN，其后也是 NaN。
    超过阈值的有限状态保留原值，其后填 NaN。不裁剪，不重开。
    """
    operator = np.asarray(O, dtype=np.float64)
    initial = np.array(q0, dtype=np.float64, copy=True)
    steps = int(n_steps)
    limit = float(blowup_threshold)
    _check_rollout_args(operator, initial, steps, limit)
    traj = np.empty((steps + 1, initial.shape[0]), dtype=np.float64)
    traj[0] = initial
    if _exceeds(initial, limit):
        traj[1:] = np.nan
        return traj, True
    state = initial
    for step in range(1, steps + 1):
        nxt = features(state[None, :])[0] @ operator
        if not np.isfinite(nxt).all():
            traj[step:] = np.nan
            return traj, True
        traj[step] = nxt
        if float(np.max(np.abs(nxt))) > limit:
            traj[step + 1 :] = np.nan
            return traj, True
        state = traj[step]
    return traj, False


def _check_rollout_args(operator, initial, steps: int, limit: float) -> None:
    if initial.ndim != 1 or operator.ndim != 2 or operator.shape[1] != initial.shape[0]:
        raise ValueError("O must have shape (p, r) and q0 shape (r,)")
    if steps < 0 or not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("n_steps must be non-negative and blowup_threshold positive")


def _exceeds(state: np.ndarray, limit: float) -> bool:
    return (not np.isfinite(state).all()) or float(np.max(np.abs(state))) > limit
