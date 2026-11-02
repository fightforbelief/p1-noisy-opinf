"""一次重复内的噪声划分和 ridge 的 λ 选择。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from noisyopinf.estimators import fit
from noisyopinf.metrics import rollout_err_h
from noisyopinf.noise import add_white
from noisyopinf.rollout import rollout
from noisyopinf.seeds import rng

LAMBDAS = tuple(10.0 ** exponent for exponent in range(-10, 3))


def noise_replicate(train, val, test, sigma: float, replicate: int):
    """训练集和验证集独立加噪。测试轨迹保持干净。同一 replicate 可复现。"""
    noisy_train = _noise_group(train, sigma, (int(replicate), 1))
    noisy_val = _noise_group(val, sigma, (int(replicate), 2))
    clean_test = [np.array(traj, dtype=np.float64, copy=True) for traj in test]
    return noisy_train, noisy_val, clean_test


def select_ridge_lambda(noisy_train, noisy_val, blowup_threshold: float) -> float:
    """在带噪验证轨迹上选 λ。并列时取网格中较大的 λ。"""
    scores = [_validation_score(noisy_train, noisy_val, lam, blowup_threshold) for lam in LAMBDAS]
    return float(LAMBDAS[larger_lambda_index(scores)])


def larger_lambda_index(scores) -> int:
    """并列时保留更靠后、也就是更大的 λ。"""
    chosen = 0
    for index, score in enumerate(scores):
        if score <= scores[chosen]:
            chosen = index
    return chosen


def _noise_group(trajs, sigma: float, key: tuple[int, int]):
    return [add_white(traj, sigma, rng(*key, index)) for index, traj in enumerate(trajs)]


def _validation_score(train, val, lam: float, blowup_threshold: float) -> float:
    operator = fit(train, "ridge", lam=lam).O
    errors = [_one_score(operator, traj, blowup_threshold) for traj in val]
    return float(np.median(np.asarray(errors, dtype=np.float64)))


def _one_score(operator: np.ndarray, traj: np.ndarray, blowup_threshold: float) -> float:
    if not np.isfinite(operator).all():
        return float("inf")
    steps = int(traj.shape[0] - 1)
    predicted, diverged = rollout(operator, traj[0], steps, blowup_threshold)
    if diverged:
        return float("inf")
    return rollout_err_h(predicted, traj, steps)
