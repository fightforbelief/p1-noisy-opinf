"""二次特征、字典序和 Hermite 修正。"""

from __future__ import annotations

import numpy as np


def quad_index(r: int) -> list[tuple[int, int]]:
    """字典序 (i, j)，0 <= i <= j <= r-1。"""
    return [(i, j) for i in range(r) for j in range(i, r)]


def diag_quad_rows(r: int) -> np.ndarray:
    """对角二次项 q_i^2 在 z 中的行号，形状 (r,)。"""
    rows = [r + pos for pos, (i, j) in enumerate(quad_index(r)) if i == j]
    return np.asarray(rows, dtype=int)


def exponents(r: int) -> np.ndarray:
    """每个特征的指数向量，形状 (p, r)。"""
    pairs = quad_index(r)
    table = np.zeros((r + len(pairs) + 1, r), dtype=int)
    for i in range(r):
        table[i, i] = 1
    for pos, (i, j) in enumerate(pairs):
        table[r + pos, i] += 1
        table[r + pos, j] += 1
    return table


def features(Q: np.ndarray) -> np.ndarray:
    """干净或带噪状态的 z(q)。Q: (n, r) -> (n, p)。"""
    states = np.asarray(Q, dtype=np.float64)
    pairs = quad_index(states.shape[1])
    quads = np.column_stack([states[:, i] * states[:, j] for i, j in pairs])
    ones = np.ones((states.shape[0], 1), dtype=np.float64)
    return np.hstack([states, quads, ones])


def hermite_table(Q: np.ndarray, s2: float) -> np.ndarray:
    """He_0 到 He_4。返回形状 (5, n, r)。"""
    states = np.asarray(Q, dtype=np.float64)
    variance = float(s2)
    x2 = states * states
    table = np.empty((5,) + states.shape, dtype=np.float64)
    table[0] = 1.0
    table[1] = states
    table[2] = x2 - variance
    table[3] = states * x2 - 3.0 * variance * states
    table[4] = x2 * x2 - 6.0 * variance * x2 + 3.0 * variance * variance
    return table


def hermite_monomial_sums(Q: np.ndarray, s2: float, alphas: np.ndarray) -> np.ndarray:
    """每个多重指数的 sum_k H_alpha(q_k)。alphas: (n_alpha, r)。"""
    states = np.asarray(Q, dtype=np.float64)
    powers = np.asarray(alphas, dtype=int)
    table = hermite_table(states, s2)
    acc = np.ones((powers.shape[0], states.shape[0]), dtype=np.float64)
    for coord in range(states.shape[1]):
        acc *= table[powers[:, coord], :, coord]
    return acc.sum(axis=1)


def corrected_features(Q: np.ndarray, s2: float) -> np.ndarray:
    """修正特征 hat z。只有对角二次项减去 s2。Q: (n, r) -> (n, p)。"""
    encoded = features(Q)
    encoded[:, diag_quad_rows(Q.shape[1])] -= float(s2)
    return encoded
