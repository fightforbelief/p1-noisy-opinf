"""S1：Galerkin–Burgers。离散映射是精确的二次多项式。"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from noisyopinf.features import features, quad_index
from noisyopinf.seeds import rng
from noisyopinf.systems.config import load_system


def mode_index(r: int) -> np.ndarray:
    """系数下标 i 对应的波数 m = floor(i/2)+1。"""
    return np.arange(r, dtype=np.float64) // 2.0 + 1.0


def basis(m: int, ng: int) -> tuple[np.ndarray, np.ndarray]:
    """实 Fourier 基 Φ、Φ_x，形状 (Ng, r)。"""
    x = np.arange(ng, dtype=np.float64) / float(ng)
    r = 2 * int(m)
    phi = np.zeros((ng, r), dtype=np.float64)
    phi_x = np.zeros((ng, r), dtype=np.float64)
    for wave in range(1, int(m) + 1):
        angle = 2.0 * np.pi * wave * x
        scale = np.sqrt(2.0)
        cosine, sine = np.cos(angle), np.sin(angle)
        col = 2 * (wave - 1)
        phi[:, col] = scale * cosine
        phi[:, col + 1] = scale * sine
        phi_x[:, col] = -2.0 * np.pi * wave * scale * sine
        phi_x[:, col + 1] = 2.0 * np.pi * wave * scale * cosine
    return phi, phi_x


def linear_diag(nu: float, r: int) -> np.ndarray:
    """L 的对角元 -ν (2π m_i)^2。"""
    return -float(nu) * (2.0 * np.pi * mode_index(r)) ** 2


def nonlinear(q: np.ndarray, phi: np.ndarray, phi_x: np.ndarray) -> np.ndarray:
    """N(q) = (1/Ng) Φ^T [-(Φq) ⊙ (Φ_x q)]。"""
    state = np.asarray(q, dtype=np.float64)
    single = state.ndim == 1
    if single:
        state = state[None, :]
    flux = -((state @ phi.T) * (state @ phi_x.T))
    projected = flux @ phi / float(phi.shape[0])
    return projected[0] if single else projected


def step_map(q: np.ndarray, phi: np.ndarray, phi_x: np.ndarray, diag: np.ndarray, h: float) -> np.ndarray:
    """显式 Euler：F(q) = q + h f(q)。"""
    state = np.asarray(q, dtype=np.float64)
    return state + float(h) * (state * diag + nonlinear(state, phi, phi_x))


def field(q: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """网格值 u = Φ q。"""
    state = np.asarray(q, dtype=np.float64)
    if state.ndim == 1:
        return phi @ state
    return state @ phi.T


def initial_state(index: int, split_id: int, m: int) -> np.ndarray:
    """每对模态 ~ N(0, (0.5/m)^2)。"""
    gen = rng(1, split_id, index)
    q0 = np.zeros(2 * int(m), dtype=np.float64)
    for wave in range(1, int(m) + 1):
        scale = 0.5 / float(wave)
        col = 2 * (wave - 1)
        q0[col] = gen.normal(0.0, scale)
        q0[col + 1] = gen.normal(0.0, scale)
    return q0


def rollout(index: int, split_id: int) -> tuple[np.ndarray, float]:
    """一条轨迹，形状 (K+1, r)，以及网格上的 max|u|。"""
    cfg = load_system("s1")
    m, ng, k_steps = int(cfg["M"]), int(cfg["Ng"]), int(cfg["K"])
    phi, phi_x = basis(m, ng)
    diag = linear_diag(float(cfg["nu"]), 2 * m)
    h = float(cfg["h"])
    q = initial_state(index, split_id, m)
    traj = np.empty((k_steps + 1, 2 * m), dtype=np.float64)
    traj[0] = q
    peak = float(np.max(np.abs(field(q, phi))))
    for step in range(k_steps):
        q = step_map(q, phi, phi_x, diag, h)
        traj[step + 1] = q
        peak = max(peak, float(np.max(np.abs(field(q, phi)))))
    return traj, peak


def true_operator() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在 rng(1, 9) 的 5p 个标准正态点上用 OLS 拟合 F。"""
    cfg = load_system("s1")
    m, ng = int(cfg["M"]), int(cfg["Ng"])
    r = 2 * m
    phi, phi_x = basis(m, ng)
    diag = linear_diag(float(cfg["nu"]), r)
    points = rng(1, 9).standard_normal(size=(5 * (r + len(quad_index(r)) + 1), r))
    images = step_map(points, phi, phi_x, diag, float(cfg["h"]))
    design = features(points)
    operator, _, _, _ = np.linalg.lstsq(design, images, rcond=None)
    residual = design @ operator - images
    constant = np.asarray(operator[-1], dtype=np.float64)
    residual_max = float(np.max(np.abs(residual)))
    constant_norm = float(np.linalg.norm(constant))
    if residual_max >= 1e-11 or constant_norm >= 1e-12:
        raise AssertionError(f"S1 residual {residual_max}, constant norm {constant_norm}")
    return operator, residual, constant
