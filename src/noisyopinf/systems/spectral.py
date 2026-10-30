"""周期一维 ETDRK4。系数与单步公式以 03 为准。"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from noisyopinf.seeds import rng

MC = 32


def wavenumbers(n: int) -> np.ndarray:
    """物理波数 k = 2π f。"""
    return 2.0 * np.pi * np.fft.rfftfreq(n, d=1.0 / n)


def dealias_mask(n: int) -> np.ndarray:
    """2/3 规则：|整数波数| < N/3 的模态保留。"""
    freq = np.fft.rfftfreq(n, d=1.0 / n)
    return np.abs(freq) < (n / 3.0)


def burgers_nhat(v: np.ndarray, k: np.ndarray, mask: np.ndarray, n: int) -> np.ndarray:
    """Nhat(v) = -0.5 i k mask rfft(irfft(v)^2)。"""
    field = np.fft.irfft(v, n=n)
    return -0.5j * k * mask * np.fft.rfft(field * field)


def sine_initial(n: int, system_id: int, split_id: int, index: int, amplitude: float) -> np.ndarray:
    """u0 = amplitude * sum_{m=1}^{3} (g_m/m) sin(2π m x + φ_m)。"""
    x = np.arange(n, dtype=np.float64) / float(n)
    gen = rng(system_id, split_id, index)
    gains = gen.normal(size=3)
    phases = gen.uniform(0.0, 2.0 * np.pi, size=3)
    field = np.zeros(n, dtype=np.float64)
    for wave in range(1, 4):
        angle = 2.0 * np.pi * wave * x + phases[wave - 1]
        field += (gains[wave - 1] / wave) * np.sin(angle)
    return float(amplitude) * field


def _coefficients(lk: np.ndarray, dt: float) -> tuple[np.ndarray, ...]:
    """Kassam–Trefethen 系数。Lk 为实数时 Q、f1、f2、f3 取实部。"""
    lh = dt * lk
    exp_full = np.exp(lh)
    exp_half = np.exp(lh / 2.0)
    angles = np.arange(1, MC + 1) - 0.5
    rr = np.exp(1j * np.pi * angles / MC)
    lr = lh[:, None] + rr[None, :]
    q = dt * np.mean((np.exp(lr / 2.0) - 1.0) / lr, axis=1)
    f1 = dt * np.mean((-4.0 - lr + np.exp(lr) * (4.0 - 3.0 * lr + lr**2)) / lr**3, axis=1)
    f2 = dt * np.mean((2.0 + lr + np.exp(lr) * (-2.0 + lr)) / lr**3, axis=1)
    f3 = dt * np.mean((-4.0 - 3.0 * lr - lr**2 + np.exp(lr) * (4.0 - lr)) / lr**3, axis=1)
    if np.isrealobj(lk) or float(np.max(np.abs(np.imag(np.asarray(lk))))) == 0.0:
        q, f1, f2, f3 = q.real, f1.real, f2.real, f3.real
    return exp_full, exp_half, q, f1, f2, f3


def _step(v: np.ndarray, coeff: tuple[np.ndarray, ...], nhat) -> np.ndarray:
    """ETDRK4 单步。"""
    exp_full, exp_half, q, f1, f2, f3 = coeff
    nv = nhat(v)
    a = exp_half * v + q * nv
    na = nhat(a)
    b = exp_half * v + q * na
    nb = nhat(b)
    c = exp_half * a + q * (2.0 * nb - nv)
    nc = nhat(c)
    return exp_full * v + nv * f1 + 2.0 * (na + nb) * f2 + nc * f3


def _nhat_factory(k: np.ndarray, mask: np.ndarray, n: int, nonlinear: bool):
    """默认 Burgers 通量；线性测试可以关掉非线性。"""
    if not nonlinear:
        return lambda v: np.zeros_like(v, dtype=np.complex128)

    def nhat(v: np.ndarray) -> np.ndarray:
        return burgers_nhat(v, k, mask, n)

    return nhat


def solve(u0, lk, dt, n_sub, n_obs, *, nonlinear: bool = True) -> np.ndarray:
    """积分 n_obs 个观测步。返回形状 (n_obs+1, N)，包含初值。"""
    state0 = np.asarray(u0, dtype=np.float64)
    n = int(state0.shape[0])
    symbol = np.asarray(lk)
    coeff = _coefficients(symbol, float(dt))
    nhat = _nhat_factory(wavenumbers(n), dealias_mask(n), n, nonlinear)
    spectrum = np.fft.rfft(state0)
    frames = np.empty((int(n_obs) + 1, n), dtype=np.float64)
    frames[0] = state0
    for obs in range(int(n_obs)):
        for _ in range(int(n_sub)):
            spectrum = _step(spectrum, coeff, nhat)
        frames[obs + 1] = np.fft.irfft(spectrum, n=n)
    return frames
