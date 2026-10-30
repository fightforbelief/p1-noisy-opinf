"""S2：粘性 Burgers，u_t + u u_x = ν u_xx。"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from noisyopinf.systems.config import load_system
from noisyopinf.systems.spectral import sine_initial, solve, wavenumbers


def linear_symbol(n: int, nu: float) -> np.ndarray:
    """L(k) = -ν k^2，实数。"""
    return -float(nu) * wavenumbers(n) ** 2


def _resolved(cfg: dict, n, dt, nu, amplitude) -> tuple[int, float, float, float, int]:
    """用显式参数覆盖 yaml。观测间隔保持 dt_obs。"""
    n_use = int(cfg["N"] if n is None else n)
    dt_use = float(cfg["dt"] if dt is None else dt)
    nu_use = float(cfg["nu"] if nu is None else nu)
    amp = float(cfg["amplitude"] if amplitude is None else amplitude)
    n_sub = int(round(float(cfg["dt_obs"]) / dt_use))
    if n_sub < 1 or abs(n_sub * dt_use - float(cfg["dt_obs"])) > 1e-9:
        raise ValueError("dt does not divide dt_obs")
    return n_use, dt_use, nu_use, amp, n_sub


def simulate(index: int, split_id: int, *, n=None, dt=None, nu=None, amplitude=None) -> np.ndarray:
    """返回形状 (K+1, N) 的轨迹。"""
    cfg = load_system("s2")
    n_use, dt_use, nu_use, amp, n_sub = _resolved(cfg, n, dt, nu, amplitude)
    u0 = sine_initial(n_use, 2, split_id, index, amp)
    return solve(u0, linear_symbol(n_use, nu_use), dt_use, n_sub, int(cfg["K"]))
