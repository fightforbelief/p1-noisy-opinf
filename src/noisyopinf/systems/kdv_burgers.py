"""S3：KdV–Burgers，u_t + u u_x = ν u_xx - δ u_xxx。"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from noisyopinf.systems.burgers import _resolved
from noisyopinf.systems.config import load_system
from noisyopinf.systems.spectral import sine_initial, solve, wavenumbers


def linear_symbol(n: int, nu: float, delta: float) -> np.ndarray:
    """L(k) = -ν k^2 + i δ k^3。"""
    k = wavenumbers(n)
    return -float(nu) * k**2 + 1j * float(delta) * k**3


def simulate(
    index: int,
    split_id: int,
    *,
    n=None,
    dt=None,
    nu=None,
    delta=None,
    amplitude=None,
) -> np.ndarray:
    """返回形状 (K+1, N) 的轨迹。"""
    cfg = load_system("s3")
    n_use, dt_use, nu_use, amp, n_sub = _resolved(cfg, n, dt, nu, amplitude)
    delta_use = float(cfg["delta"] if delta is None else delta)
    u0 = sine_initial(n_use, 3, split_id, index, amp)
    symbol = linear_symbol(n_use, nu_use, delta_use)
    return solve(u0, symbol, dt_use, n_sub, int(cfg["K"]))
