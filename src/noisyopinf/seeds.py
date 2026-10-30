"""统一随机数。所有随机数都从 MASTER_SEED 派生。"""

from __future__ import annotations

import numpy as np

MASTER_SEED = 20261029


def rng(*keys: int) -> np.random.Generator:
    """返回由主种子和整数键确定的生成器。"""
    return np.random.default_rng([MASTER_SEED, *keys])
