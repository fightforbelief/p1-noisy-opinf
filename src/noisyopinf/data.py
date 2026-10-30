"""轨迹生成、缓存和 train/val/test 划分。"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from joblib import Parallel, delayed

from noisyopinf.systems import burgers, galerkin_burgers, kdv_burgers
from noisyopinf.systems.config import load_system, project_root

SYSTEM_ID = {"s1": 1, "s2": 2, "s3": 3}
SPLIT_ID = {"train": 1, "val": 2, "test": 3}


@dataclass
class Split:
    """一个划分。S1 使用 Q，S2/S3 使用 U。"""

    U: list[np.ndarray] | None
    Q: list[np.ndarray] | None
    indices: list[int]
    skipped: list[int] = field(default_factory=list)


def data_dir() -> Path:
    """缓存目录。测试可以用 NOISYOPINF_DATA 指到临时目录。"""
    override = os.environ.get("NOISYOPINF_DATA")
    if override:
        return Path(override)
    return project_root() / "data"


def parameter_hash(system: str) -> str:
    """参数哈希。yaml 变化后旧缓存作废。"""
    payload = json.dumps(load_system(system), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(system: str, split: str, index: int) -> Path:
    return data_dir() / f"{system}_{split}_{index}.npz"


def _load_cached(path: Path, digest: str) -> np.ndarray | None:
    if not path.exists():
        return None
    with np.load(path) as loaded:
        if str(loaded["param_hash"].item()) != digest:
            return None
        return np.array(loaded["state"], dtype=np.float64)


def _store(path: Path, state: np.ndarray, digest: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, state=np.asarray(state, dtype=np.float64), param_hash=np.array(digest))


def _simulate(system: str, split: str, index: int) -> tuple[np.ndarray, bool]:
    split_id = SPLIT_ID[split]
    if system == "s1":
        state, peak = galerkin_burgers.rollout(index, split_id)
        return state, bool(np.isfinite(state).all() and peak < 5.0)
    if system == "s2":
        state = burgers.simulate(index, split_id)
    elif system == "s3":
        state = kdv_burgers.simulate(index, split_id)
    else:
        raise KeyError(system)
    return state, bool(np.isfinite(state).all())


def generate_one(system: str, split: str, index: int) -> np.ndarray | None:
    """生成或读取一条轨迹。S1 超界时返回 None，并让调用方记录 index。"""
    if system not in SYSTEM_ID or split not in SPLIT_ID:
        raise KeyError((system, split))
    digest = parameter_hash(system)
    path = _cache_path(system, split, index)
    cached = _load_cached(path, digest)
    if cached is not None:
        return cached
    state, ok = _simulate(system, split, index)
    if not ok:
        return None
    _store(path, state, digest)
    return state


def _empty(system: str) -> Split:
    if system == "s1":
        return Split(U=None, Q=[], indices=[], skipped=[])
    return Split(U=[], Q=None, indices=[], skipped=[])


def _collect(system: str, split: str, count: int) -> Split:
    """按 index 0,1,2,... 取前 count 条合格轨迹。"""
    kept: list[tuple[int, np.ndarray]] = []
    skipped: list[int] = []
    cursor = 0
    while len(kept) < count:
        batch = list(range(cursor, cursor + count - len(kept)))
        cursor = batch[-1] + 1
        produced = Parallel(n_jobs=-1, backend="loky")(
            delayed(generate_one)(system, split, index) for index in batch
        )
        for index, state in zip(batch, produced, strict=True):
            if state is None:
                skipped.append(index)
            else:
                kept.append((index, state))
        if cursor > count + 1000:
            raise RuntimeError(f"too many rejected trajectories for {system}/{split}")
    arrays = [state for _, state in kept]
    indices = [index for index, _ in kept]
    if system == "s1":
        return Split(U=None, Q=arrays, indices=indices, skipped=skipped)
    return Split(U=arrays, Q=None, indices=indices, skipped=skipped)


def get_splits(system: str, n_train_max: int, n_val: int, n_test: int) -> dict[str, Split]:
    """生成三个划分。较小的 n_train 是较大 n_train 的前缀。"""
    counts = {"train": n_train_max, "val": n_val, "test": n_test}
    splits: dict[str, Split] = {}
    for name, count in counts.items():
        if count < 0:
            raise ValueError("split sizes must be non-negative")
        splits[name] = _empty(system) if count == 0 else _collect(system, name, count)
    return splits
