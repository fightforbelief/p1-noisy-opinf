"""读取 configs/systems.yaml。参数不写死在求解器里。"""

from __future__ import annotations

from pathlib import Path

import yaml


def project_root() -> Path:
    """仓库根目录。"""
    return Path(__file__).resolve().parents[3]


def load_system(name: str) -> dict:
    """返回一个系统的参数字典。"""
    path = project_root() / "configs" / "systems.yaml"
    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if name not in document:
        raise KeyError(name)
    return dict(document[name])
