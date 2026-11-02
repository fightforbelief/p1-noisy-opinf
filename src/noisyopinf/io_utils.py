"""实验结果的 CSV、JSON 和算子归档。"""

from __future__ import annotations

import csv
import json
import os
from io import StringIO
from pathlib import Path

import numpy as np

RUN_COLUMNS = (
    "exp", "system", "rho", "sigma2_true", "sigma2_used", "sigma2_source", "r",
    "n_train", "n_pairs", "basis", "estimator", "lam", "replicate", "seed_keys",
    "err_O_rel", "err_A_rel", "err_H_rel", "err_c_abs", "err_c_scaled",
    "specrad_hat", "specrad_ref", "specrad_err", "ep_violation", "ep_violation_ref",
    "rollout_med", "rollout_med_h10", "rollout_med_h50", "rollout_mean_finite",
    "diverged_frac", "proj_floor", "gram_min_eig", "gram_cond", "n_clipped",
    "fit_seconds", "status", "error",
)

_TEXT = {"exp", "system", "sigma2_source", "basis", "estimator", "seed_keys", "status", "error"}


def write_runs(path, rows: list[dict], overwrite: bool = False) -> None:
    """按固定列写 runs.csv。不适用的值留空，不写成 0。"""
    _atomic_text(path, _runs_text(rows), overwrite)


def read_runs(path) -> list[dict]:
    """读回 runs.csv。空单元格是 None，inf 保持 inf。"""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [_parse_row(row) for row in reader]


def write_meta(path, meta: dict, overwrite: bool = False) -> None:
    """写 meta.json。"""
    payload = json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_text(path, payload, overwrite)


def read_meta(path) -> dict:
    """读 meta.json。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_operators(path, arrays: dict[str, np.ndarray], overwrite: bool = False) -> None:
    """把算子写入 npz。键名排序，先写临时文件再替换。"""
    destination = Path(path)
    _reject_existing(destination, overwrite)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **{key: arrays[key] for key in sorted(arrays)})
    os.replace(temporary, destination)


def read_operators(path) -> dict[str, np.ndarray]:
    """读回 npz 里的全部数组。"""
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def _runs_text(rows: list[dict]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=RUN_COLUMNS, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: _cell(row.get(column), column) for column in RUN_COLUMNS})
    return buffer.getvalue()


def _cell(value, column: str) -> str:
    del column
    if value is None or (isinstance(value, str) and value == ""):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return str(int(value))
    number = float(value)
    if np.isnan(number):
        return ""
    if np.isposinf(number):
        return "inf"
    if np.isneginf(number):
        return "-inf"
    return repr(number)


def _parse_row(row: dict) -> dict:
    parsed = {}
    for column in RUN_COLUMNS:
        text = row.get(column, "")
        parsed[column] = _parse_cell(text, column)
    return parsed


def _parse_cell(text: str, column: str):
    if text == "":
        return None
    if column in _TEXT:
        return text
    return float(text)


def _atomic_text(path, text: str, overwrite: bool) -> None:
    destination = Path(path)
    _reject_existing(destination, overwrite)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, destination)


def _reject_existing(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)


def runtime_facts() -> dict:
    """机器、Python 和依赖版本。这里不放任何数值结果。"""
    import importlib.metadata as metadata
    import platform
    import sys

    versions = {}
    for name in ("numpy", "scipy", "pandas", "matplotlib", "pyyaml", "pytest", "joblib", "tqdm"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = ""
    return {
        "python": sys.version.split()[0],
        "machine": platform.node(),
        "cpu_count": os.cpu_count(),
        "dependencies": versions,
    }
