"""E5 共同窗口与求解对照。不覆盖 results/e5_part1，不重估 POD。"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noisyopinf.data import get_splits, parameter_hash
from noisyopinf.e5 import array_sha256, data_fingerprint, file_sha256, write_json, write_table
from noisyopinf.e5_target_diagnostic import (
    PRED_REL_OBVIOUS,
    PRED_REL_TINY,
    REL_FLOOR,
    SOLVER_REL_FLAG,
    TRUNCATION_REL,
    diagnose_rank,
)

RANKS = (4, 6, 8, 10, 12, 16, 20)
ESTIMATOR_HASH = "a4716e630850e93bafc0acb55197ad4e7dfef0758c8ba4c06a32ccccc0e381c4"
TRAIN_FINGERPRINT = "093f84934b9c306b32b6960aba4d78016687b26fa0f6492ae3b5762a70b55f72"
BASIS_FINGERPRINT = "0b5e542067843315a153a5b52cc9ef43bd3612e50f6ed04c0e57ed619f82e951"
PARAMETER_HASH = "219c52f59203ad7fb468cec9e59f4e47d58f632105a0056e54cef5d252ba4dfa"


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "run":
        execute()
    elif command == "audit":
        audit(Path(sys.argv[2]) if len(sys.argv) > 2 else latest_output())
    elif command == "figure":
        draw(Path(sys.argv[2]) if len(sys.argv) > 2 else latest_output(), ROOT / "figures" / "fig_e5_target_diagnostic.png")
    else:
        raise SystemExit("use run, audit, or figure")


def execute() -> None:
    output = choose_output()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train_u, basis, saved, published = load_inputs()
    decomp_rows, solver_rows, moment_rows, summary_rows, arrays, problems = [], [], [], [], {}, []
    for rank in RANKS:
        prefix = np.asarray(basis[:, :rank], dtype=np.float64)
        projected = [traj @ prefix for traj in train_u]
        payload = diagnose_rank(rank, projected, saved[rank], published[rank])
        decomp_rows.extend(payload["decomp_rows"])
        solver_rows.extend(payload["solver_rows"])
        moment_rows.extend(payload["moment_rows"])
        summary_rows.append(payload["summary"])
        arrays.update(payload["arrays"])
        problems.extend(payload["problems"])
        print("RANK", rank, "iv_window", payload["summary"]["iv_window_O_rel"], "iv2_window", payload["summary"]["iv2_window_O_rel"], "iv_solver", payload["summary"]["iv_solver_gap_rel"], flush=True)
    write_table(output / "window_decomposition.csv", decomp_rows)
    write_table(output / "solver_comparison.csv", solver_rows)
    write_table(output / "moments.csv", moment_rows)
    write_table(output / "summary.csv", summary_rows)
    operator_arrays = {key: value for key, value in arrays.items() if not key.endswith("_sv")}
    spectrum_arrays = {key: value for key, value in arrays.items() if key.endswith("_sv")}
    np.savez_compressed(output / "differences.npz", **operator_arrays)
    np.savez_compressed(output / "spectra.npz", **spectrum_arrays)
    figure_path = ROOT / "figures" / "fig_e5_target_diagnostic.png"
    draw(output, figure_path)
    meta = {
        "experiment": "E5_TARGET_DIAGNOSTIC",
        "output_dir": output.name,
        "ranks": list(RANKS),
        "post_hoc": True,
        "preregistered_test": False,
        "part2_ran": False,
        "n_train": len(train_u),
        "parameter_hash": parameter_hash("s2"),
        "train_fingerprint": data_fingerprint(train_u),
        "basis_fingerprint": array_sha256(basis),
        "part1_operators_sha256": file_sha256(ROOT / "results" / "e5_part1" / "operators.npz"),
        "estimator_sha256": file_sha256(ROOT / "src" / "noisyopinf" / "estimators.py"),
        "diagnostic_sha256": file_sha256(ROOT / "src" / "noisyopinf" / "e5_target_diagnostic.py"),
        "script_sha256": file_sha256(ROOT / "scripts" / "e5_target_diagnostic.py"),
        "test_sha256": file_sha256(ROOT / "tests" / "test_e5_target_diagnostic.py"),
        "prerun_sha256": file_sha256(ROOT / "reports" / "E5_TARGET_DIAGNOSTIC_PRERUN.md"),
        "figure_sha256": file_sha256(figure_path),
        "figure_bytes": figure_path.stat().st_size,
        "truncation_rel": TRUNCATION_REL,
        "solver_rel_flag": SOLVER_REL_FLAG,
        "pred_rel_tiny": PRED_REL_TINY,
        "pred_rel_obvious": PRED_REL_OBVIOUS,
        "relative_reference_floor": REL_FLOOR,
        "elapsed_seconds": time.perf_counter() - started,
        "problem_count": len(problems),
    }
    write_json(output / "meta.json", meta)
    write_json(output / "failures.json", problems)
    manifest = []
    for path in sorted(output.iterdir()):
        if path.name == "output_manifest.json":
            continue
        manifest.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    manifest.append({"path": "figures/fig_e5_target_diagnostic.png", "bytes": figure_path.stat().st_size, "sha256": file_sha256(figure_path)})
    write_json(output / "output_manifest.json", manifest)
    print("E5_TARGET_DIAGNOSTIC_DONE", output.name, "problems", len(problems), flush=True)
    if problems:
        raise SystemExit(1)


def audit(output: Path) -> None:
    output = Path(output).resolve()
    meta = json.loads((output / "meta.json").read_text(encoding="utf-8"))
    train_u, basis, saved, published = load_inputs()
    if data_fingerprint(train_u) != meta["train_fingerprint"]:
        raise SystemExit("train fingerprint changed")
    saved_summary = _read_csv(output / "summary.csv")
    problems = []
    for row in saved_summary:
        rank = int(float(row["r"]))
        payload = diagnose_rank(rank, [traj @ basis[:, :rank] for traj in train_u], saved[rank], published[rank])
        fresh = payload["summary"]
        for key, value in fresh.items():
            if not _same(value, row[key]):
                problems.append((rank, key, value, row[key]))
                break
        problems.extend(payload["problems"])
    manifest = json.loads((output / "output_manifest.json").read_text(encoding="utf-8"))
    for item in manifest:
        path = ROOT / item["path"]
        if path.name == "output_manifest.json":
            continue
        if file_sha256(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            problems.append(("hash", item["path"]))
    print("AUDIT_PROBLEMS", len(problems), flush=True)
    for item in problems[:20]:
        print(item)
    if problems:
        raise SystemExit(1)


def draw(output: Path, figure_path: Path) -> None:
    rows = _read_csv(output / "summary.csv")
    rows = sorted(rows, key=lambda item: int(float(item["r"])))
    ranks = [int(float(item["r"])) for item in rows]
    _use_cjk_font()
    figure = plt.figure(figsize=(12.4, 11.2))
    grid = figure.add_gridspec(4, 3, hspace=0.38, wspace=0.28)
    panels = (
        (0, "O", "整体"),
        (1, "A", "A"),
        (2, "H", "H"),
    )
    for column, block, title in panels:
        _deviation_panel(figure.add_subplot(grid[0, column]), rows, ranks, block, False, title)
        _deviation_panel(figure.add_subplot(grid[1, column]), rows, ranks, block, True, title)
    cond = figure.add_subplot(grid[2, :])
    _cond_panel(cond, rows, ranks)
    with np.load(output / "spectra.npz") as spectra:
        _spectrum_panel(figure.add_subplot(grid[3, 0:2]), spectra, "iv", "QtZ", "IV 的 Q.T Z / smax")
        _spectrum_panel(figure.add_subplot(grid[3, 2]), spectra, "iv2", "Zhat", "IV2 的 Zhat / smax")
    figure.suptitle("全窗口偏离与共同窗口偏离；求解矩阵的条件数和奇异谱")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def load_inputs():
    estimator = file_sha256(ROOT / "src" / "noisyopinf" / "estimators.py")
    if estimator != ESTIMATOR_HASH:
        raise SystemExit("estimators.py hash changed")
    if parameter_hash("s2") != PARAMETER_HASH:
        raise SystemExit("S2 parameter hash changed")
    part1 = ROOT / "results" / "e5_part1"
    meta = json.loads((part1 / "meta.json").read_text(encoding="utf-8"))
    if meta["train_fingerprint"] != TRAIN_FINGERPRINT or meta["basis_fingerprint"] != BASIS_FINGERPRINT:
        raise SystemExit("part 1 fingerprints changed")
    splits = get_splits("s2", 100, 0, 0)
    train_u = splits["train"].U
    if splits["train"].indices != list(range(100)) or splits["train"].skipped:
        raise SystemExit("training index set changed")
    if data_fingerprint(train_u) != TRAIN_FINGERPRINT:
        raise SystemExit("training cache fingerprint changed")
    with np.load(part1 / "basis.npz") as loaded:
        basis = np.array(loaded["basis"], dtype=np.float64)
    if array_sha256(basis) != BASIS_FINGERPRINT or basis.shape[1] != 20:
        raise SystemExit("saved basis does not match part 1")
    with np.load(part1 / "operators.npz") as loaded:
        saved = {
            rank: {method: np.array(loaded[f"r{rank:02d}_{method}"], dtype=np.float64) for method in ("ols", "iv", "iv2")}
            for rank in RANKS
        }
    published = {int(float(row["r"])): row for row in _read_csv(part1 / "metrics.csv")}
    return train_u, basis, saved, published


def choose_output() -> Path:
    for index in range(1, 100):
        path = ROOT / "results" / f"e5_target_diagnostic_v{index}"
        if not path.exists() or not any(path.iterdir()):
            return path
    raise SystemExit("no free diagnostic directory")


def latest_output() -> Path:
    found = sorted((ROOT / "results").glob("e5_target_diagnostic_v*"))
    if not found:
        raise SystemExit("diagnostic output is missing")
    return found[-1]


def _deviation_panel(axis, rows, ranks, block, relative, title):
    series = (
        (f"iv_full_{block}_{'rel' if relative else 'abs'}", "o", "-", "C0", "IV 全窗口"),
        (f"iv_window_{block}_{'rel' if relative else 'abs'}", "o", "--", "C0", "IV 本窗口"),
        (f"iv2_full_{block}_{'rel' if relative else 'abs'}", "s", "-", "C1", "IV2 全窗口"),
        (f"iv2_window_{block}_{'rel' if relative else 'abs'}", "s", "--", "C1", "IV2 本窗口"),
    )
    for column, marker, style, color, label in series:
        axis.plot(ranks, [float(row[column]) for row in rows], marker=marker, linestyle=style, color=color, label=label)
    if block == "O":
        key = f"iv_unified_O_{'rel' if relative else 'abs'}"
        axis.plot(ranks, [float(row[key]) for row in rows], marker="^", linestyle=":", color="C2", label="IV 统一窗口")
    axis.set_yscale("log")
    axis.set_xticks(ranks)
    axis.set_xlabel("r")
    axis.set_title(f"{title} {'相对' if relative else '绝对'}偏离")
    if block == "O" and not relative:
        axis.legend(fontsize=7)


def _cond_panel(axis, rows, ranks):
    series = (
        ("iv_cond_cross", "IV 交叉矩 W.T Z"),
        ("iv_cond_QtZ", "IV 求解矩阵 Q.T Z"),
        ("iv2_cond_cross", "IV2 交叉矩，不是二阶段"),
        ("iv2_cond_Zhat", "IV2 二阶段 Zhat"),
        ("ols_full_cond_Z", "OLS 设计矩阵 Z"),
    )
    for column, label in series:
        axis.plot(ranks, [float(row[column]) for row in rows], marker="o", label=label)
    axis.set_yscale("log")
    axis.set_xticks(ranks)
    axis.set_xlabel("r")
    axis.set_title("实际求解矩阵的条件数")
    axis.legend(fontsize=7, ncol=3)


def _spectrum_panel(axis, spectra, kind, name, title):
    for rank in RANKS:
        values = np.asarray(spectra[f"r{rank:02d}_{kind}_{name}_sv"], dtype=np.float64)
        if values.size == 0 or values[0] == 0.0:
            continue
        axis.plot(np.arange(1, values.size + 1), values / values[0], label=f"r={rank}")
    axis.axhline(TRUNCATION_REL, color="black", linewidth=0.8, linestyle="--", label="截断 1e-8")
    axis.set_yscale("log")
    axis.set_xlabel("奇异值序号")
    axis.set_title(title)
    axis.legend(fontsize=7, ncol=2)


def _use_cjk_font() -> None:
    from matplotlib import font_manager
    names = {item.name for item in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if name in names:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return


def _read_csv(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _same(left, right) -> bool:
    if str(left) == str(right):
        return True
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return False
    if np.isnan(left_value) and np.isnan(right_value):
        return True
    return abs(left_value - right_value) <= 1.0e-8 + 1.0e-8 * max(abs(left_value), abs(right_value))


if __name__ == "__main__":
    main()
