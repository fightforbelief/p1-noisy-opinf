"""运行 E5 第 1 部分。smoke 只跑 r=6；run 跑配置里的全部秩。不覆盖已有目录。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noisyopinf.data import parameter_hash
from noisyopinf.e5 import (
    array_sha256,
    clean_basis,
    data_fingerprint,
    deviation_record,
    evaluate_rank,
    file_sha256,
    load_clean_splits,
    load_config,
    residual_cross,
    scaled_matrices,
    spectrum_record,
    write_json,
    write_table,
)
from noisyopinf.features import features
from noisyopinf.pod import pod_basis


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    config = load_config(ROOT / "configs" / "e5_part1.yaml")
    if command == "smoke":
        execute(config, ROOT / "results" / "e5_part1_smoke", (6,))
    elif command == "run":
        execute(config, ROOT / config["output_dir"], tuple(config["r"]))
    elif command == "audit":
        audit(ROOT / config["output_dir"])
    elif command == "rediagnose":
        rediagnose(ROOT / config["output_dir"])
    elif command == "figure":
        draw(ROOT / config["output_dir"] / "metrics.csv", ROOT / "figures" / "fig_e5_iv_target.png")
    else:
        raise SystemExit("use smoke, run, audit, or figure")


def execute(config: dict, output: Path, ranks: tuple[int, ...]) -> None:
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"{output} exists and is not empty; refusing to overwrite")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    splits = load_clean_splits(int(config["n_train"]), int(config["n_test"]))
    train_u = splits["train"].U
    test_u = splits["test"].U
    basis, values = clean_basis(train_u, int(config["r_max"]))
    rows, operators, spectra, crosses = [], {}, {}, {}
    for rank in ranks:
        payload = evaluate_rank(rank, basis, train_u, test_u, float(config["relative_reference_floor"]))
        payload["row"].update(prefix_gap(train_u, basis, rank))
        rows.append(payload["row"])
        for method, operator in payload["operators"].items():
            operators[f"r{rank:02d}_{method}"] = np.asarray(operator, dtype=np.float64)
        spectra.update(spectra_for_rank(rank, payload["train_q"]))
        crosses.update(crosses_for_rank(rank, payload["train_q"], payload["operators"]["ols"]))
        print("RANK", rank, "closure", payload["row"]["closure_rel"], flush=True)
    _store(output, config, ranks, rows, operators, spectra, crosses, basis, values, train_u, test_u, started)
    print("E5_PART1_DONE", output, flush=True)


def spectra_for_rank(rank: int, train_q: list[np.ndarray]) -> dict:
    saved = {}
    for method in ("ols", "iv", "iv2"):
        design, instruments, kappa = scaled_matrices(train_q, method)
        saved[f"r{rank:02d}_{method}_kappa"] = np.array(kappa)
        for name, matrix in (("design", design), ("instrument", instruments), ("cross", instruments.T @ design)):
            record = spectrum_record(matrix)
            saved[f"r{rank:02d}_{method}_{name}_sv"] = record["singular_values"]
            saved[f"r{rank:02d}_{method}_{name}_meta"] = np.array(
                [record["n_rows"], record["n_cols"], record["numerical_rank"], record["cond"], record["svd_cutoff"]],
                dtype=np.float64,
            )
    return saved


def crosses_for_rank(rank: int, train_q: list[np.ndarray], operator: np.ndarray) -> dict:
    saved = {}
    for method in ("ols", "iv", "iv2"):
        saved[f"r{rank:02d}_{method}_residual_cross"] = residual_cross(train_q, operator, method)
    return saved


def prefix_gap(train_u, basis: np.ndarray, rank: int) -> dict:
    fresh, _values = pod_basis(train_u, rank)
    prefix = basis[:, :rank]
    aligned = np.array(fresh, copy=True)
    for column in range(rank):
        if float(np.dot(aligned[:, column], prefix[:, column])) < 0.0:
            aligned[:, column] *= -1.0
    return {
        "diagnostic_prefix_gap_raw": float(np.max(np.abs(fresh - prefix))),
        "diagnostic_prefix_gap_sign_aligned": float(np.max(np.abs(aligned - prefix))),
    }


def _store(output, config, ranks, rows, operators, spectra, crosses, basis, values, train_u, test_u, started) -> None:
    write_table(output / "metrics.csv", rows)
    np.savez_compressed(output / "operators.npz", **operators)
    np.savez_compressed(output / "spectra.npz", **spectra)
    np.savez_compressed(output / "residual_cross.npz", **crosses)
    np.savez_compressed(output / "basis.npz", basis=basis, singular_values=values)
    summary = _diagnostic_rows(spectra, crosses)
    write_table(output / "diagnostics.csv", summary)
    meta = {
        "experiment": "E5_PART1",
        "ranks": list(ranks),
        "parameter_hash": parameter_hash("s2"),
        "train_fingerprint": data_fingerprint(train_u),
        "test_fingerprint": data_fingerprint(test_u),
        "basis_fingerprint": array_sha256(basis),
        "elapsed_seconds": time.perf_counter() - started,
        "n_train": len(train_u),
        "n_test": len(test_u),
        "rho": 0,
        "correct_intercept": False,
        "sigma2": 0,
    }
    write_json(output / "meta.json", meta)
    write_json(output / "failures.json", [])
    manifest = []
    for path in sorted(output.iterdir()):
        if path.name == "output_manifest.json":
            continue
        manifest.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    write_json(output / "output_manifest.json", manifest)


def _diagnostic_rows(spectra: dict, crosses: dict) -> list[dict]:
    rows = []
    keys = sorted(key for key in spectra if key.endswith("_design_meta"))
    for key in keys:
        rank = int(key[1:3])
        method = key.split("_")[1]
        for name in ("design", "instrument", "cross"):
            meta = spectra[f"r{rank:02d}_{method}_{name}_meta"]
            cross = crosses[f"r{rank:02d}_{method}_residual_cross"]
            rows.append({
                "r": rank,
                "method": method,
                "matrix": name,
                "n_rows": meta[0],
                "n_cols": meta[1],
                "numerical_rank": meta[2],
                "cond": meta[3],
                "svd_cutoff": meta[4],
                "residual_cross_frob": float(np.linalg.norm(cross)),
                "solver_uses_pinv": 0,
                "numerical_rank_definition": "count of singular values above eps * max(m, n) * smax",
            })
    return rows


def rediagnose(output: Path) -> None:
    """用已保存算子重写谱和残差交叉矩。不重新拟合。"""
    output = Path(output).resolve()
    meta = json.loads((output / "meta.json").read_text(encoding="utf-8"))
    splits = load_clean_splits(int(meta["n_train"]), int(meta["n_test"]))
    with np.load(output / "basis.npz") as loaded:
        basis = np.asarray(loaded["basis"])
    with np.load(output / "operators.npz") as operators:
        names = [name for name in operators.files if name.endswith("_ols")]
        saved_ops = {name: np.asarray(operators[name]) for name in names}
    spectra, crosses = {}, {}
    for name, operator in saved_ops.items():
        if not name.endswith("_ols"):
            continue
        rank = int(name[1:3])
        train_q = [item @ basis[:, :rank] for item in splits["train"].U]
        spectra.update(spectra_for_rank(rank, train_q))
        crosses.update(crosses_for_rank(rank, train_q, operator))
    np.savez_compressed(output / "spectra.npz", **spectra)
    np.savez_compressed(output / "residual_cross.npz", **crosses)
    write_table(output / "diagnostics.csv", _diagnostic_rows(spectra, crosses))
    manifest = []
    for path in sorted(output.iterdir()):
        if path.name == "output_manifest.json":
            continue
        manifest.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    write_json(output / "output_manifest.json", manifest)
    print("REDIAGNOSE_DONE", output)


def draw(metrics_path: Path, figure_path: Path) -> None:
    rows = _read_csv(metrics_path)
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    series = (
        ("iv_dev_A_rel", "o", "C0", "IV A"),
        ("iv_dev_H_rel", "o", "C1", "IV H"),
        ("iv2_dev_A_rel", "s", "C0", "IV2 A"),
        ("iv2_dev_H_rel", "s", "C1", "IV2 H"),
    )
    closures = [float(row["closure_rel"]) for row in rows]
    for column, marker, color, label in series:
        values = [float(row[column]) for row in rows]
        linestyle = "-" if column.endswith("A_rel") else "--"
        axis.plot(closures, values, marker=marker, color=color, linestyle=linestyle, label=label)
    for row, closure in zip(rows, closures, strict=True):
        axis.annotate(f"r={int(float(row['r']))}", (closure, float(row["iv_dev_A_rel"])), textcoords="offset points", xytext=(4, 4), fontsize=8)
    axis.set_xlabel("closure_rel = ||Y - Z O_clean||_F / ||Y||_F")
    axis.set_ylabel("block deviation ||.||_F / ||O_clean block||_F")
    axis.set_title("IV and IV2 targets versus clean OLS; A solid, H dashed")
    axis.legend()
    figure.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)


def audit(output: Path) -> None:
    """用已保存算子和重新加载的训练轨迹复算 closure 与偏离，不调用 fit。"""
    meta = json.loads((output / "meta.json").read_text(encoding="utf-8"))
    splits = load_clean_splits(int(meta["n_train"]), int(meta["n_test"]))
    with np.load(output / "basis.npz") as loaded:
        basis = np.asarray(loaded["basis"])
    if data_fingerprint(splits["train"].U) != meta["train_fingerprint"]:
        raise SystemExit("train fingerprint changed")
    operators = np.load(output / "operators.npz")
    rows = _read_csv(output / "metrics.csv")
    problems = []
    for row in rows:
        rank = int(float(row["r"]))
        prefix = basis[:, :rank]
        train_q = [item @ prefix for item in splits["train"].U]
        reference = operators[f"r{rank:02d}_ols"]
        closure = _closure(train_q, reference)
        if abs(closure - float(row["closure_rel"])) > 1e-10 * max(1.0, abs(closure)):
            problems.append(("closure", rank, closure, row["closure_rel"]))
        for method in ("iv", "iv2"):
            gap = deviation_record(operators[f"r{rank:02d}_{method}"], reference, 1e-12)
            saved = float(row[f"{method}_dev_O_rel"])
            if abs(gap["dev_O_rel"] - saved) > 1e-10 * max(1.0, abs(saved)):
                problems.append((method, rank, gap["dev_O_rel"], saved))
    manifest = json.loads((output / "output_manifest.json").read_text(encoding="utf-8"))
    for item in manifest:
        path = ROOT / item["path"]
        if item["path"].endswith("output_manifest.json"):
            continue
        digest = file_sha256(path)
        if digest != item["sha256"] or path.stat().st_size != item["bytes"]:
            problems.append(("hash", item["path"]))
    print("AUDIT_PROBLEMS", len(problems))
    for item in problems:
        print(item)
    if problems:
        raise SystemExit(1)


def _closure(trajs: list[np.ndarray], operator: np.ndarray) -> float:
    inputs = np.vstack([traj[:-1] for traj in trajs])
    outputs = np.vstack([traj[1:] for traj in trajs])
    residual = outputs - features(inputs) @ operator
    return float(np.linalg.norm(residual) / np.linalg.norm(outputs))


def _read_csv(path: Path) -> list[dict]:
    import csv
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
