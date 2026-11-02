"""运行 E3。先导写入 results/e3_pilot，正式结果写入 results/e3。"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONUTF8", "1")

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))
os.environ["PYTHONPATH"] = str(_SRC) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")

import numpy as np
from joblib import Parallel, delayed

from noisyopinf.data import get_splits
from noisyopinf.e1 import refuse_existing
from noisyopinf.e3 import (
    BASIS_CODE,
    BOOTSTRAP,
    ESTIMATORS,
    EXPERIMENT_ID,
    PILOT,
    SIGMA_METHODS,
    SOURCE_CODE,
    array_sha256,
    evaluate_replicate,
    file_sha256,
    load_config,
    operator_key,
    parameter_digest,
    planned_runs,
    reference_key,
    row_plan,
    summary_rows,
)
from noisyopinf.io_utils import RUN_COLUMNS, read_runs, runtime_facts, write_meta
from noisyopinf.metrics import (
    bias_rel,
    block_bias_sd,
    bootstrap_interval,
    ep_violation,
    err_A_rel,
    err_H_rel,
    err_O_rel,
    err_c_abs,
    err_c_scaled,
    iqr,
    proj_floor,
    sd_rel,
    specrad_trio,
    state_rms,
)
from noisyopinf.pod import pod_basis, project
from noisyopinf.protocol import LAMBDAS, larger_lambda_index
from noisyopinf.seeds import rng
from noisyopinf.systems.config import project_root

HOUR_LIMIT = 4.0 * 3600.0
HASH_PATHS = (
    "configs/e3.yaml",
    "configs/systems.yaml",
    "src/noisyopinf/e3.py",
    "scripts/e3_compare.py",
    "tests/test_e3.py",
    "src/noisyopinf/estimators.py",
    "src/noisyopinf/pod.py",
    "src/noisyopinf/rollout.py",
    "src/noisyopinf/metrics.py",
    "src/noisyopinf/protocol.py",
    "src/noisyopinf/noise.py",
    "src/noisyopinf/features.py",
    "src/noisyopinf/data.py",
    "src/noisyopinf/seeds.py",
    "src/noisyopinf/systems/burgers.py",
    "src/noisyopinf/systems/spectral.py",
)
_CACHE: dict[str, dict] = {}


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "pilot":
        run_pilot()
    elif mode == "run":
        run_grid()
    elif mode == "figure":
        redraw_figure()
    elif mode == "audit":
        target = project_root() / "results" / ("e3_pilot" if len(sys.argv) > 2 and sys.argv[2] == "pilot" else "e3")
        problems = audit_directory(target, project_root() / "data" / "e3_s2_states.npz")
        print("AUDIT_PASS" if not problems else "AUDIT_FAIL")
        if problems:
            print("PROBLEMS", json.dumps(problems, ensure_ascii=False))
            raise SystemExit(1)
    else:
        raise SystemExit("usage: e3_compare.py pilot|run|figure|audit [pilot]")


def run_pilot() -> None:
    """rho=0.05、replicate 0 和 1、两种基。不写入 results/e3。"""
    out = project_root() / "results" / "e3_pilot"
    refuse_existing(out)
    config = load_config()
    prepared = _prepare(config)
    manifest = _manifest(config, "pilot-start")
    started = time.perf_counter()
    results = _run_replicates(prepared, PILOT["rho"], PILOT["replicates"])
    elapsed = time.perf_counter() - started
    job_time = float(np.mean([item["seconds"] for item in results]))
    workers = _workers()
    estimate = job_time * len(config["rho"]) * int(config["replicates"]) / workers
    problems = _pilot_problems(prepared, results, estimate)
    _write_bundle(out, config, results, {
        "pass": not problems,
        "problems": problems,
        "pilot_seconds": elapsed,
        "mean_job_seconds": job_time,
        "estimated_seconds": estimate,
        "workers": workers,
    }, manifest)
    problems.extend(_saved_problems(out, results))
    finished = _manifest(config, "pilot-end")
    problems.extend(_hash_drift(manifest, finished))
    (out / "runtime_end.json").write_text(json.dumps(finished, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not problems:
        problems.extend(audit_directory(out, prepared["cache"]))
    if problems:
        payload = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        payload["pass"] = False
        payload["problems"] = problems
        write_meta(out / "meta.json", payload, overwrite=True)
    _write_output_hashes(out)
    print("PILOT_PASS" if not problems else "PILOT_FAIL")
    print("ESTIMATE_SECONDS", estimate)
    print("PILOT_SECONDS", elapsed)
    if problems:
        print("PROBLEMS", json.dumps(problems, ensure_ascii=False))
        raise SystemExit(1)


def run_grid() -> None:
    """正式网格。跟踪中的 E3 源码必须与 HEAD 一致，先导估算不得超过 4 小时。"""
    _require_committed_sources()
    out = project_root() / "results" / "e3"
    refuse_existing(out)
    _require_pilot()
    config = load_config()
    prepared = _prepare(config)
    manifest = _manifest(config, "run-start")
    out.mkdir(parents=True)
    (out / "runtime_start.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    started = time.perf_counter()
    results = []
    try:
        for rho in config["rho"]:
            batch = _run_replicates(prepared, float(rho), range(int(config["replicates"])))
            results.extend(batch)
            print("E3_RHO", rho, len(batch), flush=True)
    except Exception as exc:
        _write_bundle(out, config, results, {"pass": False, "error": str(exc)}, manifest)
        raise
    payload = {
        "pass": True,
        "elapsed_seconds": time.perf_counter() - started,
        "rows": sum(len(basis["rows"]) for item in results for basis in item["bases"].values()),
    }
    _write_bundle(out, config, results, payload, manifest)
    finished = _manifest(config, "run-end")
    drift = _hash_drift(manifest, finished)
    (out / "runtime_end.json").write_text(json.dumps(finished, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if drift:
        payload["pass"] = False
        payload["problems"] = drift
        write_meta(out / "meta.json", _meta(payload, config, manifest), overwrite=True)
        raise SystemExit("runtime source changed during the run")
    problems = audit_directory(out, prepared["cache"])
    if problems:
        payload["pass"] = False
        payload["problems"] = problems
        write_meta(out / "meta.json", _meta(payload, config, manifest), overwrite=True)
        _write_output_hashes(out)
        print("AUDIT_FAIL", json.dumps(problems, ensure_ascii=False))
        raise SystemExit(1)
    _draw(out / "summary.csv", out / "sigma2_estimation.csv", project_root() / "figures")
    _write_output_hashes(out)
    print("E3_DONE", payload["elapsed_seconds"], payload["rows"])


def redraw_figure() -> None:
    root = project_root()
    _draw(root / "results" / "e3" / "summary.csv", root / "results" / "e3" / "sigma2_estimation.csv", root / "figures")


def _prepare(config: dict) -> dict:
    splits = get_splits("s2", int(config["n_train"]), int(config["n_val"]), int(config["n_test"]))
    train = [np.asarray(item, dtype=np.float64) for item in splits["train"].U]
    val = [np.asarray(item, dtype=np.float64) for item in splits["val"].U]
    test = [np.asarray(item, dtype=np.float64) for item in splits["test"].U]
    basis, _ = pod_basis(train, int(config["r"]))
    cache = project_root() / "data" / "e3_s2_states.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, train=np.stack(train), val=np.stack(val), test=np.stack(test), v_clean=basis)
    return {"cache": cache, "train": train, "val": val, "test": test, "v_clean": basis, "indices": {name: splits[name].indices for name in ("train", "val", "test")}}


def _run_replicates(prepared: dict, rho: float, replicates) -> list[dict]:
    from noisyopinf.e1 import rho_index

    tasks = [
        {"rho": float(rho), "rho_index": rho_index(float(rho)), "replicate": int(replicate), "cache": str(prepared["cache"])}
        for replicate in replicates
    ]
    output = Parallel(n_jobs=_workers(), backend="loky")(delayed(_job)(task) for task in tasks)
    return list(output)


def _job(task: dict) -> dict:
    started = time.perf_counter()
    state = _cached(task["cache"])
    result = evaluate_replicate(
        [state["train"][index] for index in range(state["train"].shape[0])],
        [state["val"][index] for index in range(state["val"].shape[0])],
        [state["test"][index] for index in range(state["test"].shape[0])],
        task["rho"], task["rho_index"], task["replicate"], state["v_clean"],
    )
    result["seconds"] = time.perf_counter() - started
    return result


def _cached(path: str) -> dict:
    if path not in _CACHE:
        with np.load(path) as loaded:
            _CACHE[path] = {name: np.array(loaded[name], dtype=np.float64) for name in ("train", "val", "test", "v_clean")}
    return _CACHE[path]


def _workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def _pilot_problems(prepared, results, estimate: float) -> list[str]:
    problems = []
    if estimate > HOUR_LIMIT:
        problems.append(f"estimated seconds {estimate} exceed 4 hours")
    rows = [row for item in results for payload in item["bases"].values() for row in payload["rows"]]
    if len(rows) != 60:
        problems.append(f"pilot rows {len(rows)}")
    if len(prepared["test"]) != 50 or len(prepared["train"]) != 30 or len(prepared["val"]) != 20:
        problems.append("split sizes differ from 30/20/50")
    for item in results:
        if abs(item["full_order_noise_rms"] - item["sigma"]) / item["sigma"] > 0.05:
            problems.append(f"full-order noise rms {item['full_order_noise_rms']} vs sigma {item['sigma']}")
        clean = item["bases"]["clean"]
        noisy = item["bases"]["noisy"]
        if clean["basis_hash"] == noisy["basis_hash"]:
            problems.append(f"rho 0.05 replicate {item['replicate']} bases unexpectedly identical")
        if clean["basis_hash"] != array_sha256(prepared["v_clean"]):
            problems.append(f"replicate {item['replicate']} clean basis is not the shared V_clean")
        if clean["reference_hash"] == noisy["reference_hash"]:
            problems.append(f"replicate {item['replicate']} shares one reference across bases")
        for method, source in row_plan():
            if source == "na":
                continue
            left = next(row for row in clean["rows"] if row["estimator"] == method and row["sigma2_source"] == source)
            right = next(row for row in noisy["rows"] if row["estimator"] == method and row["sigma2_source"] == source)
            if not _close(left["sigma2_used"], right["sigma2_used"]):
                problems.append(f"replicate {item['replicate']} {method}/{source} sigma2 differs across bases")
        for name, payload in item["bases"].items():
            problems.extend(_basis_problems(name, payload, item))
    return problems


def _basis_problems(name: str, payload: dict, item: dict) -> list[str]:
    problems = []
    if payload["orthogonality"] > 1e-10:
        problems.append(f"{name} basis orthogonality {payload['orthogonality']}")
    if len({float(row["proj_floor"]) for row in payload["rows"]}) != 1:
        problems.append(f"{name} projection floor differs across estimators")
    thresholds = {float(item["threshold"]) for item in payload["details"]}
    if thresholds != {float(payload["threshold"])}:
        problems.append(f"{name} blowup thresholds are not shared")
    if len({float(row["n_train"]) for row in payload["rows"]}) != 1:
        problems.append(f"{name} n_train differs inside a replicate")
    oracle = next(row for row in payload["rows"] if row["estimator"] == "oracle")
    if oracle["err_O_rel"] > 1e-8:
        problems.append(f"{name} oracle err {oracle['err_O_rel']}")
    sources = [row["sigma2_source"] for row in payload["rows"]]
    if sources.count("na") != 3 or sources.count("true") != 4:
        problems.append(f"{name} source counts {sources}")
    thresholds = {row["seed_keys"] for row in payload["rows"]}
    if len(thresholds) != 1:
        problems.append(f"{name} seed keys differ")
    for method in ("oracle", "ols"):
        if next(row for row in payload["rows"] if row["estimator"] == method)["status"] != "ok":
            problems.append(f"{name} replicate {item['replicate']} {method} failed")
    for row in payload["rows"]:
        if row["status"] != "ok" and not row["error"]:
            problems.append(f"{name} failed row has no error text")
    rebuilt = _reaggregate(payload["details"])
    for row in payload["rows"]:
        key = (row["estimator"], row["sigma2_source"])
        got = rebuilt[key]
        for field in ("rollout_med", "rollout_mean_finite", "diverged_frac", "rollout_med_h10", "rollout_med_h50"):
            if not _close(row[field], got[field]):
                problems.append(f"{name} {key} {field} detail mismatch")
                break
    return problems


def _reaggregate(details: list[dict]) -> dict:
    grouped: dict[tuple, list] = {}
    for item in details:
        grouped.setdefault((item["estimator"], item["sigma2_source"]), []).append(item)
    output = {}
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda item: item["traj"])
        full = np.asarray([float(item["err_full"]) for item in rows], dtype=np.float64)
        diverged = np.asarray([bool(item["diverged"]) for item in rows])
        values = np.array([np.inf if flag else value for flag, value in zip(diverged, full, strict=True)])
        finite = values[np.isfinite(values)]
        output[key] = {
            "rollout_med": float(np.median(values)),
            "rollout_mean_finite": float(np.mean(finite)) if finite.size else float("nan"),
            "diverged_frac": float(np.mean(diverged.astype(np.float64))),
            "rollout_med_h10": float(np.median(np.asarray([item["err_h10"] for item in rows], dtype=np.float64))),
            "rollout_med_h50": float(np.median(np.asarray([item["err_h50"] for item in rows], dtype=np.float64))),
        }
    return output


def _close(left: float, right: float) -> bool:
    if np.isnan(left) and np.isnan(right):
        return True
    if np.isposinf(left) and np.isposinf(right):
        return True
    return abs(float(left) - float(right)) <= 1e-12


def _require_pilot() -> None:
    path = project_root() / "results" / "e3_pilot" / "meta.json"
    pilot = json.loads(path.read_text(encoding="utf-8"))
    if not pilot.get("pass") or float(pilot.get("estimated_seconds", HOUR_LIMIT + 1)) > HOUR_LIMIT:
        raise RuntimeError(f"pilot gate is not open: pass={pilot.get('pass')} estimate={pilot.get('estimated_seconds')}")


def _require_committed_sources() -> None:
    root = project_root()
    dirty = subprocess.check_output(["git", "status", "--short", "--", *HASH_PATHS], cwd=root, text=True)
    if dirty.strip():
        raise RuntimeError("tracked E3 sources differ from HEAD:\n" + dirty)


def _write_bundle(out: Path, config: dict, results: list[dict], payload: dict, manifest: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = [row for item in results for payload_basis in item["bases"].values() for row in payload_basis["rows"]]
    rows = _sort_rows(rows)
    operators = {}
    for item in results:
        for payload_basis in item["bases"].values():
            operators.update(payload_basis["operators"])
    _write_runs(out / "runs.csv", rows)
    if rows and operators:
        from noisyopinf.e3 import summary_rows as build_summary

        active_cases = _active_cases(config, results)
        _write_table(out / "summary.csv", build_summary(active_cases, rows, operators))
        _write_operators(out / "operators.npz", operators)
    _write_bases(out / "bases.npz", results)
    _write_sigma(out / "sigma2_estimation.csv", results)
    _write_ridge(out / "ridge.csv", results)
    _write_details(out / "rollout_details.npz", results)
    _write_basis_audit(out / "basis_audit.csv", results)
    _write_failures(out / "failures.json", rows)
    manifest_path = out / "runtime_start.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_meta(out / "meta.json", _meta(payload, config, manifest), overwrite=True)
    _write_output_hashes(out)


def _active_cases(config: dict, results: list[dict]) -> list[dict]:
    del config
    from noisyopinf.e1 import rho_index

    grouped: dict[tuple, set[int]] = {}
    for item in results:
        for basis in item["bases"]:
            grouped.setdefault((float(item["rho"]), basis), set()).add(int(item["replicate"]))
    cases = []
    for (rho, basis), reps in sorted(grouped.items(), key=lambda item: (item[0][0], 0 if item[0][1] == "clean" else 1)):
        if reps != set(range(max(reps) + 1)):
            raise RuntimeError(f"replicate ids are not a prefix for rho={rho} basis={basis}: {sorted(reps)}")
        cases.append({"rho": rho, "basis": basis, "rho_index": rho_index(rho), "replicates": len(reps)})
    return cases


def _sort_rows(rows: list[dict]) -> list[dict]:
    method_order = {method: index for index, method in enumerate(ESTIMATORS)}
    source_order = {"na": 0, "true": 1, "est_trailing": 2, "est_gd": 3}
    basis_order = {"clean": 0, "noisy": 1}
    return sorted(rows, key=lambda row: (
        float(row["rho"]), basis_order[row["basis"]], int(row["replicate"]),
        method_order[row["estimator"]], source_order[row["sigma2_source"]],
    ))


def _write_runs(path: Path, rows: list[dict]) -> None:
    from noisyopinf.io_utils import write_runs

    write_runs(path, [_public_row(row) for row in rows], overwrite=True)


def _public_row(row: dict) -> dict:
    return {column: row.get(column) for column in RUN_COLUMNS}


def _write_table(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _cell(value) for key, value in row.items()})
    path.write_text(buffer.getvalue(), encoding="utf-8", newline="\n")


def _write_operators(path: Path, arrays: dict[str, np.ndarray]) -> None:
    _replace_npz(path, {key: arrays[key] for key in sorted(arrays)})


def _write_bases(path: Path, results: list[dict]) -> None:
    arrays = {}
    for item in results:
        clean = item["bases"]["clean"].get("basis")
        noisy = item["bases"]["noisy"].get("basis")
        if clean is not None:
            arrays["v_clean"] = clean
        if noisy is not None:
            arrays[f"v_rho{item['rho_index']}_noisy_r{item['replicate']:03d}"] = noisy
    if arrays:
        _replace_npz(path, arrays)


def _replace_npz(path: Path, arrays: dict) -> None:
    temporary = path.with_name(path.stem + ".writing.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _write_sigma(path: Path, results: list[dict]) -> None:
    fields = [
        "rho", "replicate", "sigma2_true", "sigma2_est_trailing", "sigma2_est_gd",
        "rel_trailing", "rel_gd", "applicable_trailing", "applicable_gd",
        "N", "K_tot", "beta", "clean_half_trailing", "clean_half_gd", "error_trailing", "error_gd",
    ]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for item in sorted(results, key=lambda value: (value["rho"], value["replicate"])):
        trailing = item["estimates"]["est_trailing"]
        gavish = item["estimates"]["est_gd"]
        writer.writerow({
            "rho": _cell(item["rho"]), "replicate": item["replicate"], "sigma2_true": _cell(item["sigma2_true"]),
            "sigma2_est_trailing": _cell(trailing["value"]), "sigma2_est_gd": _cell(gavish["value"]),
            "rel_trailing": _cell(trailing["relative"]), "rel_gd": _cell(gavish["relative"]),
            "applicable_trailing": trailing["applicable"], "applicable_gd": gavish["applicable"],
            "N": trailing["diagnostics"].get("N", ""), "K_tot": trailing["diagnostics"].get("K_tot", ""),
            "beta": _cell(trailing["diagnostics"].get("beta")),
            "clean_half_trailing": _cell(trailing["diagnostics"].get("clean_half_eigenvalue_over_sigma2")),
            "clean_half_gd": _cell(gavish["diagnostics"].get("clean_half_eigenvalue_over_sigma2")),
            "error_trailing": trailing["error"], "error_gd": gavish["error"],
        })
    path.write_text(buffer.getvalue(), encoding="utf-8", newline="\n")


def _write_ridge(path: Path, results: list[dict]) -> None:
    fields = ["rho", "basis", "replicate", "lam", "score", "chosen", "n_tied", "n_nonfinite", "seed_keys"]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for item in results:
        for basis, payload in item["bases"].items():
            for lam, score in zip(LAMBDAS, payload["ridge"]["scores"]):
                writer.writerow({
                    "rho": _cell(item["rho"]), "basis": basis, "replicate": item["replicate"],
                    "lam": _cell(lam), "score": _cell(score), "chosen": _cell(payload["ridge"]["lam"]),
                    "n_tied": payload["ridge"]["n_tied"], "n_nonfinite": payload["ridge"]["n_nonfinite"],
                    "seed_keys": f"2,4,3,{item['rho_index']},{item['replicate']},traj+100",
                })
    path.write_text(buffer.getvalue(), encoding="utf-8", newline="\n")


def _write_details(path: Path, results: list[dict]) -> None:
    details = [item for result in results for payload in result["bases"].values() for item in payload["details"]]
    details.sort(key=lambda item: (item["rho"], BASIS_CODE[item["basis"]], item["replicate"], ESTIMATORS.index(item["estimator"]), SOURCE_CODE[item["sigma2_source"]], item["traj"]))
    payload = {
        "rho": np.asarray([item["rho"] for item in details], dtype=np.float64),
        "basis": np.asarray([BASIS_CODE[item["basis"]] for item in details], dtype=np.int16),
        "estimator": np.asarray([ESTIMATORS.index(item["estimator"]) for item in details], dtype=np.int16),
        "sigma2_source": np.asarray([SOURCE_CODE[item["sigma2_source"]] for item in details], dtype=np.int16),
        "replicate": np.asarray([item["replicate"] for item in details], dtype=np.int16),
        "traj": np.asarray([item["traj"] for item in details], dtype=np.int16),
        "err_full": np.asarray([item["err_full"] for item in details], dtype=np.float64),
        "err_h10": np.asarray([item["err_h10"] for item in details], dtype=np.float64),
        "err_h50": np.asarray([item["err_h50"] for item in details], dtype=np.float64),
        "diverged": np.asarray([item["diverged"] for item in details], dtype=np.bool_),
        "first_step": np.asarray([item["first_step"] for item in details], dtype=np.int16),
        "threshold": np.asarray([item["threshold"] for item in details], dtype=np.float64),
        "proj_floor_traj": np.asarray([item["proj_floor_traj"] for item in details], dtype=np.float64),
        "proj_floor": np.asarray([item["proj_floor"] for item in details], dtype=np.float64),
    }
    _replace_npz(path, payload)
    codes = {
        "basis": BASIS_CODE,
        "estimator": {name: index for index, name in enumerate(ESTIMATORS)},
        "sigma2_source": SOURCE_CODE,
    }
    path.with_suffix(".codes.json").write_text(json.dumps(codes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_basis_audit(path: Path, results: list[dict]) -> None:
    fields = ["rho", "basis", "replicate", "threshold", "orthogonality", "basis_hash", "reference_hash", "proj_floor", "clean_q_rms"]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for item in results:
        for basis, payload in item["bases"].items():
            writer.writerow({
                "rho": _cell(item["rho"]), "basis": basis, "replicate": item["replicate"],
                "threshold": _cell(payload["threshold"]), "orthogonality": _cell(payload["orthogonality"]),
                "basis_hash": payload["basis_hash"], "reference_hash": payload["reference_hash"],
                "proj_floor": _cell(payload["proj_floor"]), "clean_q_rms": _cell(payload["clean_q_rms"]),
            })
    path.write_text(buffer.getvalue(), encoding="utf-8", newline="\n")


def _write_failures(path: Path, rows: list[dict]) -> None:
    failed = [
        {"rho": row["rho"], "basis": row["basis"], "estimator": row["estimator"], "sigma2_source": row["sigma2_source"], "replicate": row["replicate"], "error": row["error"]}
        for row in rows if row["status"] != "ok"
    ]
    path.write_text(json.dumps(failed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_output_hashes(out: Path) -> None:
    rows = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "output_manifest.json":
            rows.append({"path": str(path.relative_to(project_root())), "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    (out / "output_manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _saved_problems(out: Path, results: list[dict]) -> list[str]:
    problems = []
    from noisyopinf.io_utils import read_runs

    rows = read_runs(out / "runs.csv")
    if len(rows) != sum(len(payload["rows"]) for item in results for payload in item["bases"].values()):
        problems.append("saved run count differs")
    return problems


def _manifest(config: dict, phase: str) -> dict:
    root = project_root()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    status = subprocess.check_output(["git", "status", "--short"], cwd=root, text=True, stderr=subprocess.STDOUT)
    return {
        "phase": phase,
        "git_head": head,
        "git_status": status,
        "command": " ".join(sys.argv),
        "experiment_id": EXPERIMENT_ID,
        "seed_protocol": "noise keys (2, 4, 3, rho_index, replicate, traj); validation traj + 100; bootstrap " + BOOTSTRAP,
        "parameter_hash": parameter_digest(),
        "files": {relative: file_sha256(root / relative) for relative in HASH_PATHS},
        "config": config,
        "runtime": runtime_facts(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _hash_drift(started: dict, finished: dict) -> list[str]:
    problems = []
    for relative, digest in started["files"].items():
        if finished["files"].get(relative) != digest:
            problems.append(f"hash changed during run: {relative}")
    return problems


def _meta(payload: dict, config: dict, manifest: dict) -> dict:
    meta = runtime_facts()
    meta.update(payload)
    meta.update({
        "git_head": manifest["git_head"],
        "config": config,
        "bootstrap": BOOTSTRAP,
        "experiment_id": EXPERIMENT_ID,
        "parameter_hash": manifest["parameter_hash"],
        "threshold_rule": config.get("threshold_rule"),
        "planned_runs": planned_runs(config),
    })
    return meta


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
        return repr(value)
    return str(value)


def _draw(summary_path: Path, sigma_path: Path, folder: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    rows = _read_summary(summary_path)
    sigma_rows = _read_sigma(sigma_path)
    folder.mkdir(parents=True, exist_ok=True)
    _operator_figure(rows, folder / "fig_e3_operator_err.png")
    _rollout_figure(rows, folder / "fig_e3_rollout.png")
    _basis_figure(rows, folder / "fig_e3_basis_effect.png")
    _sigma_figure(rows, sigma_rows, folder / "fig_e3_sigma_source.png")
    print("FIGURES", folder)


def _read_summary(path: Path) -> list[dict]:
    text = {"exp", "system", "basis", "estimator", "sigma2_source", "failed_replicates", "status"}
    with path.open(encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = []
    for raw in raw_rows:
        row = {}
        for key, value in raw.items():
            if key in text or value == "":
                row[key] = value
            elif value == "nan":
                row[key] = float("nan")
            elif value == "inf":
                row[key] = float("inf")
            else:
                row[key] = float(value)
        rows.append(row)
    return rows


def _read_sigma(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = []
    for raw in raw_rows:
        row = {}
        for key, value in raw.items():
            if value in {"", "nan"}:
                row[key] = float("nan")
            elif value in {"True", "False"}:
                row[key] = value == "True"
            else:
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
        rows.append(row)
    return rows


def _styles() -> dict[str, tuple[str, str]]:
    return {
        "oracle": ("C0", "o"), "ols": ("C1", "s"), "ridge": ("C2", "^"), "mc": ("C3", "D"),
        "mc_psd": ("C4", "v"), "iv": ("C5", "P"), "iv2": ("C6", "X"),
    }


def _selected(rows, basis: str, method: str):
    source = "est_trailing" if method in {"mc", "mc_psd", "iv", "iv2"} else "na"
    return sorted(
        (row for row in rows if row["basis"] == basis and row["estimator"] == method and row["sigma2_source"] == source),
        key=lambda item: item["rho"],
    )


def _operator_figure(rows, path: Path) -> None:
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    for axis, key, title in zip(axes, ("err_A_rel_median", "err_H_rel_median"), ("err_A_rel", "err_H_rel"), strict=True):
        for method, (color, marker) in _styles().items():
            chosen = _selected(rows, "noisy", method)
            xs, ys = [], []
            for row in chosen:
                value = row[key]
                if np.isfinite(value) and value > 0.0:
                    xs.append(row["rho"])
                    ys.append(value)
                else:
                    print("OPERATOR_OMITTED", key, method, row["rho"], value)
            axis.plot(xs, ys, color=color, marker=marker, label=method)
        axis.set_yscale("log")
        axis.set_xlabel("rho (linear, includes 0)")
        axis.set_ylabel(title)
        axis.set_title("basis=noisy, sigma2=est_trailing or na")
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _rollout_figure(rows, path: Path) -> None:
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for method, (color, marker) in _styles().items():
        chosen = _selected(rows, "noisy", method)
        xs, ys = [], []
        for row in chosen:
            value = row["rollout_med_median"]
            if np.isfinite(value):
                xs.append(row["rho"])
                ys.append(value)
            else:
                axes[0].annotate("inf", (row["rho"], 0.92), xycoords=("data", "axes fraction"), color=color, fontsize=7, ha="center")
                print("ROLLOUT_INF", method, row["rho"], value)
        axes[0].plot(xs, ys, color=color, marker=marker, label=method)
        axes[1].plot([row["rho"] for row in chosen], [row["diverged_frac_median"] for row in chosen], color=color, marker=marker, label=method)
    floors = _selected(rows, "noisy", "ols")
    axes[0].plot([row["rho"] for row in floors], [row["proj_floor_median"] for row in floors], color="black", linestyle="--", label="median proj_floor")
    axes[0].set_ylabel("rollout_med")
    axes[1].set_ylabel("diverged_frac")
    axes[1].set_xlabel("rho (linear, includes 0)")
    axes[0].set_title("basis=noisy, sigma2=est_trailing or na; inf is marked")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _basis_figure(rows, path: Path) -> None:
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for method, color in (("mc", "C3"), ("iv2", "C6")):
        for basis, marker in (("clean", "o"), ("noisy", "s")):
            chosen = _selected(rows, basis, method)
            xs, ys = [], []
            for row in chosen:
                value = row["err_O_rel_median"]
                if np.isfinite(value):
                    xs.append(row["rho"])
                    ys.append(value)
                else:
                    print("BASIS_OMITTED", method, basis, row["rho"], value)
            axis.plot(xs, ys, color=color, marker=marker, label=f"{method}/{basis}")
    axis.set_xlabel("rho (linear, includes 0)")
    axis.set_ylabel("median err_O_rel vs its own O_clean")
    axis.set_title("sigma2=est_trailing; each basis uses its own reference")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _sigma_figure(rows, sigma_rows, path: Path) -> None:
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for source, marker in (("true", "o"), ("est_trailing", "s"), ("est_gd", "^")):
        chosen = sorted(
            (row for row in rows if row["basis"] == "noisy" and row["estimator"] == "mc" and row["sigma2_source"] == source),
            key=lambda item: item["rho"],
        )
        axes[0].plot([row["rho"] for row in chosen], [row["err_O_rel_median"] for row in chosen], marker=marker, label=source)
    for key, marker, label in (("rel_trailing", "s", "|rel trailing|"), ("rel_gd", "^", "|rel gavish-donoho|")):
        xs, ys = [], []
        for row in sigma_rows:
            value = abs(float(row[key]))
            if np.isfinite(value) and value > 0.0:
                xs.append(row["rho"])
                ys.append(value)
            else:
                print("SIGMA_REL_OMITTED", key, row.get("rho"), row.get("replicate"), row[key])
        axes[1].scatter(xs, ys, marker=marker, label=label, s=16)
    axes[0].set_title("mc err_O_rel, basis=noisy")
    axes[1].set_title("sigma2 relative error")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.set_xlabel("rho (linear, includes 0)")
        axis.legend(fontsize=8)
    axes[0].set_ylabel("median err_O_rel")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def audit_directory(out: Path, cache: Path) -> list[str]:
    """从归档重算算子误差、汇总、bootstrap、ridge、逐轨迹预测和 σ² 相对误差。"""
    from noisyopinf.e1 import rho_index

    problems = []
    rows = read_runs(out / "runs.csv")
    with np.load(out / "operators.npz") as loaded:
        operators = {key: np.array(loaded[key], dtype=np.float64) for key in loaded.files}
    with np.load(out / "bases.npz") as loaded:
        bases = {key: np.array(loaded[key], dtype=np.float64) for key in loaded.files}
    with np.load(cache) as loaded:
        train = np.array(loaded["train"], dtype=np.float64)
        test = np.array(loaded["test"], dtype=np.float64)
    checked_rms = set()
    basis_rows = _read_table(out / "basis_audit.csv")
    rms_of = {
        (float(item["rho"]), item["basis"], int(float(item["replicate"]))): float(item["clean_q_rms"])
        for item in basis_rows
    }
    hash_of = {
        (float(item["rho"]), item["basis"], int(float(item["replicate"]))): item
        for item in basis_rows
    }
    for row in rows:
        index = rho_index(float(row["rho"]))
        replicate = int(row["replicate"])
        hat = operators[operator_key(index, row["basis"], row["estimator"], row["sigma2_source"], replicate)]
        ref = operators[reference_key(index, row["basis"], replicate)]
        record = hash_of[(float(row["rho"]), row["basis"], replicate)]
        if array_sha256(ref) != record["reference_hash"]:
            problems.append(f"reference hash mismatch {row['basis']} r{replicate}")
        matrix = _basis_matrix(bases, index, row["basis"], replicate)
        if array_sha256(matrix) != record["basis_hash"]:
            problems.append(f"basis hash mismatch {row['basis']} r{replicate}")
        token = (float(row["rho"]), row["basis"], replicate)
        if token not in checked_rms:
            projected_train = np.vstack([project(train[item], matrix) for item in range(train.shape[0])])
            if not _near(state_rms(projected_train), float(record["clean_q_rms"])):
                problems.append(f"clean projection rms mismatch {row['basis']} r{replicate}")
            checked_rms.add(token)
        if row["status"] != "ok":
            continue
        rms = rms_of[(float(row["rho"]), row["basis"], replicate)]
        hat_spec = specrad_trio(hat, ref)
        expected = {
            "err_O_rel": err_O_rel(hat, ref),
            "err_A_rel": err_A_rel(hat, ref),
            "err_H_rel": err_H_rel(hat, ref),
            "err_c_abs": err_c_abs(hat, ref),
            "err_c_scaled": err_c_scaled(hat, ref, rms),
            "specrad_hat": hat_spec[0],
            "specrad_ref": hat_spec[1],
            "specrad_err": hat_spec[2],
            "proj_floor": float(record["proj_floor"]),
        }
        projected = np.vstack([project(test[item], matrix) for item in range(test.shape[0])])
        expected["ep_violation"] = ep_violation(hat, projected)
        expected["ep_violation_ref"] = ep_violation(ref, projected)
        floor = float(proj_floor(matrix, np.vstack(test)))
        if not _near(floor, float(record["proj_floor"])):
            problems.append(f"proj floor mismatch {row['basis']} r{replicate}")
        for field, value in expected.items():
            if not _near(row[field], value):
                problems.append(f"{field} mismatch {row['estimator']}/{row['sigma2_source']} {row['basis']} r{replicate}")
                break
    problems.extend(_audit_summary(out, rows, operators))
    problems.extend(_audit_details(out, rows))
    problems.extend(_audit_ridge(out, rows))
    problems.extend(_audit_sigma(out))
    return problems[:40]


def _basis_matrix(bases: dict, rho_index_value: int, basis: str, replicate: int) -> np.ndarray:
    if basis == "clean":
        return bases["v_clean"]
    return bases[f"v_rho{rho_index_value}_noisy_r{replicate:03d}"]


def _audit_summary(out: Path, rows: list[dict], operators: dict) -> list[str]:
    from noisyopinf.e1 import rho_index

    saved = _read_summary(out / "summary.csv")
    saved_of = {
        (float(item["rho"]), item["basis"], item["estimator"], item["sigma2_source"]): item
        for item in saved
    }
    grouped = {}
    for row in rows:
        grouped.setdefault((float(row["rho"]), row["basis"]), set()).add(int(row["replicate"]))
    cases = []
    for (rho, basis), reps in sorted(grouped.items(), key=lambda item: (item[0][0], 0 if item[0][1] == "clean" else 1)):
        cases.append({"rho": rho, "basis": basis, "rho_index": rho_index(rho), "replicates": len(reps)})
    by_row = {
        (float(row["rho"]), row["basis"], row["estimator"], row["sigma2_source"], int(row["replicate"])): row
        for row in rows
    }
    rebuilt = []
    problems = []
    for case in cases:
        for method, source in row_plan():
            chosen = [by_row[(case["rho"], case["basis"], method, source, replicate)] for replicate in range(case["replicates"])]
            key = (case["rho"], case["basis"], method, source)
            summary = saved_of[key]
            for name in ("err_O_rel", "err_A_rel", "err_H_rel", "rollout_med", "rollout_mean_finite", "diverged_frac", "rollout_med_h10", "rollout_med_h50", "proj_floor"):
                q25, median, q75 = _independent_quartiles([item[name] for item in chosen])
                for label, value in (("q25", q25), ("median", median), ("q75", q75)):
                    if not _near(summary[f"{name}_{label}"], value):
                        problems.append(f"summary {name}_{label} {method}/{source} {case['basis']} rho={case['rho']}")
                        break
            references = [operators[reference_key(case["rho_index"], case["basis"], replicate)] for replicate in range(case["replicates"])]
            stacked = [operators[operator_key(case["rho_index"], case["basis"], method, source, replicate)] for replicate in range(case["replicates"])]
            common = all(np.array_equal(references[0], item) for item in references[1:]) if len(references) > 1 else True
            if common and all(item["status"] == "ok" for item in chosen):
                if not _near(summary["bias_O_rel"], bias_rel(stacked, references[0])) or not _near(summary["sd_O_rel"], sd_rel(stacked, references[0])):
                    problems.append(f"bias/sd {method}/{source} {case['basis']} rho={case['rho']}")
                blocks = block_bias_sd(stacked, references[0])
                for field, value in blocks.items():
                    if not _near(summary[field], value):
                        problems.append(f"{field} {method}/{source} {case['basis']} rho={case['rho']}")
                        break
            elif not all(np.isnan(summary[field]) for field in ("bias_O_rel", "sd_O_rel", "bias_A_rel", "sd_A_rel", "bias_H_rel", "sd_H_rel")):
                problems.append(f"bias should be unavailable {method}/{source} {case['basis']} rho={case['rho']}")
            gaps = []
            source_gaps = []
            for replicate in range(case["replicates"]):
                current = by_row[(case["rho"], case["basis"], method, source, replicate)]["err_O_rel"]
                baseline = by_row[(case["rho"], case["basis"], "ols", "na", replicate)]["err_O_rel"]
                gaps.append(float(current) - float(baseline))
                if method in SIGMA_METHODS:
                    truth = by_row[(case["rho"], case["basis"], method, "true", replicate)]["err_O_rel"]
                    source_gaps.append(float(current) - float(truth))
            rebuilt.append((np.asarray(gaps, dtype=np.float64), np.asarray(source_gaps, dtype=np.float64) if source_gaps else None, summary))
    generated = rng(0, 77)
    for gaps, _source_gaps, summary in rebuilt:
        if not _near(summary["paired_median"], float(np.median(gaps))) or not _near(summary["paired_iqr"], iqr(gaps)):
            problems.append(f"paired gap {summary['estimator']}/{summary['sigma2_source']} {summary['basis']} rho={summary['rho']}")
        low, high = bootstrap_interval(gaps, generated, 2000)
        if not _near(summary["boot_low"], low) or not _near(summary["boot_high"], high):
            problems.append(f"bootstrap {summary['estimator']}/{summary['sigma2_source']} {summary['basis']} rho={summary['rho']}")
    generated = rng(0, 78)
    for _gaps, source_gaps, summary in rebuilt:
        if source_gaps is None:
            if not (np.isnan(summary["source_boot_low"]) and np.isnan(summary["source_boot_high"])):
                problems.append(f"source bootstrap should be empty {summary['estimator']}")
            continue
        low, high = bootstrap_interval(source_gaps, generated, 2000)
        if not _near(summary["source_paired_median"], float(np.median(source_gaps))) or not _near(summary["source_boot_low"], low) or not _near(summary["source_boot_high"], high):
            problems.append(f"source bootstrap {summary['estimator']}/{summary['sigma2_source']} {summary['basis']} rho={summary['rho']}")
    return problems


def _audit_details(out: Path, rows: list[dict]) -> list[str]:
    with np.load(out / "rollout_details.npz") as loaded:
        payload = {key: np.array(loaded[key]) for key in loaded.files}
    inverse_basis = {value: key for key, value in BASIS_CODE.items()}
    inverse_source = {value: key for key, value in SOURCE_CODE.items()}
    grouped: dict[tuple, list] = {}
    for index in range(payload["rho"].shape[0]):
        key = (
            float(payload["rho"][index]),
            inverse_basis[int(payload["basis"][index])],
            ESTIMATORS[int(payload["estimator"][index])],
            inverse_source[int(payload["sigma2_source"][index])],
            int(payload["replicate"][index]),
        )
        grouped.setdefault(key, []).append(index)
    problems = []
    for row in rows:
        key = (float(row["rho"]), row["basis"], row["estimator"], row["sigma2_source"], int(row["replicate"]))
        indexes = grouped[key]
        order = np.argsort([int(payload["traj"][item]) for item in indexes])
        indexes = [indexes[item] for item in order]
        full = np.asarray([float(payload["err_full"][item]) for item in indexes])
        diverged = np.asarray([bool(payload["diverged"][item]) for item in indexes])
        values = np.array([np.inf if flag else value for flag, value in zip(diverged, full, strict=True)])
        finite = values[np.isfinite(values)]
        expected = {
            "rollout_med": float(np.median(values)),
            "rollout_mean_finite": float(np.mean(finite)) if finite.size else float("nan"),
            "diverged_frac": float(np.mean(diverged.astype(np.float64))),
            "rollout_med_h10": float(np.median(payload["err_h10"][indexes])),
            "rollout_med_h50": float(np.median(payload["err_h50"][indexes])),
        }
        thresholds = {float(payload["threshold"][item]) for item in indexes}
        if len(thresholds) != 1:
            problems.append(f"detail thresholds differ {key}")
        if row["status"] != "ok":
            continue
        for field, value in expected.items():
            if not _near(row[field], value):
                problems.append(f"detail {field} {key}")
                break
    return problems


def _audit_ridge(out: Path, rows: list[dict]) -> list[str]:
    table = _read_table(out / "ridge.csv")
    grouped: dict[tuple, list] = {}
    for item in table:
        grouped.setdefault((float(item["rho"]), item["basis"], int(float(item["replicate"]))), []).append(item)
    problems = []
    for key, records in grouped.items():
        records = sorted(records, key=lambda item: float(item["lam"]))
        scores = [float(item["score"]) for item in records]
        chosen = float(LAMBDAS[larger_lambda_index(scores)])
        if not _near(chosen, float(records[0]["chosen"])):
            problems.append(f"ridge choice {key}")
        matched = [row for row in rows if float(row["rho"]) == key[0] and row["basis"] == key[1] and int(row["replicate"]) == key[2] and row["estimator"] == "ridge"]
        if len(matched) != 1 or not _near(float(matched[0]["lam"]), chosen):
            problems.append(f"ridge run lambda {key}")
    return problems


def _audit_sigma(out: Path) -> list[str]:
    problems = []
    for row in _read_sigma(out / "sigma2_estimation.csv"):
        true = float(row["sigma2_true"])
        for field, rel in (("sigma2_est_trailing", "rel_trailing"), ("sigma2_est_gd", "rel_gd")):
            value = float(row[field])
            if not np.isfinite(value):
                expected = float("nan")
            elif true == 0.0:
                expected = 0.0 if value == 0.0 else float("nan")
            else:
                expected = (value - true) / true
            if not _near(float(row[rel]), expected):
                problems.append(f"sigma relative {rel} rho={row['rho']} r={row['replicate']}")
    return problems


def _independent_quartiles(values) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0 or np.isnan(data).any():
        return float("nan"), float("nan"), float("nan")
    ordered = np.sort(data)
    return _independent_at(ordered, 0.25), float(np.median(ordered)), _independent_at(ordered, 0.75)


def _independent_at(ordered: np.ndarray, quantile: float) -> float:
    position = float(quantile) * (ordered.size - 1)
    low = int(np.floor(position))
    high = int(np.ceil(position))
    if low == high or np.isinf(ordered[low]) or np.isinf(ordered[high]):
        return float(ordered[low] if low == high else np.inf)
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def _read_table(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _near(left, right, tol: float = 1e-8) -> bool:
    left = float(left)
    right = float(right)
    if np.isnan(left) and np.isnan(right):
        return True
    if np.isinf(left) or np.isinf(right):
        return bool(np.isinf(left) and np.isinf(right) and np.sign(left) == np.sign(right))
    return abs(left - right) <= tol * max(1.0, abs(left), abs(right))


if __name__ == "__main__":
    main()
