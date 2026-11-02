"""运行 E4。先导写入 results/e4_pilot，正式结果写入 results/e4。"""

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
from noisyopinf.e1 import refuse_existing, rho_index
from noisyopinf.e3 import _quartiles
from noisyopinf.e4 import (
    CRITERIA,
    ESTIMATORS,
    EXPERIMENT_ID,
    KAPPAS,
    PILOT,
    RANKS,
    SIGMA_METHODS,
    aggregate_median,
    aggregate_snr,
    array_sha256,
    evaluate_replicate,
    file_sha256,
    load_config,
    mode_energies,
    operator_key,
    parameter_digest,
    planned_runs,
    reference_key,
    select_from_curves,
    snr_rank,
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
    "configs/e4.yaml",
    "configs/systems.yaml",
    "src/noisyopinf/e4.py",
    "scripts/e4_rank_tradeoff.py",
    "tests/test_e4.py",
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
)
_CACHE: dict[str, dict] = {}
MEASURES = (
    "err_O_rel", "err_A_rel", "err_H_rel", "err_c_abs", "err_c_scaled",
    "specrad_hat", "specrad_ref", "specrad_err", "ep_violation", "ep_violation_ref",
    "rollout_med", "rollout_med_h10", "rollout_med_h50", "rollout_mean_finite", "diverged_frac",
    "proj_floor", "clean_val_med", "noisy_val_med",
)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "pilot":
        run_pilot()
    elif mode == "run":
        run_grid()
    elif mode == "figure":
        root = project_root()
        _draw(root / "results" / "e4", root / "figures")
    elif mode == "audit":
        target = project_root() / "results" / ("e4_pilot" if len(sys.argv) > 2 and sys.argv[2] == "pilot" else "e4")
        problems = audit_directory(target, project_root() / "data" / "e4_s2_states.npz")
        print("AUDIT_PASS" if not problems else "AUDIT_FAIL")
        if problems:
            print("PROBLEMS", json.dumps(problems, ensure_ascii=False))
            raise SystemExit(1)
    else:
        raise SystemExit("usage: e4_rank_tradeoff.py pilot|run|figure|audit [pilot]")


def run_pilot() -> None:
    out = project_root() / "results" / "e4_pilot"
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
    _write_bundle(out, config, results, {"pass": not problems, "problems": problems, "pilot_seconds": elapsed, "mean_job_seconds": job_time, "estimated_seconds": estimate, "workers": workers}, manifest)
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
    _require_committed_sources()
    out = project_root() / "results" / "e4"
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
            print("E4_RHO", rho, len(batch), flush=True)
    except Exception as exc:
        _write_bundle(out, config, results, {"pass": False, "error": str(exc)}, manifest)
        raise
    payload = {"pass": True, "elapsed_seconds": time.perf_counter() - started, "rows": _count_rows(results)}
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
    _draw(out, project_root() / "figures")
    _write_output_hashes(out)
    print("E4_DONE", payload["elapsed_seconds"], payload["rows"])


def _prepare(config: dict) -> dict:
    source = project_root() / "data" / "e3_s2_states.npz"
    if source.exists():
        with np.load(source) as loaded:
            train = np.array(loaded["train"], dtype=np.float64)
            val = np.array(loaded["val"], dtype=np.float64)
            test = np.array(loaded["test"], dtype=np.float64)
    else:
        splits = get_splits("s2", int(config["n_train"]), int(config["n_val"]), int(config["n_test"]))
        train = np.stack([np.asarray(item, dtype=np.float64) for item in splits["train"].U])
        val = np.stack([np.asarray(item, dtype=np.float64) for item in splits["val"].U])
        test = np.stack([np.asarray(item, dtype=np.float64) for item in splits["test"].U])
    cache = project_root() / "data" / "e4_s2_states.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        np.savez(cache, train=train, val=val, test=test)
    return {"cache": cache, "train": train, "val": val, "test": test}


def _run_replicates(prepared: dict, rho: float, replicates) -> list[dict]:
    tasks = [
        {"rho": float(rho), "rho_index": rho_index(float(rho)), "replicate": int(replicate), "cache": str(prepared["cache"])}
        for replicate in replicates
    ]
    return list(Parallel(n_jobs=_workers(), backend="loky")(delayed(_job)(task) for task in tasks))


def _job(task: dict) -> dict:
    started = time.perf_counter()
    state = _cached(task["cache"])
    result = evaluate_replicate(
        [state["train"][index] for index in range(state["train"].shape[0])],
        [state["val"][index] for index in range(state["val"].shape[0])],
        [state["test"][index] for index in range(state["test"].shape[0])],
        task["rho"], task["rho_index"], task["replicate"],
    )
    result["seconds"] = time.perf_counter() - started
    return result


def _cached(path: str) -> dict:
    if path not in _CACHE:
        with np.load(path) as loaded:
            _CACHE[path] = {name: np.array(loaded[name], dtype=np.float64) for name in ("train", "val", "test")}
    return _CACHE[path]


def _workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def _pilot_problems(prepared, results, estimate: float) -> list[str]:
    problems = []
    if estimate > HOUR_LIMIT:
        problems.append(f"estimated seconds {estimate} exceed 4 hours")
    rows = [row for item in results for payload in item["ranks"].values() for row in payload["rows"]]
    if len(rows) != 48:
        problems.append(f"pilot rows {len(rows)}")
    if prepared["train"].shape[0] != 30 or prepared["val"].shape[0] != 20 or prepared["test"].shape[0] != 50:
        problems.append("split sizes differ from 30/20/50")
    for item in results:
        if item["truncation_gap"] > 1e-8:
            problems.append(f"replicate {item['replicate']} truncation gap {item['truncation_gap']}")
        if abs(item["full_order_noise_rms"] - item["sigma"]) / item["sigma"] > 0.05:
            problems.append(f"noise rms {item['full_order_noise_rms']} vs sigma {item['sigma']}")
        sigmas = {float(row["sigma2_used"]) for payload in item["ranks"].values() for row in payload["rows"] if row["estimator"] in SIGMA_METHODS}
        if len(sigmas) != 1:
            problems.append(f"replicate {item['replicate']} sigma2 changes with rank")
        hashes = {payload["basis_hash"] for payload in item["ranks"].values()}
        if len(hashes) != len(RANKS):
            problems.append(f"replicate {item['replicate']} rank prefixes are not distinct")
        for rank, payload in item["ranks"].items():
            if payload["orthogonality"] > 1e-8:
                problems.append(f"r={rank} orthogonality {payload['orthogonality']}")
            thresholds = {float(detail["threshold"]) for detail in payload["details"]}
            if thresholds != {float(payload["threshold"])}:
                problems.append(f"r={rank} thresholds differ")
            ols = next(row for row in payload["rows"] if row["estimator"] == "ols")
            if ols["status"] != "ok" or ols["sigma2_source"] != "na" or ols["sigma2_used"] is not None:
                problems.append(f"r={rank} ols sigma handling")
            ridge = next(row for row in payload["rows"] if row["estimator"] == "ridge")
            if ridge["sigma2_source"] != "na":
                problems.append(f"r={rank} ridge uses sigma2")
            for method in SIGMA_METHODS:
                row = next(item_row for item_row in payload["rows"] if item_row["estimator"] == method)
                if row["sigma2_source"] != "est_trailing":
                    problems.append(f"r={rank} {method} source {row['sigma2_source']}")
            if not _reaggregate_matches(payload):
                problems.append(f"r={rank} replicate {item['replicate']} detail mismatch")
    return problems


def _reaggregate_matches(payload: dict) -> bool:
    grouped: dict[str, list] = {}
    for item in payload["details"]:
        grouped.setdefault(item["estimator"], []).append(item)
    for row in payload["rows"]:
        if row["status"] != "ok":
            continue
        chosen = sorted(grouped[row["estimator"]], key=lambda item: item["traj"])
        values = np.asarray([np.inf if item["diverged"] else item["err_full"] for item in chosen])
        if not _close(row["rollout_med"], float(np.median(values))):
            return False
    return True


def _write_bundle(out: Path, config: dict, results: list[dict], payload: dict, manifest: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = _sort_rows([row for item in results for payload_rank in item["ranks"].values() for row in payload_rank["rows"]])
    operators = {}
    for item in results:
        for payload_rank in item["ranks"].values():
            operators.update(payload_rank["operators"])
    _write_runs(out / "runs.csv", rows)
    if rows:
        _write_table(out / "summary.csv", _summary(rows, operators))
        _replace_npz(out / "operators.npz", {key: operators[key] for key in sorted(operators)})
    _write_sigma(out / "sigma2_estimation.csv", results)
    _write_ridge(out / "ridge.csv", results)
    _write_details(out / "rollout_details.npz", results)
    _write_validation(out / "validation_details.npz", results)
    _write_basis(out / "bases.npz", results)
    _write_basis_audit(out / "basis_audit.csv", results)
    curves = _curves(rows)
    _write_table(out / "rank_curves.csv", curves)
    _write_table(out / "rank_selection.csv", _with_test_performance(select_from_curves(curves), rows))
    _write_snr(out / "snr_rank.csv", results)
    _write_failures(out / "failures.json", rows)
    (out / "runtime_start.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_meta(out / "meta.json", _meta(payload, config, manifest), overwrite=True)
    _write_output_hashes(out)


def _summary(rows: list[dict], operators: dict[str, np.ndarray]) -> list[dict]:
    grouped = {(float(row["rho"]), int(row["r"]), row["estimator"], int(row["replicate"])): row for row in rows}
    cases = []
    seen = []
    for row in rows:
        key = (float(row["rho"]), int(row["r"]), row["estimator"])
        if key not in seen:
            seen.append(key)
    for rho, rank, method in seen:
        reps = sorted(rep for (item_rho, item_rank, item_method, rep) in grouped if item_rho == rho and item_rank == rank and item_method == method)
        chosen = [grouped[(rho, rank, method, rep)] for rep in reps]
        references = [operators[reference_key(rho_index(rho), rank, rep)] for rep in reps]
        stacked = [operators[operator_key(rho_index(rho), rank, method, rep)] for rep in reps]
        summary = {
            "exp": "e4", "system": "s2", "rho": rho, "r": rank, "basis": "noisy", "estimator": method,
            "sigma2_source": chosen[0]["sigma2_source"], "replicates": len(reps), "n_train": chosen[0]["n_train"],
            "n_failed": sum(row["status"] != "ok" for row in chosen), "status": "ok" if all(row["status"] == "ok" for row in chosen) else "failed",
        }
        for name in MEASURES:
            q25, median, q75 = _quartiles([row[name] for row in chosen])
            summary[f"{name}_q25"] = q25
            summary[f"{name}_median"] = median
            summary[f"{name}_q75"] = q75
            summary[f"{name}_iqr"] = 0.0 if q25 == q75 and (np.isfinite(q25) or np.isinf(q25)) else (float("nan") if np.isnan(q25) or np.isnan(q75) else float(q75 - q25))
        common = all(np.array_equal(references[0], item) for item in references[1:]) if len(references) > 1 else True
        if common and all(row["status"] == "ok" for row in chosen):
            summary["bias_O_rel"] = bias_rel(stacked, references[0])
            summary["sd_O_rel"] = sd_rel(stacked, references[0])
            summary.update(block_bias_sd(stacked, references[0]))
        else:
            for field in ("bias_O_rel", "sd_O_rel", "bias_A_rel", "sd_A_rel", "bias_H_rel", "sd_H_rel"):
                summary[field] = float("nan")
        gaps = []
        for rep in reps:
            gaps.append(float(grouped[(rho, rank, method, rep)]["err_O_rel"]) - float(grouped[(rho, rank, "ols", rep)]["err_O_rel"]))
        summary["_gaps"] = np.asarray(gaps, dtype=np.float64)
        summary["paired_median"] = float(np.median(summary["_gaps"]))
        summary["paired_iqr"] = iqr(summary["_gaps"])
        cases.append(summary)
    generated = rng(0, 77)
    for summary in cases:
        low, high = bootstrap_interval(summary.pop("_gaps"), generated, 2000)
        summary["boot_low"] = low
        summary["boot_high"] = high
    return cases


def _curves(rows: list[dict]) -> list[dict]:
    return [
        {"rho": row["rho"], "r": row["r"], "estimator": row["estimator"], "replicate": row["replicate"], "clean_val_med": row["clean_val_med"], "noisy_val_med": row["noisy_val_med"], "test_med": row["rollout_med"]}
        for row in rows
    ]


def _with_test_performance(selected: list[dict], rows: list[dict]) -> list[dict]:
    lookup = {}
    for row in rows:
        lookup.setdefault((float(row["rho"]), int(row["r"]), row["estimator"]), []).append(float(row["rollout_med"]))
    for item in selected:
        values = lookup[(float(item["rho"]), int(item["r_selected"]), item["estimator"])]
        item["test_rollout_median"] = aggregate_median(values)
    return selected


def _write_sigma(path: Path, results: list[dict]) -> None:
    fields = ["rho", "replicate", "sigma2_true", "sigma2_est_trailing", "relative", "applicable", "N", "K_tot", "beta", "error"]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for item in sorted(results, key=lambda value: (value["rho"], value["replicate"])):
        estimate = item["estimate"]
        writer.writerow({
            "rho": _cell(item["rho"]), "replicate": item["replicate"], "sigma2_true": _cell(item["sigma2_true"]),
            "sigma2_est_trailing": _cell(estimate["value"]), "relative": _cell(estimate["relative"]),
            "applicable": estimate["applicable"], "N": estimate["diagnostics"].get("N", ""),
            "K_tot": estimate["diagnostics"].get("K_tot", ""), "beta": _cell(estimate["diagnostics"].get("beta")),
            "error": estimate["error"],
        })
    path.write_text(buffer.getvalue(), encoding="utf-8", newline="\n")


def _write_ridge(path: Path, results: list[dict]) -> None:
    fields = ["rho", "r", "replicate", "lam", "score", "chosen", "n_tied", "n_nonfinite"]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for item in results:
        for rank, payload in item["ranks"].items():
            for lam, score in zip(LAMBDAS, payload["ridge"]["scores"]):
                writer.writerow({"rho": _cell(item["rho"]), "r": rank, "replicate": item["replicate"], "lam": _cell(lam), "score": _cell(score), "chosen": _cell(payload["ridge"]["lam"]), "n_tied": payload["ridge"]["n_tied"], "n_nonfinite": payload["ridge"]["n_nonfinite"]})
    path.write_text(buffer.getvalue(), encoding="utf-8", newline="\n")


def _write_details(path: Path, results: list[dict]) -> None:
    details = [item for result in results for payload in result["ranks"].values() for item in payload["details"]]
    details.sort(key=lambda item: (item["rho"], item["r"], ESTIMATORS.index(item["estimator"]), item["replicate"], item["traj"]))
    _replace_npz(path, {
        "rho": np.asarray([item["rho"] for item in details], dtype=np.float64),
        "r": np.asarray([item["r"] for item in details], dtype=np.int16),
        "estimator": np.asarray([ESTIMATORS.index(item["estimator"]) for item in details], dtype=np.int16),
        "replicate": np.asarray([item["replicate"] for item in details], dtype=np.int16),
        "traj": np.asarray([item["traj"] for item in details], dtype=np.int16),
        "err_full": np.asarray([item["err_full"] for item in details], dtype=np.float64),
        "err_h10": np.asarray([item["err_h10"] for item in details], dtype=np.float64),
        "err_h50": np.asarray([item["err_h50"] for item in details], dtype=np.float64),
        "diverged": np.asarray([item["diverged"] for item in details], dtype=np.bool_),
        "first_step": np.asarray([item["first_step"] for item in details], dtype=np.int16),
        "threshold": np.asarray([item["threshold"] for item in details], dtype=np.float64),
        "proj_floor": np.asarray([item["proj_floor"] for item in details], dtype=np.float64),
    })


def _write_validation(path: Path, results: list[dict]) -> None:
    rows = [item for result in results for payload in result["ranks"].values() for item in payload["validation"]]
    kind = {"clean": 0, "noisy": 1}
    _replace_npz(path, {
        "rho": np.asarray([item["rho"] for item in rows], dtype=np.float64),
        "r": np.asarray([item["r"] for item in rows], dtype=np.int16),
        "estimator": np.asarray([ESTIMATORS.index(item["estimator"]) for item in rows], dtype=np.int16),
        "replicate": np.asarray([item["replicate"] for item in rows], dtype=np.int16),
        "kind": np.asarray([kind[item["kind"]] for item in rows], dtype=np.int16),
        "traj": np.asarray([item["traj"] for item in rows], dtype=np.int16),
        "err": np.asarray([item["err"] for item in rows], dtype=np.float64),
        "diverged": np.asarray([item["diverged"] for item in rows], dtype=np.bool_),
    })


def _write_basis(path: Path, results: list[dict]) -> None:
    arrays = {}
    for item in results:
        arrays[f"v_rho{item['rho_index']}_rep{item['replicate']:03d}"] = item["basis"]
        arrays[f"s_rho{item['rho_index']}_rep{item['replicate']:03d}"] = item["singular_values"]
        arrays[f"lam_rho{item['rho_index']}_rep{item['replicate']:03d}"] = item["energies"]
    _replace_npz(path, arrays)


def _write_basis_audit(path: Path, results: list[dict]) -> None:
    fields = ["rho", "r", "replicate", "threshold", "orthogonality", "basis_hash", "reference_hash", "proj_floor", "clean_q_rms", "truncation_gap", "parent_basis_hash"]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for item in results:
        for rank, payload in item["ranks"].items():
            writer.writerow({"rho": _cell(item["rho"]), "r": rank, "replicate": item["replicate"], "threshold": _cell(payload["threshold"]), "orthogonality": _cell(payload["orthogonality"]), "basis_hash": payload["basis_hash"], "reference_hash": payload["reference_hash"], "proj_floor": _cell(payload["proj_floor"]), "clean_q_rms": _cell(payload["clean_q_rms"]), "truncation_gap": _cell(item["truncation_gap"]), "parent_basis_hash": item["basis_hash"]})
    path.write_text(buffer.getvalue(), encoding="utf-8", newline="\n")


def _write_snr(path: Path, results: list[dict]) -> None:
    fields = ["rho", "replicate", "kappa", "sigma2", "mode", "lambda", "passes", "r_snr", "empty"]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for item in results:
        for kappa in KAPPAS:
            chosen, passed = snr_rank(item["energies"], item["estimate"]["value"], kappa)
            for index, value in enumerate(item["energies"]):
                writer.writerow({"rho": _cell(item["rho"]), "replicate": item["replicate"], "kappa": kappa, "sigma2": _cell(item["estimate"]["value"]), "mode": index + 1, "lambda": _cell(float(value)), "passes": int(index + 1 in passed), "r_snr": "" if chosen is None else chosen, "empty": int(chosen is None)})
    path.write_text(buffer.getvalue(), encoding="utf-8", newline="\n")


def _write_failures(path: Path, rows: list[dict]) -> None:
    failed = [{"rho": row["rho"], "r": row["r"], "estimator": row["estimator"], "replicate": row["replicate"], "error": row["error"]} for row in rows if row["status"] != "ok"]
    path.write_text(json.dumps(failed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sort_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: (float(row["rho"]), int(row["r"]), int(row["replicate"]), ESTIMATORS.index(row["estimator"])))


def _count_rows(results: list[dict]) -> int:
    return sum(len(payload["rows"]) for item in results for payload in item["ranks"].values())


def _write_runs(path: Path, rows: list[dict]) -> None:
    from noisyopinf.io_utils import write_runs

    write_runs(path, [{column: row.get(column) for column in RUN_COLUMNS} for row in rows], overwrite=True)


def _write_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _cell(value) for key, value in row.items()})
    path.write_text(buffer.getvalue(), encoding="utf-8", newline="\n")


def _replace_npz(path: Path, arrays: dict) -> None:
    temporary = path.with_name(path.stem + ".writing.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _write_output_hashes(out: Path) -> None:
    rows = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "output_manifest.json":
            rows.append({"path": str(path.relative_to(project_root())), "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    (out / "output_manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _require_pilot() -> None:
    pilot = json.loads((project_root() / "results" / "e4_pilot" / "meta.json").read_text(encoding="utf-8"))
    if not pilot.get("pass") or float(pilot.get("estimated_seconds", HOUR_LIMIT + 1)) > HOUR_LIMIT:
        raise RuntimeError(f"pilot gate is not open: pass={pilot.get('pass')} estimate={pilot.get('estimated_seconds')}")


def _require_committed_sources() -> None:
    dirty = subprocess.check_output(["git", "status", "--short", "--", *HASH_PATHS], cwd=project_root(), text=True)
    if dirty.strip():
        raise RuntimeError("tracked E4 sources differ from HEAD:\n" + dirty)


def _manifest(config: dict, phase: str) -> dict:
    root = project_root()
    return {
        "phase": phase,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True, stderr=subprocess.STDOUT),
        "command": " ".join(sys.argv),
        "experiment_id": EXPERIMENT_ID,
        "seed_protocol": "noise keys (2, 4, 4, rho_index, replicate, traj); validation traj + 100; rank tie smaller_r; snr plot lower central median",
        "parameter_hash": parameter_digest(),
        "files": {relative: file_sha256(root / relative) for relative in HASH_PATHS},
        "config": config,
        "runtime": runtime_facts(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _hash_drift(started: dict, finished: dict) -> list[str]:
    return [f"hash changed during run: {relative}" for relative, digest in started["files"].items() if finished["files"].get(relative) != digest]


def _meta(payload: dict, config: dict, manifest: dict) -> dict:
    meta = runtime_facts()
    meta.update(payload)
    meta.update({"git_head": manifest["git_head"], "config": config, "experiment_id": EXPERIMENT_ID, "parameter_hash": manifest["parameter_hash"], "planned_runs": planned_runs(config), "tie_rule": config["tie_rule"], "rank_aggregate": config["rank_aggregate"], "snr_plot_aggregate": config["snr_plot_aggregate"]})
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


def _close(left: float, right: float) -> bool:
    if np.isnan(left) and np.isnan(right):
        return True
    if np.isposinf(left) and np.isposinf(right):
        return True
    return abs(float(left) - float(right)) <= 1e-12


def audit_directory(out: Path, cache: Path) -> list[str]:
    problems = []
    rows = read_runs(out / "runs.csv")
    with np.load(out / "operators.npz") as loaded:
        operators = {key: np.array(loaded[key], dtype=np.float64) for key in loaded.files}
    with np.load(out / "bases.npz") as loaded:
        bases = {key: np.array(loaded[key], dtype=np.float64) for key in loaded.files}
    with np.load(cache) as loaded:
        train = np.array(loaded["train"], dtype=np.float64)
        test = np.array(loaded["test"], dtype=np.float64)
    audit_rows = _read_table(out / "basis_audit.csv")
    audit_of = {(float(item["rho"]), int(float(item["r"])), int(float(item["replicate"]))): item for item in audit_rows}
    checked = set()
    for row in rows:
        index = rho_index(float(row["rho"]))
        rank = int(row["r"])
        replicate = int(row["replicate"])
        hat = operators[operator_key(index, rank, row["estimator"], replicate)]
        ref = operators[reference_key(index, rank, replicate)]
        record = audit_of[(float(row["rho"]), rank, replicate)]
        if array_sha256(ref) != record["reference_hash"]:
            problems.append(f"reference hash {rank} r{replicate}")
        parent = bases[f"v_rho{index}_rep{replicate:03d}"]
        prefix = parent[:, :rank]
        if array_sha256(prefix) != record["basis_hash"]:
            problems.append(f"prefix hash {rank} r{replicate}")
        token = (float(row["rho"]), rank, replicate)
        if token not in checked:
            if not _close_num(state_rms(np.vstack([project(train[item], prefix) for item in range(train.shape[0])])), float(record["clean_q_rms"])):
                problems.append(f"rms {token}")
            if not _close_num(float(proj_floor(prefix, np.vstack(test))), float(record["proj_floor"])):
                problems.append(f"floor {token}")
            energies = mode_energies(np.vstack([project(train[item], parent) for item in range(train.shape[0])]))
            if not np.allclose(energies, bases[f"lam_rho{index}_rep{replicate:03d}"]):
                problems.append(f"lambda {token}")
            checked.add(token)
        if row["status"] != "ok":
            continue
        expected = {
            "err_O_rel": err_O_rel(hat, ref), "err_A_rel": err_A_rel(hat, ref), "err_H_rel": err_H_rel(hat, ref),
            "err_c_abs": err_c_abs(hat, ref), "err_c_scaled": err_c_scaled(hat, ref, float(record["clean_q_rms"])),
        }
        spec = specrad_trio(hat, ref)
        expected.update({"specrad_hat": spec[0], "specrad_ref": spec[1], "specrad_err": spec[2]})
        projected = np.vstack([project(test[item], prefix) for item in range(test.shape[0])])
        expected["ep_violation"] = ep_violation(hat, projected)
        for field, value in expected.items():
            if not _close_num(row[field], value):
                problems.append(f"{field} {row['estimator']} r={rank} rep={replicate}")
                break
    problems.extend(_audit_details(out, rows))
    problems.extend(_audit_selection(out, rows))
    problems.extend(_audit_snr(out))
    problems.extend(_audit_sigma_constant(rows))
    problems.extend(_audit_ridge(out, rows))
    return problems[:40]


def _audit_details(out: Path, rows: list[dict]) -> list[str]:
    with np.load(out / "rollout_details.npz") as loaded:
        payload = {key: np.array(loaded[key]) for key in loaded.files}
    grouped: dict[tuple, list] = {}
    for index in range(payload["rho"].shape[0]):
        key = (float(payload["rho"][index]), int(payload["r"][index]), ESTIMATORS[int(payload["estimator"][index])], int(payload["replicate"][index]))
        grouped.setdefault(key, []).append(index)
    problems = []
    for row in rows:
        if row["status"] != "ok":
            continue
        key = (float(row["rho"]), int(row["r"]), row["estimator"], int(row["replicate"]))
        indexes = grouped[key]
        order = np.argsort([int(payload["traj"][item]) for item in indexes])
        indexes = [indexes[item] for item in order]
        values = np.asarray([np.inf if bool(payload["diverged"][item]) else float(payload["err_full"][item]) for item in indexes])
        if not _close_num(row["rollout_med"], float(np.median(values))):
            problems.append(f"detail rollout {key}")
    return problems


def _audit_selection(out: Path, rows: list[dict]) -> list[str]:
    curves = _read_numeric(out / "rank_curves.csv")
    rebuilt = select_from_curves(curves)
    saved = _read_numeric(out / "rank_selection.csv")
    problems = []
    saved_of = {(float(item["rho"]), item["estimator"], item["criterion"]): item for item in saved}
    for item in rebuilt:
        found = saved_of[(float(item["rho"]), item["estimator"], item["criterion"])]
        if int(float(found["r_selected"])) != int(item["r_selected"]) or not _close_num(float(found["selection_score"]), float(item["selection_score"])):
            problems.append(f"selection {item['criterion']} {item['estimator']} rho={item['rho']}")
    with np.load(out / "validation_details.npz") as loaded:
        payload = {key: np.array(loaded[key]) for key in loaded.files}
    kind_name = {0: "clean", 1: "noisy"}
    grouped: dict[tuple, list] = {}
    for index in range(payload["rho"].shape[0]):
        key = (float(payload["rho"][index]), int(payload["r"][index]), ESTIMATORS[int(payload["estimator"][index])], int(payload["replicate"][index]), kind_name[int(payload["kind"][index])])
        grouped.setdefault(key, []).append(float(payload["err"][index]))
    curve_of = {(float(item["rho"]), int(float(item["r"])), item["estimator"], int(float(item["replicate"]))): item for item in curves}
    for key, values in grouped.items():
        rho, rank, method, replicate, kind = key
        field = "clean_val_med" if kind == "clean" else "noisy_val_med"
        if not _close_num(float(curve_of[(rho, rank, method, replicate)][field]), aggregate_median(values)):
            problems.append(f"validation curve {key}")
            break
    return problems


def _audit_snr(out: Path) -> list[str]:
    rows = _read_numeric(out / "snr_rank.csv")
    grouped: dict[tuple, list] = {}
    for row in rows:
        grouped.setdefault((float(row["rho"]), int(float(row["replicate"])), int(float(row["kappa"]))), []).append(row)
    problems = []
    for key, items in grouped.items():
        items = sorted(items, key=lambda item: int(float(item["mode"])))
        energies = np.asarray([float(item["lambda"]) for item in items])
        sigma2 = float(items[0]["sigma2"])
        kappa = key[2]
        chosen, passed = snr_rank(energies, sigma2, kappa)
        recorded = items[0]["r_snr"]
        recorded_value = None if recorded in {"", None} else int(float(recorded))
        if recorded_value != chosen or int(float(items[0]["empty"])) != int(chosen is None):
            problems.append(f"snr {key}")
        if any(int(float(item["passes"])) != int(int(float(item["mode"])) in passed) for item in items):
            problems.append(f"snr pass flags {key}")
    return problems


def _audit_sigma_constant(rows: list[dict]) -> list[str]:
    grouped: dict[tuple, set] = {}
    for row in rows:
        if row["estimator"] not in SIGMA_METHODS:
            if row["sigma2_source"] != "na":
                return [f"sigma source on {row['estimator']}"]
            continue
        grouped.setdefault((float(row["rho"]), int(row["replicate"]), row["estimator"]), set()).add(float(row["sigma2_used"]))
    return [f"sigma2 varies {key}" for key, values in grouped.items() if len(values) != 1]


def _audit_ridge(out: Path, rows: list[dict]) -> list[str]:
    table = _read_numeric(out / "ridge.csv")
    grouped: dict[tuple, list] = {}
    for item in table:
        grouped.setdefault((float(item["rho"]), int(float(item["r"])), int(float(item["replicate"]))), []).append(item)
    problems = []
    for key, records in grouped.items():
        records = sorted(records, key=lambda item: float(item["lam"]))
        chosen = float(LAMBDAS[larger_lambda_index([float(item["score"]) for item in records])])
        matched = [row for row in rows if float(row["rho"]) == key[0] and int(row["r"]) == key[1] and int(row["replicate"]) == key[2] and row["estimator"] == "ridge"]
        if len(matched) != 1 or not _close_num(float(matched[0]["lam"]), chosen):
            problems.append(f"ridge {key}")
    return problems


def _read_table(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_numeric(path: Path) -> list[dict]:
    rows = []
    for raw in _read_table(path):
        row = {}
        for key, value in raw.items():
            if value in {"", "nan"}:
                row[key] = "" if key in {"r_snr", "estimator", "criterion"} and value == "" else (value if key in {"estimator", "criterion"} else float("nan"))
            elif key in {"estimator", "criterion"}:
                row[key] = value
            else:
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
        rows.append(row)
    return rows


def _close_num(left, right, tol: float = 1e-8) -> bool:
    left = float(left)
    right = float(right)
    if np.isnan(left) and np.isnan(right):
        return True
    if np.isinf(left) or np.isinf(right):
        return bool(np.isinf(left) and np.isinf(right) and np.sign(left) == np.sign(right))
    return abs(left - right) <= tol * max(1.0, abs(left), abs(right))


def _draw(out: Path, folder: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    summary = _read_numeric(out / "summary.csv")
    selected = _read_numeric(out / "rank_selection.csv")
    snr = _read_numeric(out / "snr_rank.csv")
    folder.mkdir(parents=True, exist_ok=True)
    _heatmap(summary, selected, folder / "fig_e4_heatmap.png")
    _rstar(selected, snr, folder / "fig_e4_rstar.png")
    print("FIGURES", folder)


def _nearest(values, target: float) -> int:
    return int(np.argmin([abs(float(item) - float(target)) for item in values]))


def _heatmap(summary, selected, path: Path) -> None:
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    rhos = [0.01, 0.02, 0.05, 0.1, 0.2]
    for axis, method in zip(axes.ravel(), ESTIMATORS, strict=True):
        grid = np.full((len(RANKS), len(rhos)), np.nan)
        for row_index, rank in enumerate(RANKS):
            for column, rho in enumerate(rhos):
                match = [item for item in summary if item["estimator"] == method and int(float(item["r"])) == rank and abs(float(item["rho"]) - rho) < 1e-12]
                value = float(match[0]["rollout_med_median"])
                if np.isfinite(value) and value > 0.0:
                    grid[row_index, column] = value
                else:
                    print("HEATMAP_NONFINITE", method, rank, rho, value)
                    axis.scatter(column, row_index, marker="x", color="black", s=30, zorder=3)
        image = axis.imshow(np.log10(grid), aspect="auto", origin="lower", cmap="viridis")
        marks = [item for item in selected if item["estimator"] == method and item["criterion"] == "val"]
        for item in marks:
            axis.scatter(_nearest(rhos, float(item["rho"])), list(RANKS).index(int(float(item["r_selected"]))), facecolors="none", edgecolors="white", s=80, linewidths=1.5)
        axis.set_xticks(range(len(rhos)), [str(rho) for rho in rhos])
        axis.set_yticks(range(len(RANKS)), [str(rank) for rank in RANKS])
        axis.set_title(method)
        axis.set_xlabel("rho")
        axis.set_ylabel("r")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("test rollout_med (log10); white circle is r_star_val, not fit on the test set")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _rstar(selected, snr, path: Path) -> None:
    import matplotlib.pyplot as plt
    rhos = [0.01, 0.02, 0.05, 0.1, 0.2]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for axis, method in zip(axes.ravel(), ESTIMATORS, strict=True):
        for criterion, marker, label in (("val", "o", "r_star_val"), ("noisyval", "s", "r_star_noisyval"), ("test", "x", "r_star_test descriptive")):
            chosen = [item for item in selected if item["estimator"] == method and item["criterion"] == criterion]
            axis.plot([float(item["rho"]) for item in chosen], [float(item["r_selected"]) for item in chosen], marker=marker, label=label)
        for kappa, style in zip(KAPPAS, ("-", "--", ":", "-."), strict=True):
            ys = []
            for rho in rhos:
                values = []
                for replicate in sorted({int(float(item["replicate"])) for item in snr if abs(float(item["rho"]) - rho) < 1e-12 and int(float(item["kappa"])) == kappa and int(float(item["mode"])) == 1}):
                    sample = next(item for item in snr if abs(float(item["rho"]) - rho) < 1e-12 and int(float(item["replicate"])) == replicate and int(float(item["kappa"])) == kappa and int(float(item["mode"])) == 1)
                    text = sample["r_snr"]
                    values.append(None if text in {"", "nan"} or (isinstance(text, float) and np.isnan(text)) else int(float(text)))
                aggregated, _empty = aggregate_snr(values)
                ys.append(np.nan if aggregated is None else aggregated)
                if aggregated is None:
                    print("SNR_EMPTY", kappa, rho)
            axis.plot(rhos, ys, linestyle=style, color="black", label=f"snr kappa={kappa}")
        axis.set_title(method)
        axis.set_xlabel("rho")
        axis.set_ylabel("rank")
        axis.legend(fontsize=7)
    figure.suptitle("selection ranks by criterion; SNR lines use the lower central median across replicates")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
