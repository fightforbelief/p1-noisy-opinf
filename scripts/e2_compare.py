"""运行 E2。先导写入 results/e2_pilot，正式结果写入 results/e2。"""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from joblib import Parallel, delayed

from noisyopinf.e1 import nested_prefix, refuse_existing, rho_index, true_operator
from noisyopinf.e2 import (
    BOOTSTRAP,
    ESTIMATORS,
    LAMBDAS,
    N_TEST,
    N_VAL,
    PILOT,
    SIGMA_METHODS,
    _workers,
    add_split_noise,
    csv_row,
    evaluate_job,
    expand_cases,
    extrapolate_seconds,
    load_config,
    load_s1_split,
    operator_key,
    pairs_per_estimator,
    shared_threshold,
    sigma2_of,
    summary_rows,
    write_state_cache,
)
from noisyopinf.estimators import fit
from noisyopinf.io_utils import (
    read_operators,
    read_runs,
    runtime_facts,
    write_meta,
    write_operators,
    write_runs,
)
from noisyopinf.metrics import rollout_aggregate, rollout_err_h
from noisyopinf.protocol import select_ridge_lambda
from noisyopinf.rollout import rollout
from noisyopinf.systems.config import project_root

HOUR_LIMIT = 4.0 * 3600.0


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "pilot":
        run_pilot()
    elif mode == "run":
        run_grid()
    elif mode == "figure":
        redraw_figure()
    else:
        raise SystemExit("usage: e2_compare.py pilot|run|figure")


def run_pilot() -> None:
    """rho=0.05、n_train=30、replicate 0 和 1。不写入 results/e2。"""
    out = project_root() / "results" / "e2_pilot"
    refuse_existing(out)
    config = load_config()
    prepared = _prepare()
    case = _pilot_case()
    started = time.perf_counter()
    results = _run_case(prepared, case, PILOT["replicates"])
    elapsed = time.perf_counter() - started
    estimate = extrapolate_seconds(elapsed, len(PILOT["replicates"]))
    problems = _pilot_problems(prepared, case, results, estimate)
    payload = {
        "pass": not problems, "problems": problems, "pilot_seconds": elapsed,
        "estimated_seconds": estimate, "pair_weighted_seconds": estimate * 140.0 / 90.0,
    }
    _write_bundle(out, config, prepared["truth"], [_pilot_case()], results, payload)
    problems.extend(_saved_problems(out, results))
    if problems:
        payload["pass"] = False
        payload["problems"] = problems
        write_meta(out / "meta.json", _meta(payload, config), overwrite=True)
    print("PILOT_PASS" if not problems else "PILOT_FAIL")
    print("ESTIMATE_SECONDS", estimate)
    print("PAIR_WEIGHTED_SECONDS", payload["pair_weighted_seconds"])
    if problems:
        raise SystemExit(1)


def run_grid() -> None:
    """18 个配置。先导未通过，或按配对数加权后的估算超过 4 小时，则不运行。"""
    out = project_root() / "results" / "e2"
    refuse_existing(out)
    _require_pilot()
    config = load_config()
    prepared = _prepare()
    out.mkdir(parents=True)
    started = time.perf_counter()
    results = []
    try:
        for case in expand_cases(config):
            results.extend(_run_case(prepared, case, range(case["replicates"])))
            print("E2_CASE", case["rho"], case["n_train"], flush=True)
    except Exception as exc:
        _write_partial(out, config, prepared["truth"], results, exc)
        raise
    payload = {"pass": True, "elapsed_seconds": time.perf_counter() - started, "rows": sum(len(item["rows"]) for item in results)}
    _write_bundle(out, config, prepared["truth"], expand_cases(config), results, payload)
    _draw(out / "summary.csv", project_root() / "figures")
    print("E2_DONE", payload["elapsed_seconds"], payload["rows"])


def redraw_figure() -> None:
    """只根据已有 summary.csv 重画五张图。"""
    _draw(project_root() / "results" / "e2" / "summary.csv", project_root() / "figures")


def _prepare() -> dict:
    train = load_s1_split(1, 100)
    val = load_s1_split(2, N_VAL)
    test = load_s1_split(3, N_TEST)
    cache = project_root() / "data" / "e2_s1_states.npz"
    write_state_cache(cache, train, val, test)
    thresholds = {n_train: shared_threshold(train[:n_train]) for n_train in (10, 30, 100)}
    return {"train": train, "val": val, "test": test, "truth": true_operator(), "cache": cache, "thresholds": thresholds}


def _pilot_case() -> dict:
    return {"rho": PILOT["rho"], "n_train": PILOT["n_train"], "rho_index": rho_index(PILOT["rho"]), "replicates": len(PILOT["replicates"])}


def _run_case(prepared: dict, case: dict, replicates) -> list[dict]:
    truth = prepared["truth"]
    tasks = [_task(prepared, case, replicate, truth) for replicate in replicates]
    return Parallel(n_jobs=_workers(), backend="loky")(delayed(evaluate_job)(task) for task in tasks)


def _task(prepared: dict, case: dict, replicate: int, truth: np.ndarray) -> dict:
    return {
        "rho": case["rho"], "rho_index": case["rho_index"], "n_train": case["n_train"],
        "replicate": int(replicate), "threshold": prepared["thresholds"][case["n_train"]],
        "cache": str(prepared["cache"]), "truth": truth,
    }


def _require_pilot() -> None:
    path = project_root() / "results" / "e2_pilot" / "meta.json"
    pilot = json.loads(path.read_text(encoding="utf-8"))
    weighted = float(pilot.get("pair_weighted_seconds", pilot["estimated_seconds"]))
    if not pilot.get("pass") or weighted > HOUR_LIMIT:
        raise RuntimeError(f"pilot gate is not open: pass={pilot.get('pass')} weighted={weighted}")


def _pilot_problems(prepared, case, results, estimate: float) -> list[str]:
    problems = _prefix_problems(prepared["train"])
    rows = [row for item in results for row in item["rows"]]
    if len(results) != 2 or {row["estimator"] for row in rows} != set(ESTIMATORS):
        problems.append("pilot did not run seven estimators on two replicates")
    problems.extend(_fit_problems(prepared, case, results))
    problems.extend(_metric_problems(prepared, case, results, rows))
    if estimate <= 0.0:
        problems.append("estimate is not positive")
    return problems


def _prefix_problems(train) -> list[str]:
    problems = []
    if any(item is not origin for item, origin in zip(nested_prefix(train, 10), train[:10], strict=True)):
        problems.append("n_train=10 is not a nested prefix")
    if any(item is not origin for item, origin in zip(nested_prefix(train, 30), train[:30], strict=True)):
        problems.append("n_train=30 is not a nested prefix")
    return problems


def _fit_problems(prepared, case, results) -> list[str]:
    problems = []
    clean = prepared["train"][: case["n_train"]]
    item = results[0]
    sigma = float(np.sqrt(item["sigma2"]))
    noisy = add_split_noise(clean, sigma, case["rho_index"], item["replicate"], False)
    if not np.allclose(item["operators"]["oracle"], fit(clean, "oracle").O, rtol=1e-8, atol=1e-8):
        problems.append("oracle was not fit on the clean prefix")
    if not np.allclose(item["operators"]["ols"], fit(noisy, "ols").O, rtol=1e-8, atol=1e-8):
        problems.append("ols was not fit on the shared noisy training data")
    limit = prepared["thresholds"][case["n_train"]]
    if any(row["blowup_threshold"] != limit for result in results for row in result["rows"]):
        problems.append("estimators do not share the clean-prefix threshold")
    if item["ridge"]["lam"] != select_ridge_lambda(noisy, add_split_noise(prepared["val"], sigma, case["rho_index"], item["replicate"], True), limit):
        problems.append("ridge lambda does not match select_ridge_lambda")
    return problems


def _metric_problems(prepared, case, results, rows) -> list[str]:
    problems = []
    limit = prepared["thresholds"][case["n_train"]]
    clean = prepared["train"][: case["n_train"]]
    length = int(clean[0].shape[0])
    if any(row["n_pairs"] != pairs_per_estimator(row["estimator"], case["n_train"], length) for row in rows):
        problems.append("pairs cross trajectories or use the wrong length")
    if len({row["sigma2_true"] for row in rows}) != 1 or rows[0]["sigma2_true"] != sigma2_of(clean, case["rho"]):
        problems.append("sigma2 is not shared")
    if any(_bad_source(row) for row in rows):
        problems.append("sigma2_source does not follow the estimator")
    finite = [row for row in rows if row["status"] != "ok" or not np.isfinite(row["err_O_rel"])]
    if finite:
        problems.append("non-finite pilot fit: " + ",".join(f"{row['estimator']}:{row['error']}" for row in finite))
    problems.extend(_rollout_problems(prepared["test"], results[0], limit))
    manual = summary_rows([case], rows, _archive(results, prepared["truth"]), prepared["truth"])
    stored = [row["err_O_rel"] for row in rows if row["estimator"] == "ols"]
    if abs(float(np.median(stored)) - manual[ESTIMATORS.index("ols")]["err_O_rel_median"]) > 1e-12:
        problems.append("summary median does not match the rows")
    return problems


def _bad_source(row: dict) -> bool:
    if row["estimator"] in SIGMA_METHODS:
        return row["sigma2_source"] != "true" or row["sigma2_used"] != row["sigma2_true"]
    return row["sigma2_source"] != "na" or row["sigma2_used"] is not None


def _rollout_problems(test, item, limit: float) -> list[str]:
    operator = item["operators"]["ols"]
    errors, flags, early = [], [], []
    for traj in test:
        steps = int(traj.shape[0] - 1)
        predicted, diverged = rollout(operator, traj[0], steps, limit)
        errors.append(float("inf") if diverged else rollout_err_h(predicted, traj, steps))
        flags.append(diverged)
        early.append(rollout_err_h(predicted, traj, 10))
    summary = rollout_aggregate(errors, flags)
    row = next(entry for entry in item["rows"] if entry["estimator"] == "ols")
    h50 = [rollout_err_h(rollout(operator, traj[0], int(traj.shape[0] - 1), limit)[0], traj, 50) for traj in test]
    problems = []
    if not _close_metric(summary["rollout_med"], row["rollout_med"]):
        problems.append("full-horizon median does not match an independent recalculation")
    if not _close_metric(float(np.median(early)), row["rollout_med_h10"]):
        problems.append("h=10 median does not match an independent recalculation")
    if not _close_metric(float(np.median(h50)), row["rollout_med_h50"]):
        problems.append("h=50 median does not match an independent recalculation")
    return problems


def _close_metric(left: float, right: float) -> bool:
    if np.isnan(left) and np.isnan(right):
        return True
    if left == right:
        return True
    if not np.isfinite(left) or not np.isfinite(right):
        return False
    return abs(left - right) <= 1e-8 * max(1.0, abs(right))


def _saved_problems(out: Path, results) -> list[str]:
    problems = []
    loaded = read_runs(out / "runs.csv")
    rows = _sort_rows([row for item in results for row in item["rows"]])
    if [row["replicate"] for row in loaded] != [row["replicate"] for row in rows]:
        problems.append("runs.csv did not round-trip")
    saved = read_operators(out / "operators.npz")
    key = operator_key(results[0]["rho_index"], results[0]["n_train"], "ols", results[0]["replicate"])
    if key not in saved or not np.allclose(saved[key], results[0]["operators"]["ols"], equal_nan=True):
        problems.append("operator archive does not match the fit")
    return problems


def _write_partial(out: Path, config, truth, results, exc: Exception) -> None:
    rows = _sort_rows([row for item in results for row in item["rows"]])
    write_meta(out / "meta.json", _meta({"pass": False, "error": str(exc), "rows": len(rows)}, config))
    write_runs(out / "runs.csv", [csv_row(row) for row in rows])
    _write_ridge(out / "ridge.csv", results)
    _write_failures(out / "failures.json", rows)
    if rows:
        write_operators(out / "operators.npz", _archive(results, truth))


def _write_bundle(out: Path, config, truth, cases, results, payload: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = _sort_rows([row for item in results for row in item["rows"]])
    write_meta(out / "meta.json", _meta(payload, config))
    write_runs(out / "runs.csv", [csv_row(row) for row in rows])
    if rows:
        _write_table(out / "summary.csv", summary_rows(cases, rows, _archive(results, truth), truth))
    _write_ridge(out / "ridge.csv", results)
    _write_failures(out / "failures.json", rows)
    if rows:
        write_operators(out / "operators.npz", _archive(results, truth))


def _archive(results, truth) -> dict[str, np.ndarray]:
    arrays = {"true": np.asarray(truth, dtype=np.float64)}
    for item in results:
        for method, operator in item["operators"].items():
            arrays[operator_key(item["rho_index"], item["n_train"], method, item["replicate"])] = operator
    return arrays


def _sort_rows(rows: list[dict]) -> list[dict]:
    order = {method: index for index, method in enumerate(ESTIMATORS)}
    return sorted(rows, key=lambda row: (float(row["rho"]), int(row["n_train"]), int(row["replicate"]), order[row["estimator"]]))


def _meta(payload: dict, config: dict) -> dict:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root(), text=True).strip()
    meta = runtime_facts()
    meta.update(payload)
    meta.update({
        "git_head": head, "config": config, "bootstrap": BOOTSTRAP,
        "threshold_rule": "1e3 times the max infinity norm of the clean training prefix",
        "validation_offset": 100, "experiment_id": 2,
        "rho_axis": "linear, includes rho=0",
    })
    return meta


def _write_table(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _cell(value) for key, value in row.items()})
    path.write_text(buffer.getvalue(), encoding="utf-8", newline="\n")


def _write_ridge(path: Path, results) -> None:
    fields = ["rho", "n_train", "replicate", "lam", "score", "chosen", "n_tied", "n_nonfinite", "seconds", "seed_keys"]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for item in results:
        for lam, score in zip(LAMBDAS, item["ridge"]["scores"]):
            writer.writerow({
                "rho": _cell(item["rho"]), "n_train": item["n_train"], "replicate": item["replicate"],
                "lam": _cell(lam), "score": _cell(score), "chosen": _cell(item["ridge"]["lam"]),
                "n_tied": item["ridge"]["n_tied"], "n_nonfinite": item["ridge"]["n_nonfinite"],
                "seconds": _cell(item["ridge"]["seconds"]),
                "seed_keys": f"1,4,2,{item['rho_index']},{item['replicate']},traj+100",
            })
    path.write_text(buffer.getvalue(), encoding="utf-8", newline="\n")


def _write_failures(path: Path, rows: list[dict]) -> None:
    failed = [
        {"rho": row["rho"], "n_train": row["n_train"], "estimator": row["estimator"], "replicate": row["replicate"], "error": row["error"]}
        for row in rows if row["status"] != "ok"
    ]
    path.write_text(json.dumps(failed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def _draw(summary_path: Path, folder: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    rows = _read_summary(summary_path)
    folder.mkdir(parents=True, exist_ok=True)
    _bias_sd(rows, folder / "fig_e2_bias_sd.png")
    _rmse(rows, folder / "fig_e2_rmse_vs_n.png")
    _curves(rows, folder)
    print("FIGURES", folder)


def _read_summary(path: Path) -> list[dict]:
    text_fields = {"exp", "system", "estimator", "failed_replicates", "status", "sigma2_source"}
    with path.open(encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = []
    for raw in raw_rows:
        row = {key: (value if key in text_fields or value == "" else float(value)) for key, value in raw.items()}
        row["n_train"] = int(row["n_train"])
        rows.append(row)
    return rows


def _styles() -> dict[str, tuple[str, str]]:
    return {
        "oracle": ("C0", "o"), "ols": ("C1", "s"), "ridge": ("C2", "^"), "mc": ("C3", "D"),
        "mc_psd": ("C4", "v"), "iv": ("C5", "P"), "iv2": ("C6", "X"),
    }


def _bias_sd(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt
    chosen = [row for row in rows if row["n_train"] == 30]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    panels = (("bias_A_rel", "A bias"), ("sd_A_rel", "A sd"), ("bias_H_rel", "H bias"), ("sd_H_rel", "H sd"))
    for axis, (key, title) in zip(axes.ravel(), panels, strict=True):
        _lines(axis, chosen, key, "rho")
        axis.set_title(title)
        axis.set_xlabel("rho (linear, includes 0)")
        axis.set_ylabel(key)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _rmse(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt
    chosen = [row for row in rows if abs(row["rho"] - 0.1) < 1e-12]
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for method, (color, marker) in _styles().items():
        subset = sorted((row for row in chosen if row["estimator"] == method), key=lambda item: item["n_train"])
        xs, ys = [], []
        for row in subset:
            value = row["err_O_rel_median"]
            if np.isfinite(value) and value > 0.0:
                xs.append(row["n_train"])
                ys.append(value)
            else:
                print("RMSE_NOT_POSITIVE", method, row["n_train"], value)
        axis.plot(xs, ys, color=color, marker=marker, label=method)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("n_train")
    axis.set_ylabel("median err_O_rel")
    axis.set_title("rho=0.1; nonpositive medians are printed, not drawn as positive")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _curves(rows: list[dict], folder: Path) -> None:
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
    _facet(axes, rows, "specrad_err_median", "specrad_err", False, True)
    figure.suptitle("median and quartile interval. rho is linear and includes 0")
    figure.tight_layout()
    figure.savefig(folder / "fig_e2_specrad.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    _facet(axes, rows, "ep_violation_median", "ep_violation", True, False)
    figure.suptitle("log vertical axis; nonpositive medians are printed and omitted")
    figure.tight_layout()
    figure.savefig(folder / "fig_e2_ep.png", dpi=160)
    plt.close(figure)
    _rollout(rows, folder / "fig_e2_rollout.png")


def _facet(axes, rows, key: str, label: str, logarithmic: bool, quartiles: bool) -> None:
    for axis, n_train in zip(axes, (10, 30, 100), strict=True):
        _lines(axis, [row for row in rows if row["n_train"] == n_train], key, "rho", logarithmic, quartiles)
        axis.set_title(f"n_train={n_train}")
        axis.set_xlabel("rho (linear, includes 0)")
        axis.set_ylabel(label)


def _lines(axis, rows, key: str, x_key: str, logarithmic: bool = False, quartiles: bool = False) -> None:
    for method, (color, marker) in _styles().items():
        subset = sorted((row for row in rows if row["estimator"] == method), key=lambda item: item[x_key])
        xs, ys, low, high = [], [], [], []
        for row in subset:
            value = row[key]
            if not _plottable(value, logarithmic):
                print("OMITTED", key, method, row["rho"], row["n_train"], value)
                continue
            xs.append(row[x_key])
            ys.append(value)
            if quartiles:
                low.append(max(0.0, value - row[key.replace("_median", "_q25")]))
                high.append(max(0.0, row[key.replace("_median", "_q75")] - value))
        if quartiles and xs:
            axis.errorbar(xs, ys, yerr=[low, high], color=color, fmt=marker, capsize=2, label=method)
        else:
            axis.plot(xs, ys, color=color, marker=marker, label=method)
    if logarithmic:
        axis.set_yscale("log")
    axis.legend(fontsize=7)


def _plottable(value: float, logarithmic: bool) -> bool:
    if not np.isfinite(value):
        return False
    return not logarithmic or value > 0.0


def _rollout(rows, path: Path) -> None:
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for column, n_train in enumerate((10, 30, 100)):
        chosen = [row for row in rows if row["n_train"] == n_train]
        _finite_or_inf(axes[0, column], chosen, "rollout_med_median")
        _lines(axes[1, column], chosen, "diverged_frac_median", "rho")
        axes[0, column].set_title(f"n_train={n_train}")
        axes[1, column].set_xlabel("rho (linear, includes 0)")
    axes[0, 0].set_ylabel("rollout_med")
    axes[1, 0].set_ylabel("diverged_frac")
    figure.suptitle("inf medians are marked and are not drawn as finite numbers")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _finite_or_inf(axis, rows, key: str) -> None:
    for method, (color, marker) in _styles().items():
        subset = sorted((row for row in rows if row["estimator"] == method), key=lambda item: item["rho"])
        xs, ys = [], []
        for row in subset:
            value = row[key]
            if np.isfinite(value):
                xs.append(row["rho"])
                ys.append(value)
            else:
                axis.annotate("inf", (row["rho"], 0.92), xycoords=("data", "axes fraction"), color=color, fontsize=7, ha="center")
                print("ROLLOUT_INF", method, row["rho"], row["n_train"], value)
        axis.plot(xs, ys, color=color, marker=marker, label=method)
    axis.legend(fontsize=7)


if __name__ == "__main__":
    main()
