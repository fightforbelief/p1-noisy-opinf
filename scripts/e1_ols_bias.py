"""运行 E1。先导和正式结果分开放。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONUTF8", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from noisyopinf.e1 import (
    PILOT,
    expand_cases,
    extrapolate_seconds,
    fit_case,
    fit_replicate,
    limit_bundle,
    load_config,
    load_s1_train,
    nested_prefix,
    operator_key,
    refuse_existing,
    run_record,
    sigma2_of,
    summarize_case,
    training_rms,
    true_operator,
)
from noisyopinf.io_utils import read_operators, runtime_facts, write_meta, write_operators, write_runs
from noisyopinf.systems.config import project_root

SUMMARY_FIELDS = (
    "exp", "system", "rho", "rho_index", "n_train", "r", "replicates", "sigma2", "estimator",
    "n_failed", "failed_replicates", "theory_gram_cond", "theory_finite", "theory_match_rel",
    "seed_pattern", "status",
)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "pilot":
        run_pilot()
    elif mode == "run":
        run_grid()
    elif mode == "figure":
        redraw_figure()
    elif mode == "rebuild-summary":
        rebuild_summary()
    else:
        raise SystemExit("usage: e1_ols_bias.py pilot|run|figure|rebuild-summary")


def run_pilot() -> None:
    """rho=0.05、n_train=30、前 10 次重复。不写入 results/e1。"""
    out = project_root() / "results" / "e1_pilot"
    refuse_existing(out)
    started = time.perf_counter()
    config = load_config()
    trajectories, peaks = load_s1_train(100)
    truth = true_operator()
    case = _pilot_case(config)
    fit_started = time.perf_counter()
    summary, rows, results, limit, diag = _execute(trajectories, truth, case, PILOT["replicates"])
    fit_elapsed = time.perf_counter() - fit_started
    problems = _pilot_problems(trajectories, truth, summary, rows, results, limit, diag)
    estimate = extrapolate_seconds(fit_elapsed, PILOT["replicates"])
    payload = {
        "pass": not problems, "problems": problems, "estimated_seconds": estimate,
        "pilot_fit_seconds": fit_elapsed, "elapsed_seconds": time.perf_counter() - started,
        "max_peak": max(peaks), "summary": summary,
    }
    arrays = _arrays(truth, [(case, results, limit)])
    _write_bundle(out, [summary], rows, arrays, payload, config)
    problems.extend(_saved_problems(out, rows, results, case))
    if problems:
        payload["pass"] = False
        payload["problems"] = problems
        write_meta(out / "meta.json", _meta(payload, config), overwrite=True)
    print("PILOT_PASS" if not problems else "PILOT_FAIL")
    print("ESTIMATE_SECONDS", estimate)
    if problems:
        raise SystemExit(1)


def run_grid() -> None:
    """12 个配置、每个 200 次。先导未通过或估算超过 4 小时时不运行。"""
    out = project_root() / "results" / "e1"
    refuse_existing(out)
    pilot = json.loads((project_root() / "results" / "e1_pilot" / "meta.json").read_text(encoding="utf-8"))
    if not pilot.get("pass") or float(pilot["estimated_seconds"]) > 4.0 * 3600.0:
        raise RuntimeError("pilot gate is not open")
    started = time.perf_counter()
    config = load_config()
    trajectories, _ = load_s1_train(100)
    truth = true_operator()
    summaries, rows, bundles = [], [], []
    for case in expand_cases(config):
        summary, case_rows, results, limit, _ = _execute(trajectories, truth, case, case["replicates"])
        summaries.append(summary)
        rows.extend(case_rows)
        bundles.append((case, results, limit))
    meta = {"pass": all(item["status"] == "ok" for item in summaries), "elapsed_seconds": time.perf_counter() - started}
    _write_bundle(out, summaries, rows, _arrays(truth, bundles), meta, config)
    _write_figure(summaries, project_root() / "figures" / "fig_e1_bias_theory_vs_sim.png")
    print("E1_DONE", meta["elapsed_seconds"], meta["pass"])


def _pilot_case(config: dict) -> dict:
    found = [case for case in expand_cases(config) if case["rho"] == PILOT["rho"] and case["n_train"] == PILOT["n_train"]]
    if len(found) != 1:
        raise RuntimeError("pilot case is missing from the E1 grid")
    return found[0]


def _execute(trajectories, truth, case: dict, replicates: int):
    clean = nested_prefix(trajectories, case["n_train"])
    active = dict(case)
    active["sigma2"] = sigma2_of(clean, case["rho"])
    limit, diag = limit_bundle(clean, active["sigma2"])
    results = fit_case(clean, float(np.sqrt(active["sigma2"])), case["rho_index"], replicates)
    stack = np.stack([item["O"] for item in results], axis=0)
    summary = summarize_case(active, stack, limit, truth, diag)
    rows = [run_record(active, item, truth, training_rms(clean)) for item in results]
    return summary, rows, results, limit, diag


def _pilot_problems(trajectories, truth, summary, rows, results, limit, diag) -> list[str]:
    clean = nested_prefix(trajectories, 30)
    problems = []
    if any(item is not origin for item, origin in zip(nested_prefix(trajectories, 10), trajectories[:10], strict=True)):
        problems.append("training prefix is not nested")
    if any(item is not origin for item, origin in zip(clean, trajectories[:30], strict=True)):
        problems.append("n_train=30 is not a prefix")
    if truth.shape != (45, 8) or not np.isfinite(truth).all():
        problems.append("O_true is not a finite (45, 8) operator")
    if limit.shape != truth.shape or not np.isfinite(limit).all():
        problems.append("O_lim is not finite")
    if any(not item["finite"] for item in results):
        problems.append("a pilot fit is non-finite")
    if any(item["n_pairs"] != 30 * (clean[0].shape[0] - 1) for item in results):
        problems.append("pairs cross trajectories or drop steps")
    if not (diag["matches_ols_limit_rel"] < 1e-8):
        problems.append("diagnostic Gram does not match ols_limit")
    if len(rows) != PILOT["replicates"] or summary["n_failed"] != 0:
        problems.append("pilot rows or failures are unexpected")
    if abs(summary["sigma2"] - sigma2_of(clean, 0.05)) > 0.0:
        problems.append("sigma2 changed inside the case")
    again = fit_replicate(clean, float(np.sqrt(summary["sigma2"])), summary["rho_index"], 0)
    if not np.allclose(again["O"], results[0]["O"]):
        problems.append("replicate 0 is not reproduced by its seed")
    mean = np.mean(np.stack([item["O"] for item in results], axis=0), axis=0)
    denom = float(np.linalg.norm(limit - truth))
    if denom == 0.0:
        problems.append("theory bias denominator is zero")
    else:
        manual = float(np.linalg.norm(mean - limit) / denom)
        if abs(manual - summary["O_theory_gap_rel"]) > 1e-12:
            problems.append("summary gap does not match the manual norm")
    return problems


def _saved_problems(out: Path, rows, results, case: dict) -> list[str]:
    from noisyopinf.io_utils import read_operators, read_runs
    problems = []
    loaded = read_runs(out / "runs.csv")
    if len(loaded) != len(rows) or [row["replicate"] for row in loaded] != [row["replicate"] for row in rows]:
        problems.append("runs.csv did not round-trip")
    saved = read_operators(out / "operators.npz")
    key = operator_key("rep", case["rho_index"], case["n_train"], results[0]["replicate"])
    if key not in saved or not np.allclose(saved[key], results[0]["O"]):
        problems.append("operator archive does not match the fit")
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    if "git_head" not in meta or "estimated_seconds" not in meta:
        problems.append("pilot metadata is incomplete")
    return problems


def _arrays(truth, bundles) -> dict[str, np.ndarray]:
    arrays = {"true": truth}
    for case, results, limit in bundles:
        arrays[operator_key("lim", case["rho_index"], case["n_train"])] = limit
        for item in results:
            key = operator_key("rep", case["rho_index"], case["n_train"], item["replicate"])
            arrays[key] = item["O"]
    return arrays


def _meta(meta: dict, config: dict) -> dict:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root(), text=True).strip()
    payload = runtime_facts()
    payload.update(meta)
    payload.update({"git_head": head, "config": config, "error_bar": "delta-method SE of the Frobenius bias norm under a diagonal covariance"})
    return payload


def _write_bundle(out: Path, summaries: list[dict], rows: list[dict], arrays: dict, meta: dict, config: dict) -> None:
    refuse_existing(out)
    out.mkdir(parents=True)
    payload = _meta(meta, config)
    write_meta(out / "meta.json", payload)
    write_runs(out / "runs.csv", rows)
    _write_summary(out / "summary.csv", summaries)
    write_operators(out / "operators.npz", arrays)


def rebuild_summary() -> None:
    """用已归档算子重写 summary.csv，不重新拟合。"""
    config = load_config()
    trajectories, _ = load_s1_train(100)
    truth = true_operator()
    _replace_summary(project_root() / "results" / "e1", expand_cases(config), trajectories, truth, None)
    _replace_summary(project_root() / "results" / "e1_pilot", [_pilot_case(config)], trajectories, truth, PILOT["replicates"])


def _replace_summary(out: Path, cases: list[dict], trajectories, truth, count: int | None) -> None:
    saved = read_operators(out / "operators.npz")
    if not np.allclose(saved["true"], truth):
        raise RuntimeError("archived truth does not match the current operator")
    rows = [_summary_from_archive(trajectories, truth, saved, case, count) for case in cases]
    _write_summary(out / "summary.csv", rows)
    print("SUMMARY_REBUILT", out.name, len(rows))


def _summary_from_archive(trajectories, truth, saved, case: dict, count: int | None) -> dict:
    clean = nested_prefix(trajectories, case["n_train"])
    sigma2 = sigma2_of(clean, case["rho"])
    limit, diag = limit_bundle(clean, sigma2)
    archived = saved[operator_key("lim", case["rho_index"], case["n_train"])]
    if not np.allclose(limit, archived, rtol=1e-8, atol=1e-8):
        raise RuntimeError("archived limit does not match ols_limit")
    replicates = case["replicates"] if count is None else count
    stack = np.stack([saved[operator_key("rep", case["rho_index"], case["n_train"], item)] for item in range(replicates)])
    active = dict(case)
    active["sigma2"] = sigma2
    return summarize_case(active, stack, limit, truth, diag)


def _write_summary(path: Path, rows: list[dict]) -> None:
    import csv
    from io import StringIO
    fields = list(SUMMARY_FIELDS)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_cell(row.get(field)) for field in fields})
    path.write_text(buffer.getvalue(), encoding="utf-8", newline="\n")


def _csv_cell(value) -> str:
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


def redraw_figure() -> None:
    """只根据已有 summary.csv 重画图，不重新拟合。"""
    source = project_root() / "results" / "e1" / "summary.csv"
    _write_figure(_read_summary(source), project_root() / "figures" / "fig_e1_bias_theory_vs_sim.png")


def _summary_number(value: str):
    if value == "True":
        return True
    if value == "False":
        return False
    return float(value)


def _read_summary(path: Path) -> list[dict]:
    import csv
    text_fields = {"exp", "system", "estimator", "failed_replicates", "status", "seed_pattern"}
    with path.open(encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = []
    for raw in raw_rows:
        row = {}
        for key, value in raw.items():
            row[key] = value if key in text_fields or value == "" else _summary_number(value)
        row["n_train"] = int(row["n_train"])
        rows.append(row)
    return rows


def _write_figure(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = {10: ("o", "C0"), 30: ("s", "C1"), 100: ("^", "C2")}
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    for axis, block in zip(axes, ("A", "H"), strict=True):
        for n_train, (marker, color) in styles.items():
            chosen = [row for row in rows if int(row["n_train"]) == n_train]
            chosen.sort(key=lambda item: float(item["rho"]))
            rho = [float(row["rho"]) for row in chosen]
            axis.plot(rho, [row[f"{block}_theory_bias_norm"] for row in chosen], color=color, label=f"theory n={n_train}")
            axis.errorbar(
                rho, [row[f"{block}_sim_bias_norm"] for row in chosen],
                yerr=[2.0 * row[f"{block}_bias_se_delta"] for row in chosen],
                fmt=marker, color=color, capsize=3, label=f"MC n={n_train}",
            )
        axis.set_xscale("log")
        axis.set_xlabel("rho")
        axis.set_ylabel("Frobenius bias norm")
        axis.set_title(block)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
