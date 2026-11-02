"""只读核对 E4 的投影口径、SNR 删失和选秩稳定性。不重拟合，不写 results/e4。"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from noisyopinf.metrics import proj_floor
from noisyopinf.protocol import LAMBDAS, larger_lambda_index

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e4"
DEST = ROOT / "reports" / "e4_audit"
RANKS = (4, 6, 8, 10, 12, 16)
EST = ("ols", "ridge", "mc", "iv2")
RHOS = (0.01, 0.02, 0.05, 0.1, 0.2)
RHO_INDEX = {0.01: 1, 0.02: 2, 0.05: 3, 0.1: 4, 0.2: 5}
KAPPAS = (1, 4, 16, 64)
TOL = 1e-8


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def num(value: str) -> float:
    if value in {"", "nan", "None"}:
        return float("nan")
    if value == "inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    return float(value)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def median(values) -> float:
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0 or np.isnan(data).any():
        return float("nan")
    return float(np.median(data))


def lower_central(values: list[int]) -> int:
    ordered = sorted(values)
    count = len(ordered)
    index = count // 2 if count % 2 else count // 2 - 1
    return ordered[index]


def choose(scores: dict[int, float]) -> tuple[int, float, int, int, float]:
    def key(item):
        rank, score = item
        if np.isnan(score):
            return (2, 0.0, rank)
        if not np.isfinite(score):
            return (1, 0.0, rank)
        return (0, float(score), rank)

    ordered = sorted(scores.items(), key=key)
    best = ordered[0][1]
    tied = [rank for rank, score in scores.items() if (np.isposinf(score) and np.isposinf(best)) or score == best]
    return min(tied), float(best), len(tied), int(ordered[1][0]), float(ordered[1][1])


def snr_max(energies: np.ndarray, sigma2: float, kappa: float) -> tuple[int | None, bool]:
    passed = [i + 1 for i, value in enumerate(energies) if np.isfinite(value) and value >= kappa * sigma2]
    if not passed:
        return None, False
    return max(passed), len(passed) == len(energies)


def main() -> None:
    reuse = "--reuse" in sys.argv
    if reuse and (DEST / "projection_floor_audit.csv").exists() and (DEST / "snr_replicate_detail.csv").exists():
        audit_rows = read_csv(DEST / "projection_floor_audit.csv")
        snr_detail = read_csv(DEST / "snr_replicate_detail.csv")
        for row in snr_detail:
            row["rho"] = num(row["rho"])
            row["replicate"] = int(float(row["replicate"]))
            row["kappa"] = int(float(row["kappa"]))
            row["r_snr"] = int(float(row["r_snr"]))
            row["delta_vs_rep_val"] = int(float(row["delta_vs_rep_val"]))
            row["right_censored_at_rank_16"] = int(float(row["right_censored_at_rank_16"]))
        raw = defaultdict(list)
        for row in snr_detail:
            if row["estimator"] == "ols":
                raw[(row["rho"], row["kappa"])].append(row["r_snr"])
        curves = read_csv(OUT / "rank_curves.csv")
        width = 16
        pooled_mismatch = 0
        violations = []
        _finish(curves, audit_rows, snr_detail, raw, width, pooled_mismatch, violations)
        return
    with np.load(ROOT / "data" / "e4_s2_states.npz") as loaded:
        test = np.asarray(loaded["test"], dtype=np.float64)
    curves = read_csv(OUT / "rank_curves.csv")
    sigma_rows = read_csv(OUT / "sigma2_estimation.csv")
    sigma_of = {(num(row["rho"]), int(float(row["replicate"]))): num(row["sigma2_est_trailing"]) for row in sigma_rows}
    summary = read_csv(OUT / "summary.csv")
    summary_of = {(num(row["rho"]), int(float(row["r"])), row["estimator"]): row for row in summary}

    with np.load(OUT / "rollout_details.npz") as details, np.load(OUT / "bases.npz") as bases:
        n = details["rho"].shape[0]
        grouped = defaultdict(list)
        violations = []
        pooled_mismatch = 0
        for index in range(n):
            rho = float(details["rho"][index])
            rank = int(details["r"][index])
            method = EST[int(details["estimator"][index])]
            rep = int(details["replicate"][index])
            traj = int(details["traj"][index])
            grouped[(rho, rank, method, rep)].append((traj, bool(details["diverged"][index]), float(details["err_full"][index]), float(details["proj_floor"][index])))
        floor_rows = []
        per_config = defaultdict(list)
        for (rho, rank, method, rep), items in grouped.items():
            basis = np.asarray(bases[f"v_rho{RHO_INDEX[rho]}_rep{rep:03d}"][:, :rank], dtype=np.float64)
            items = sorted(items, key=lambda item: item[0])
            own = []
            roll = []
            below = 0
            worst = 0.0
            for traj, diverged, err, _stored in items:
                truth = test[traj]
                floor = float(proj_floor(basis, truth))
                own.append(floor)
                value = np.inf if diverged else err
                roll.append(value)
                if np.isfinite(value):
                    gap = floor - value
                    scale = max(1.0, abs(floor), abs(value))
                    if gap > TOL * scale:
                        below += 1
                        worst = max(worst, gap)
                        if len(violations) < 20:
                            violations.append((rho, rank, method, rep, traj, value, floor, gap))
            pooled = float(proj_floor(basis, np.vstack(test)))
            stored = items[0][3]
            if abs(pooled - stored) > TOL * max(1.0, abs(pooled)):
                pooled_mismatch += 1
            record = {
                "rho": rho, "r": rank, "estimator": method, "replicate": rep,
                "pooled_floor": pooled, "stored_pooled_floor": stored,
                "per_traj_floor_median": float(np.median(own)),
                "rollout_median": float(np.median(roll)),
                "n_below_own_floor": below, "max_violation": worst,
            }
            per_config[(rho, rank, method)].append(record)
        audit_rows = []
        for rho in RHOS:
            for method in EST:
                for rank in RANKS:
                    block = per_config[(rho, rank, method)]
                    pooled_med = median(item["pooled_floor"] for item in block)
                    rollout_med = median(item["rollout_median"] for item in block)
                    stored_summary = num(summary_of[(rho, rank, method)]["proj_floor_median"])
                    summary_roll = num(summary_of[(rho, rank, method)]["rollout_med_median"])
                    audit_rows.append({
                        "rho": rho, "estimator": method, "r": rank,
                        "pooled_floor_median": pooled_med,
                        "stored_summary_proj_floor_median": stored_summary,
                        "per_traj_floor_median_of_medians": median(item["per_traj_floor_median"] for item in block),
                        "rollout_median_of_medians": rollout_med,
                        "stored_summary_rollout_median": summary_roll,
                        "rollout_below_pooled_reference": int(np.isfinite(rollout_med) and rollout_med < pooled_med),
                        "n_traj_below_own_floor": int(sum(item["n_below_own_floor"] for item in block)),
                        "max_violation": max(item["max_violation"] for item in block),
                        "pooled_matches_summary": int(abs(pooled_med - stored_summary) <= TOL * max(1.0, abs(pooled_med)) or (np.isinf(pooled_med) and np.isinf(stored_summary))),
                    })
    write_csv(DEST / "projection_floor_audit.csv", audit_rows)

    # SNR
    snr_detail = []
    raw = defaultdict(list)
    with np.load(OUT / "bases.npz") as bases:
        width = int(np.asarray(bases["v_rho1_rep000"]).shape[1])
        for rho in RHOS:
            for rep in range(10):
                energies = np.asarray(bases[f"lam_rho{RHO_INDEX[rho]}_rep{rep:03d}"], dtype=np.float64)
                sigma2 = sigma_of[(rho, rep)]
                for kappa in KAPPAS:
                    chosen, all_pass = snr_max(energies, sigma2, kappa)
                    raw[(rho, kappa)].append(chosen)
                    censored = int(all_pass and chosen == width)
                    for method in EST:
                        snr_detail.append({
                            "rho": rho, "replicate": rep, "kappa": kappa, "estimator": method,
                            "n_modes_stored": width, "r_snr": "" if chosen is None else chosen,
                            "right_censored_at_rank_16": censored,
                            "sigma2": sigma2, "lambda_16": float(energies[-1]),
                            "n_modes_passing": int(np.sum(energies >= kappa * sigma2)),
                        })
    # attach replicate validation ranks
    by_rep = defaultdict(list)
    for row in curves:
        by_rep[(num(row["rho"]), row["estimator"], int(float(row["replicate"])))].append(row)
    picks = {}
    scores_of = {}
    for key, rows in by_rep.items():
        for field, name in (("clean_val_med", "val"), ("noisy_val_med", "noisyval"), ("test_med", "test")):
            score = {int(float(item["r"])): num(item[field]) for item in rows}
            chosen, best, ties, second_rank, second = choose(score)
            picks[(key[0], key[1], key[2], name)] = chosen
            scores_of[(key[0], key[1], key[2], name)] = (score, chosen, best, second_rank, second)
    for row in snr_detail:
        row["r_val_rep"] = picks[(row["rho"], row["estimator"], row["replicate"], "val")]
        row["delta_vs_rep_val"] = int(row["r_snr"]) - row["r_val_rep"]
    write_csv(DEST / "snr_replicate_detail.csv", snr_detail)
    _finish(curves, audit_rows, snr_detail, raw, width, pooled_mismatch, violations)


def _finish(curves, audit_rows, snr_detail, raw, width, pooled_mismatch, violations) -> None:
    picks = {}
    by_rep = defaultdict(list)
    for row in curves:
        by_rep[(num(row["rho"]), row["estimator"], int(float(row["replicate"])))].append(row)
    for key, rows in by_rep.items():
        for field, name in (("clean_val_med", "val"), ("noisy_val_med", "noisyval"), ("test_med", "test")):
            score = {int(float(item["r"])): num(item[field]) for item in rows}
            picks[(key[0], key[1], key[2], name)] = choose(score)[0]
    recon = []
    for rho in RHOS:
        for method in EST:
            curve_scores = {}
            for field, name in (("clean_val_med", "val"), ("noisy_val_med", "noisyval"), ("test_med", "test")):
                curve_scores[name] = {}
                for rank in RANKS:
                    curve_scores[name][rank] = median(num(item[field]) for item in curves if num(item["rho"]) == rho and item["estimator"] == method and int(float(item["r"])) == rank)
            for kappa in KAPPAS:
                deltas = [int(row["delta_vs_rep_val"]) for row in snr_detail if row["rho"] == rho and row["estimator"] == method and int(row["kappa"]) == kappa]
                agg = lower_central(raw[(rho, kappa)])
                r_val = choose(curve_scores["val"])[0]
                censored_reps = sum(int(row["right_censored_at_rank_16"]) for row in snr_detail if row["rho"] == rho and row["estimator"] == method and int(row["kappa"]) == kappa) // 1
                # each estimator repeats the same censor flag; count unique reps
                recon.append({
                    "rho": rho, "estimator": method, "kappa": kappa,
                    "aggregate_r_snr_lower_central": agg,
                    "ordinary_median_r_snr": float(np.median(raw[(rho, kappa)])),
                    "aggregate_r_val": r_val,
                    "aggregate_delta": int(agg - r_val),
                    "mean_abs_rep": float(np.mean(np.abs(deltas))),
                    "median_abs_rep": float(np.median(np.abs(deltas))),
                    "mean_signed_rep": float(np.mean(deltas)),
                    "within_2_reps": int(sum(abs(item) <= 2 for item in deltas)),
                    "n_reps": len(deltas),
                    "n_censored_reps": int(sum(1 for item in raw[(rho, kappa)] if item == 16 and all(int(row["right_censored_at_rank_16"]) for row in snr_detail if row["rho"] == rho and row["replicate"] == raw[(rho, kappa)].index(item) and int(row["kappa"]) == kappa and row["estimator"] == method))),
                })
    # The censored count above is messy if duplicate 16s. Recompute cleanly.
    for row in recon:
        flags = [int(item["right_censored_at_rank_16"]) for item in snr_detail if item["rho"] == row["rho"] and item["estimator"] == row["estimator"] and int(item["kappa"]) == row["kappa"]]
        row["n_censored_reps"] = int(sum(flags))
        row["label"] = "RIGHT_CENSORED_AT_RANK_16" if row["n_censored_reps"] == row["n_reps"] else "OBSERVED_WITHIN_16"
    write_csv(DEST / "snr_reconciliation.csv", recon)

    # stability and gaps
    stability = []
    gaps = []
    for rho in RHOS:
        for method in EST:
            for name, field in (("val", "clean_val_med"), ("noisyval", "noisy_val_med"), ("test", "test_med")):
                chosen_ranks = [picks[(rho, method, rep, name)] for rep in range(10)]
                counts = Counter(chosen_ranks)
                scores = {rank: median(num(item[field]) for item in curves if num(item["rho"]) == rho and item["estimator"] == method and int(float(item["r"])) == rank) for rank in RANKS}
                agg, best, ties, second_rank, second = choose(scores)
                gap = float(second - best) if np.isfinite(best) and np.isfinite(second) else float("nan")
                relative = gap / max(abs(best), 1e-12) if np.isfinite(gap) else float("nan")
                stability.append({
                    "rho": rho, "estimator": method, "criterion": name,
                    "error_curve_rank": agg,
                    "lower_central_of_replicate_ranks": lower_central(chosen_ranks),
                    "arithmetic_median_of_replicate_ranks": float(np.median(chosen_ranks)),
                    "two_statistics_differ": int(agg != lower_central(chosen_ranks)),
                    "n4": counts[4], "n6": counts[6], "n8": counts[8], "n10": counts[10], "n12": counts[12], "n16": counts[16],
                    "n_unique": len(counts), "n_match_error_curve": sum(item == agg for item in chosen_ranks),
                    "unanimous": int(len(counts) == 1),
                    "min_pick": min(chosen_ranks), "max_pick": max(chosen_ranks),
                })
                gaps.append({
                    "rho": rho, "estimator": method, "criterion": name,
                    "error_curve_rank": agg, "second_rank": second_rank,
                    "best_score": best, "second_score": second, "absolute_gap": gap, "relative_gap": relative,
                    "close_exploratory_5pct": int(np.isfinite(relative) and relative < 0.05),
                    "threshold_status": "EXPLORATORY_NOT_PREREGISTERED",
                    "n_tied_exact": ties,
                })
    write_csv(DEST / "rank_stability_final.csv", stability)
    write_csv(DEST / "rank_gap_final.csv", gaps)

    # ridge lambda check
    ridge = read_csv(OUT / "ridge.csv")
    runs = read_csv(OUT / "runs.csv")
    ridge_bad = 0
    groups = defaultdict(list)
    for row in ridge:
        groups[(num(row["rho"]), int(float(row["r"])), int(float(row["replicate"])))].append(row)
    for key, rows in groups.items():
        rows = sorted(rows, key=lambda item: float(item["lam"]))
        chosen = float(LAMBDAS[larger_lambda_index([float(item["score"]) for item in rows])])
        matched = [row for row in runs if num(row["rho"]) == key[0] and int(float(row["r"])) == key[1] and int(float(row["replicate"])) == key[2] and row["estimator"] == "ridge"]
        if len(rows) != 13 or len(matched) != 1 or abs(num(matched[0]["lam"]) - chosen) > 1e-12 * max(1.0, abs(chosen)):
            ridge_bad += 1
    below = sum(int(row["rollout_below_pooled_reference"]) for row in audit_rows)
    own = sum(int(row["n_traj_below_own_floor"]) for row in audit_rows)
    print("POOLED_BELOW", below, "OWN_VIOLATIONS", own, "POOLED_MISMATCH", pooled_mismatch, "RIDGE_BAD", ridge_bad)
    print("BASIS_WIDTH", width)
    print("VIOLATIONS", violations)
    for kappa in KAPPAS:
        block = [row for row in recon if row["kappa"] == kappa]
        print("KAPPA", kappa, "mean_abs", float(np.mean([row["mean_abs_rep"] for row in block])), "within", sum(row["within_2_reps"] for row in block), "/", sum(row["n_reps"] for row in block))
    for rho in (0.05, 0.1, 0.2):
        block = [row for row in recon if row["kappa"] == 64 and row["rho"] == rho]
        print("K64", rho, float(np.mean([row["mean_abs_rep"] for row in block])))
    for name in ("val", "noisyval"):
        rows = [row for row in stability if row["criterion"] == name]
        print(name, "unanimous", sum(int(row["unanimous"]) for row in rows), "differ", sum(int(row["two_statistics_differ"]) for row in rows))
    for name in ("val", "noisyval", "test"):
        rows = [row for row in gaps if row["criterion"] == name]
        print("close", name, sum(int(row["close_exploratory_5pct"]) for row in rows))


if __name__ == "__main__":
    main()
