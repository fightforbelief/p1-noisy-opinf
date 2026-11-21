"""Post hoc block diagnostics for the sealed Lorenz-96 grid. Run from the project root."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "evidence" / "crossover"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(states, output):
    spec = json.loads((HERE / "crossover_protocol.json").read_text())
    local_seal = states / "seal.json"
    meta = json.loads(local_seal.read_text())
    reference = json.loads((ARCHIVE / "seal.json").read_text())
    assert all(meta[key] == reference[key] for key in ("protocol", "runner", "base"))
    scores = states / "scores.csv"
    manifest = json.loads((ARCHIVE / "score_manifest.json").read_text())
    assert digest(ARCHIVE / "scores.csv") == manifest["scores"]
    if digest(scores) != manifest["scores"]:
        from check_replay import compare
        compare(ARCHIVE, states, "scores.csv", ["block", "refinement", "rank", "lags", "h", "tau"])
    anchors = np.arange(spec["first_anchor"], spec["last_anchor"]+1, spec["stride"])
    offsets = {0, *spec["tau_ticks"], *(-i*h for i in spec["lags"] for h in spec["h_ticks"])}
    times = np.unique(anchors[:, None] + sorted(offsets))
    with scores.open(newline="") as f:
        selected = [row for row in csv.DictReader(f) if row["refinement"] == "1" and
                    row["lags"] == "2" and max(float(row["h"]), float(row["tau"])) <= .005]
    blocks = []
    for b in range(spec["blocks"]):
        filename = f"b{b}r1_states.npz"
        source = states / filename
        assert digest(source) == meta["inputs"][filename]
        with np.load(source) as data:
            q = data["q"]
        rows = [row for row in selected if int(row["block"]) == b]
        assert len(rows) == 18 and all(row["valid"] == "1" for row in rows)
        design, cosine, shares = [], [], []
        for row in rows:
            h, tau, norm, predicted, cos = (float(row[k]) for k in
                                             ("h", "tau", "norm", "predicted", "cosine"))
            ratio = h/tau
            pure = predicted / ((1 + 3*ratio)*norm)
            design.append([pure, pure*ratio])
            cosine.append(cos)
            rank = int(row["rank"])
            current = q[:, np.searchsorted(times, anchors), :rank]
            future = q[:, np.searchsorted(times, anchors + round(tau/spec["tick"])), :rank]
            shares.append(norm / np.linalg.norm(future-current))
        blocks.append((np.array(design), np.array(cosine), np.array(shares)))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["block", "cells", "intercept", "mixed_coefficient", "calibrated_scale",
                         "calibrated_median_error", "joint_median_error", "median_increment_share",
                         "min_increment_share", "max_increment_share"])
        for i, (design, cos, shares) in enumerate(blocks):
            coefficients = np.linalg.lstsq(design, cos, rcond=None)[0]
            training = [case for b, case in enumerate(blocks) if b != i]
            x = np.concatenate([case[0][:, 0] for case in training])
            y = np.concatenate([case[1] for case in training])
            scale = (x @ y) / (x @ x)

            def median_error(amplitude):
                return np.median(np.sqrt((amplitude-cos)**2 + np.maximum(0, 1-cos**2)))

            writer.writerow([i, len(cos), *coefficients, scale, median_error(scale*design[:, 0]),
                             median_error(design @ [1, 3]), np.median(shares), min(shares), max(shares)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("states", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summarize(args.states, args.output)
