"""Transfer derivative coefficients from January to two observed tide-gauge months."""
import argparse
import calendar
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = HERE / "tide_protocol.json"
PLAN = json.loads(SPEC.read_text())
WINDOWS = (42, 60, 90)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_month(archive, manifest, station, month):
    name = f"{station}_{month}.json"
    raw = archive.extractfile(name).read()
    assert sha(raw) == manifest[name]["sha256"]
    data = json.loads(raw)
    assert data["metadata"]["id"] == station
    start = datetime.strptime(month, "%Y%m")
    n = calendar.monthrange(start.year, start.month)[1] * 240
    heights = np.full(n, np.nan)
    seen = set()
    for row in data["data"]:
        minutes = (datetime.strptime(row["t"], "%Y-%m-%d %H:%M") - start).total_seconds()/60
        assert 0 <= minutes < n*6 and minutes % 6 == 0 and minutes not in seen
        seen.add(minutes)
        if row["q"] == "v" and all(f == "0" for f in row["f"].split(",")) and row["v"]:
            heights[int(minutes/6)] = float(row["v"])
    return heights


def anchors(x, calibration):
    offsets = {0, *(t//6 for t in PLAN["horizon_minutes"]),
               *(-m*h//6 for h in PLAN["lag_minutes"] for m in PLAN["lags"])}
    if calibration:
        offsets.update(range(-15, 16))
    ix = np.arange(-min(offsets), len(x)-max(offsets))
    return ix[np.all(np.isfinite(x[ix[:, None]+sorted(offsets)]), axis=1)]


def energy_coefficients(x, ix, window):
    offsets = np.arange(-window//6, window//6+1)
    hours = offsets/10
    weights = np.linalg.pinv(np.polynomial.polynomial.polyvander(hours, 3))
    local = x[ix[:, None]+offsets]
    velocity = local @ weights[1]
    acceleration = local @ (2*weights[2])
    centered = x[ix] - x[ix].mean()
    z = np.column_stack((np.ones(len(ix)), centered))
    jet = np.column_stack((z, velocity))
    values = []
    for design, derivative in ((z, velocity), (jet, acceleration)):
        beta, _, rank, _ = np.linalg.lstsq(design, derivative, rcond=None)
        if rank != design.shape[1]:
            raise ValueError("rank-deficient calibration jet")
        normal = derivative-design@beta
        values.append(float(normal@normal/(centered@centered)))
    return values


def get_month(archive, manifest, station, month):
    x = read_month(archive, manifest, station, month)
    ix = anchors(x, month == PLAN["months"][0])
    return x, ix


def calibrate(archive_path, output):
    output.mkdir(parents=True, exist_ok=False)
    coefficients = {}
    with tarfile.open(archive_path) as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        for station in PLAN["stations"]:
            x, ix = get_month(archive, manifest, station, PLAN["months"][0])
            coefficients[station] = dict(anchors=len(ix), eligible=bool(len(ix)>=.7*len(x)),
                values={str(w): energy_coefficients(x, ix, w) for w in WINDOWS} if len(ix)>=.7*len(x) else {})
    sealed = dict(protocol=sha(SPEC.read_bytes()), runner=sha(Path(__file__).read_bytes()),
                  archive=sha(archive_path.read_bytes()), stations=coefficients)
    (output/"seal.json").write_text(json.dumps(sealed, indent=2)+"\n", encoding="utf8")


def fitted(x, ix, m, h, tau):
    current, y = x[ix], x[ix+tau//6]
    z = np.column_stack((np.ones(len(ix)), current))
    w = np.column_stack([np.ones(len(ix)), *(x[ix-j*h//6] for j in range(1, m+1))])
    q, _ = np.linalg.qr(w, mode="reduced")
    if np.linalg.matrix_rank(w) != w.shape[1]:
        raise ValueError("rank-deficient instruments")
    ols = np.linalg.lstsq(z, y, rcond=None)[0]
    iv, _, rank, _ = np.linalg.lstsq(q.T@z, q.T@y, rcond=None)
    if rank != 2:
        raise ValueError("unidentified projected regression")
    d = z@(iv-ols)
    np.testing.assert_allclose(d, (iv[1]-ols[1])*(current-current.mean()), atol=2e-12, rtol=1e-10)
    return z, y, ols, iv, d


def factor(m, h, tau, separable=False):
    h, tau = h/60, tau/60
    if m == 1:
        return h*tau
    return h*h*tau*(tau if separable else tau+3*h)/2


def score(archive_path, output):
    seal = json.loads((output/"seal.json").read_text())
    assert seal["archive"] == sha(archive_path.read_bytes()) and seal["protocol"] == sha(SPEC.read_bytes())
    assert seal["runner"] == sha(Path(__file__).read_bytes())
    rows = []
    with tarfile.open(archive_path) as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        for station, calibration in seal["stations"].items():
            train = {}
            for month in PLAN["months"]:
                x, ix = get_month(archive, manifest, station, month)
                for m in PLAN["lags"]:
                    for h in PLAN["lag_minutes"]:
                        for tau in PLAN["horizon_minutes"]:
                            key = (m, h, tau)
                            row = dict(station=station, month=month, lags=m, h=h, tau=tau, anchors=len(ix),
                                       slots=len(x), valid=0, passed=0)
                            rows.append(row)
                            if not calibration["eligible"] or len(ix)<.7*len(x):
                                continue
                            z, y, ols, iv, d = fitted(x, ix, m, h, tau)
                            if month == PLAN["months"][0]:
                                train[key] = ols, iv
                            base, lagged = train[key]
                            centered = z[:,1]-z[:,1].mean()
                            size = np.linalg.norm(d)
                            increment = np.linalg.norm(y-z[:,1])
                            if not size or not increment:
                                continue
                            gap = size / np.sqrt(len(ix))
                            share = size / increment
                            row.update(valid=1, slope_change=iv[1]-ols[1], gap_m=gap, share=share,
                                       increment_m=increment/np.sqrt(len(ix)),
                                       forecast_ols_m=np.sqrt(np.mean((z@base-y)**2)),
                                       forecast_iv_m=np.sqrt(np.mean((z@lagged-y)**2)))
                            for width in WINDOWS:
                                a = calibration["values"][str(width)][m-1]
                                predicted = -a*factor(m,h,tau)*centered
                                row[f"error_{width}"] = np.linalg.norm(predicted-d)/size
                                if width == 60:
                                    row["predicted_m"] = np.linalg.norm(predicted)/np.sqrt(len(ix))
                                    row["separable_error"] = np.linalg.norm(-a*factor(m,h,tau,True)*centered-d)/size
                            row["passed"] = int(gap>=.02 and share>=.05 and row["error_60"]<=.25)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (output/"scores.csv").open("x", newline="", encoding="utf8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["calibrate","score"])
    p.add_argument("archive", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    (calibrate if args.stage == "calibrate" else score)(args.archive, args.output)
    print(args.stage, args.output, flush=True)
