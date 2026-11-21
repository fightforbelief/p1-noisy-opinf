"""Observed two-lag displacement in distinct segments of measured vertical force."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

HERE = Path(__file__).resolve().parent
SPEC = HERE/"force_protocol.json"
PLAN = json.loads(SPEC.read_text())
ARMS = ("raw","lowpass6")
WINDOWS = (3,4,5)
SUBJECTS = [f"GaCo{i:02d}" for i in range(1,13)]
SOS = butter(4, 6, fs=100, output="sos")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observations(record, third, arm):
    part = np.array_split(record, 3)[third]
    times = part[:, 0]
    assert np.all(np.isfinite(part))
    dt = np.diff(times)
    cuts = np.flatnonzero((dt < .008-1e-9) | (dt > .012+1e-9))+1
    histories, loads = [], []
    for run in np.split(part, cuts):
        if len(run) < 200:
            continue
        t = run[:,0]-run[0,0]
        grid = np.arange(int(np.floor(t[-1]*100+1e-9))+1)/100
        left = np.interp(grid,t,run[:,1])
        bilateral = left + np.interp(grid,t,run[:,2])
        if arm == "lowpass6":
            left = sosfiltfilt(SOS,left)
        left = left[100:-100]
        if len(left) <= 24:
            continue
        histories.append(np.lib.stride_tricks.sliding_window_view(left,25))
        loads.append(bilateral[112:-112])
    if not histories:
        return np.empty((0,25)), np.empty(0)
    return np.vstack(histories), np.concatenate(loads)


def coefficient(history, width):
    offsets = np.arange(-width,width+1)
    t = offsets/100
    weights = np.linalg.pinv(np.polynomial.polynomial.polyvander(t,4))
    local = history[:,12+offsets]
    velocity = local@weights[1]
    acceleration = local@(2*weights[2])
    x = history[:,12]
    z = np.column_stack([np.ones(len(x)),x-x.mean(),velocity])
    beta, _, rank, _ = np.linalg.lstsq(z,acceleration,rcond=None)
    if rank != z.shape[1]:
        raise ValueError("deficient derivative design")
    g = acceleration-z@beta
    return float(g@g/np.sum((x-x.mean())**2))


def fit(history, h, tau):
    x, y = history[:,12], history[:,12+tau//10]
    z = np.column_stack([np.ones(len(x)),x])
    w = np.column_stack([z[:,0],history[:,12-h//10],history[:,12-2*h//10]])
    q, _ = np.linalg.qr(w,mode="reduced")
    ols = np.linalg.lstsq(z,y,rcond=None)[0]
    iv, _, rank, _ = np.linalg.lstsq(q.T@z,q.T@y,rcond=None)
    assert rank == 2 and np.linalg.matrix_rank(w) == 3
    d = z@(iv-ols)
    np.testing.assert_allclose(d,(x-x.mean())*(iv[1]-ols[1]), rtol=1e-9, atol=1e-9)
    return z,y,ols,iv,d


def calibrate(data, output):
    output.mkdir(parents=True,exist_ok=False)
    records = {}
    with np.load(data) as archive:
        assert set(archive.files)==set(SUBJECTS)
        for subject in SUBJECTS:
            records[subject] = {}
            for arm in ARMS:
                history,_ = observations(archive[subject],0,arm)
                records[subject][arm] = {str(w):coefficient(history,w) for w in WINDOWS} if len(history)>=1500 else {}
    seal = dict(protocol=digest(SPEC), runner=digest(Path(__file__)), data=digest(data), coefficients=records)
    (output/"seal.json").write_text(json.dumps(seal,indent=2)+"\n",encoding="utf8")


def score(data,output):
    seal = json.loads((output/"seal.json").read_text())
    assert seal["protocol"]==digest(SPEC) and seal["runner"]==digest(Path(__file__)) and seal["data"]==digest(data)
    rows = []
    with np.load(data) as archive:
        for subject in SUBJECTS:
            for arm in ARMS:
                a = seal["coefficients"][subject][arm]
                train = {}
                for third in range(3):
                    history,bilateral = observations(archive[subject],third,arm)
                    for h in PLAN["h_ms"]:
                        for tau in PLAN["tau_ms"]:
                            row = dict(subject=subject,arm=arm,third=third+1,h=h,tau=tau,anchors=len(history),valid=0,passed=0)
                            rows.append(row)
                            if not a or len(history)<1500:
                                continue
                            z,y,ols,iv,d = fit(history,h,tau)
                            if not third:
                                train[h,tau] = ols,iv
                            base,lagged = train[h,tau]
                            rms = lambda x: np.sqrt(np.mean(x*x))
                            gap = rms(d)
                            inc = rms(y-z[:,1])
                            if min(gap,inc,bilateral.mean())<=0:
                                continue
                            ksep = (h/1000)**2*(tau/1000)**2/2
                            joint = ksep*(1+3*h/tau)
                            centered = z[:,1]-z[:,1].mean()
                            row.update(valid=1,gap_n=gap,bilateral_n=bilateral.mean(),increment_n=inc,
                                       load_share=gap/bilateral.mean(),increment_share=gap/inc,
                                       slope_change=iv[1]-ols[1],predicted_n=rms(joint*a["3"]*centered),
                                       separable_error=rms(-ksep*a["3"]*centered-d)/gap,
                                       forecast_ols_n=rms(z@base-y),forecast_iv_n=rms(z@lagged-y))
                            for w in WINDOWS:
                                row[f"error_{10*w}"] = rms(-joint*a[str(w)]*centered-d)/gap
                            row["passed"] = int(row["load_share"]>=.01 and row["increment_share"]>=.05 and row["error_30"]<=.25)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (output/"scores.csv").open("x",newline="",encoding="utf8") as f:
        writer = csv.DictWriter(f,fieldnames=columns,lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["calibrate","score"])
    p.add_argument("data",type=Path)
    p.add_argument("output",type=Path)
    args = p.parse_args()
    (calibrate if args.stage=="calibrate" else score)(args.data,args.output)
    print(args.stage,args.output,flush=True)
