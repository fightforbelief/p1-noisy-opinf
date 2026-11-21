"""Prospective PDE derivative predictions. Run 'seal' and then 'score' on Windows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

from study import HERE, ROOT, Parallel, delayed, features, instrument_basis, np, solve_triangular
from noisyopinf.pod import pod_basis
from noisyopinf.systems.config import load_system
from noisyopinf.systems.spectral import _coefficients, _step, dealias_mask, sine_initial, wavenumbers

SPEC = HERE / "derivative_protocol.json"
PLAN = json.loads(SPEC.read_text())
ANCHORS = np.arange(PLAN["anchor_first"], PLAN["anchor_last"]+1, PLAN["anchor_stride"])
OFFSETS = sorted({0, PLAN["fixed_horizon_ticks"], *PLAN["lag_ticks"],
                  *(-m*h for m in PLAN["lags"] for h in PLAN["lag_ticks"])})
TICKS = np.unique(ANCHORS[:, None]+OFFSETS)
NORM = np.linalg.norm


def parameters(system):
    if system == "burgers":
        cfg = load_system("s2")
        return cfg["N"], cfg["nu"], cfg["amplitude"], 2
    cfg = json.loads((HERE/"protocol.json").read_text())["allen_cahn"]
    return cfg["n"], cfg["nu"], 0.5, 4


def nonlinear(u, system, k):
    if system == "burgers":
        return -0.5j*k*dealias_mask(u.shape[-1])*np.fft.rfft(u*u)
    return 3*np.fft.rfft(u-u**3)


def time_derivatives(u, system):
    """Semidiscrete rhs and its directional derivative times the rhs at each anchor."""
    n, nu, _, _ = parameters(system)
    k = wavenumbers(n)
    linear = -nu*k**2
    v = np.fft.rfft(u)
    du = np.fft.irfft(linear*v + nonlinear(u, system, k), n=n)
    if system == "burgers":
        jac = -1j*k*dealias_mask(n)*np.fft.rfft(u*du)
    else:
        jac = 3*np.fft.rfft((1-3*u*u)*du)
    ddu = np.fft.irfft(linear*np.fft.rfft(du)+jac, n=n)
    return du, ddu


def sample(system, index, refine):
    n, nu, amp, seed = parameters(system)
    k = wavenumbers(n)
    sub = PLAN["step_per_tick"]*refine
    coeff = _coefficients(-nu*k*k, PLAN["tick"]/sub)
    v = np.fft.rfft(sine_initial(n, seed, PLAN["split"], index, amp))
    def rhs(spectrum):
        return nonlinear(np.fft.irfft(spectrum, n=n), system, k)
    fields = np.empty((len(TICKS), n))
    previous = 0
    for j, tick in enumerate(TICKS):
        for _ in range((tick-previous)*sub):
            v = _step(v, coeff, rhs)
        fields[j] = np.fft.irfft(v, n=n)
        previous = tick
    return fields


def feature_jet(q, dq, ddq):
    i, j = np.triu_indices(q.shape[1])
    zero = np.zeros(len(q))
    first = np.column_stack([dq, dq[:, i]*q[:, j]+q[:, i]*dq[:, j], zero])
    second = np.column_stack([ddq, ddq[:, i]*q[:, j]+2*dq[:, i]*dq[:, j]+q[:, i]*ddq[:, j], zero])
    return first, second


def predictions(q, dq, ddq, y_fixed, m):
    z = features(q)
    u, r = np.linalg.qr(z, mode="reduced")
    first, second = feature_jet(q, dq, ddq)
    jet = z if m == 1 else np.column_stack([z, first[:, :-1]])
    j = instrument_basis(jet)
    derivative = first if m == 1 else second
    normal = derivative-j@(j.T@derivative)
    ell = solve_triangular(r.T, normal.T, lower=True).T
    inc = y_fixed-q
    residual = inc-u@(u.T@inc)
    fixed = -ell.T@residual
    coupled = -m*ell.T@normal[:, :q.shape[1]]
    # Coefficient units keep predictions invariant to the sign conventions of a later QR.
    return dict(fixed=solve_triangular(r, fixed), coupled=solve_triangular(r, coupled),
                angular=NORM(ell, 2), jet_rank=j.shape[1], jet_cols=jet.shape[1])


def names():
    for system in PLAN["systems"]:
        for block in range(PLAN["blocks"]):
            yield system, block, 1
        yield system, 0, 2


def prefix(system, block, refinement):
    return f"{system}_{block}_dt{refinement}"


def offset_states(q, offset):
    return q[:, np.searchsorted(TICKS, ANCHORS+offset), :].reshape(-1, q.shape[-1])


def seal_block(path, system, block, refinement):
    start = PLAN["first_index"] + block*PLAN["trajectories"]
    fields = np.array(Parallel(n_jobs=4)(delayed(sample)(system, start+i, refinement)
                                         for i in range(PLAN["trajectories"])))
    anchors = fields[:, np.searchsorted(TICKS, ANCHORS), :]
    basis, _ = pod_basis(list(anchors), max(PLAN["ranks"]))
    q = fields@basis
    # The prediction function receives only anchors, their derivatives and a fixed-horizon response.
    du, ddu = time_derivatives(anchors, system)
    dq = (du@basis).reshape(-1, basis.shape[1])
    ddq = (ddu@basis).reshape(dq.shape)
    stem = prefix(system, block, refinement)
    np.savez(path/(stem+"_observations.npz"), q=q, basis=basis)
    archive = {}
    for rank in PLAN["ranks"]:
        for m in PLAN["lags"]:
            pred = predictions(offset_states(q, 0)[:, :rank], dq[:, :rank], ddq[:, :rank],
                               offset_states(q, PLAN["fixed_horizon_ticks"])[:, :rank], m)
            archive.update({f"r{rank}_m{m}_{key}": val for key,val in pred.items()})
    np.savez(path/(stem+"_prediction.npz"), **archive)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(path):
    path.mkdir(parents=True, exist_ok=False)
    started = time.time()
    for system, block, ref in names():
        seal_block(path, system, block, ref)
        print("SEALED", system, block, ref, flush=True)
    hashes = {f.name: sha(f) for f in sorted(path.glob("*.npz"))}
    meta = dict(inputs=hashes, protocol=sha(SPEC), runner=sha(Path(__file__)),
                started=started, sealed=time.time(), numpy=np.__version__, python=platform.python_version())
    (path/"seal.json").write_text(json.dumps(meta, indent=2), encoding="utf8")


def observed(u, q, y, w):
    """Independent projected fit to increments and numerically stable residual refit."""
    qt = instrument_basis(w)
    coupling = qt.T@u
    inc = y-q
    ols = u.T@inc
    e = inc-u@ols
    # Residualizing twice enforces orthogonality to working precision.
    e -= u@(u.T@e)
    defect = u-qt@coupling
    stable = -np.linalg.solve(coupling.T@coupling, defect.T@(e-qt@(qt.T@e)))
    direct, _, rank, singular = np.linalg.lstsq(coupling, qt.T@inc, rcond=None)
    return stable, direct-ols, NORM(defect, 2)/singular[-1], qt.shape[1], rank


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_block(path, system, block, ref, comparison):
    stem = prefix(system, block, ref)
    with np.load(path/(stem+"_observations.npz")) as a:
        all_q, basis = a["q"], a["basis"]
    rows = []
    with np.load(path/(stem+"_prediction.npz")) as pred:
        for rank in PLAN["ranks"]:
            all_r = all_q[:, :, :rank]
            q = offset_states(all_r, 0)
            z = features(q)
            u, r = np.linalg.qr(z, mode="reduced")
            for ticks in PLAN["lag_ticks"]:
                h = ticks*PLAN["tick"]
                past = []
                for m in PLAN["lags"]:
                    past.append(features(offset_states(all_r, -m*ticks))[:, :-1])
                    w = np.column_stack([*past, np.ones(len(q))])
                    key = f"r{rank}_m{m}_"
                    for arm in ("fixed","coupled"):
                        y = offset_states(all_r, PLAN["fixed_horizon_ticks"] if arm=="fixed" else ticks)
                        actual, direct, angle, wr, cr = observed(u, q, y, w)
                        guess = (r@pred[key+arm])*h**(m if arm=="fixed" else 2*m)
                        size, mag = NORM(actual), NORM(guess)
                        err = NORM(guess-actual)/size if size else np.inf
                        cos = np.vdot(guess, actual)/(size*mag) if size*mag else 0.
                        resolved = size > 1e-13*NORM(y-q)
                        full = (wr==w.shape[1] and cr==z.shape[1] and
                                pred[key+"jet_rank"]==pred[key+"jet_cols"])
                        row = dict(system=system, block=block, refinement=ref, rank=rank, ticks=ticks,
                                   h=h, lags=m, horizon=arm, norm=size, predicted=mag,
                                   cosine=float(cos), error=err, angle=angle,
                                   predicted_angle=float(pred[key+"angular"])*h**m,
                                   direct_error=NORM(actual-direct)/max(size, np.finfo(float).tiny),
                                   resolved=int(resolved), full=int(full),
                                   usable=int(resolved and full and err<=0.25 and cos>=0.95))
                        # Compare actual displacements across time integrators in physical coordinates.
                        if block==0:
                            ck=(system,rank,ticks,m,arm)
                            d=u@actual
                            if ref==1:
                                comparison[ck]=(d, basis[:, :rank])
                            else:
                                old, old_basis=comparison[ck]
                                # Full physical-space difference, using the shared FOM grid.
                                diff=d@basis[:, :rank].T-old@old_basis.T
                                row["refine_error"]=NORM(diff)/NORM(old)
                        rows.append(row)
    return rows


def score(path):
    seal_path = path/"seal.json"
    meta=json.loads(seal_path.read_text())
    if meta["protocol"]!=sha(SPEC) or meta["runner"]!=sha(Path(__file__)):
        raise ValueError("Sealed protocol or runner changed")
    for name, digest in meta["inputs"].items():
        if sha(path/name)!=digest:
            raise ValueError(name)
    rows, comparison = [], {}
    for system, block, ref in names():
        rows.extend(score_block(path, system, block, ref, comparison))
    for row in rows:
        row.setdefault("refine_error","")
    write_csv(path/"scores.csv",rows)
    print("SCORED",len(rows),sha(path/"scores.csv"),"SEAL",sha(seal_path),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["seal","score"])
    parser.add_argument("--output",type=Path,default=ROOT/"results"/"derivative_prediction_v1")
    args=parser.parse_args()
    (seal if args.stage=="seal" else score)(args.output)
