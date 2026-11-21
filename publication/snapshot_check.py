"""Offline derivative prediction from three auxiliary state measurements per anchor."""
import argparse
import json
from pathlib import Path

from derivative_study import (ANCHORS, PLAN, ROOT, TICKS, Parallel, delayed, np, parameters,
                              sample, nonlinear, _coefficients, _step, wavenumbers,
                              offset_states, predictions, features, observed, NORM, write_csv,
                              prefix, sha)

# Forward-only auxiliary samples at 0.00075, 0.00150, 0.00225. None is a fit row
# from the prospective instrument or response arrays, including adjacent anchors.
DELTA_TICKS=3
DELTA=PLAN["tick"]*DELTA_TICKS


def forward_measurements(system,index,basis):
    initial=sample(system,index,1)[np.searchsorted(TICKS,ANCHORS)]
    n,nu,_,_=parameters(system)
    k=wavenumbers(n)
    sub=PLAN["step_per_tick"]
    coeff=_coefficients(-nu*k*k,PLAN["tick"]/sub)
    spectrum=np.fft.rfft(initial)
    def rhs(v):
        return nonlinear(np.fft.irfft(v,n=n),system,k)
    out=[initial@basis]
    for _ in range(3):
        for _ in range(DELTA_TICKS*sub):
            spectrum=_step(spectrum,coeff,rhs)
        out.append(np.fft.irfft(spectrum,n=n)@basis)
    return np.stack(out)


def differentiate(frames, spacing):
    q0,q1,q2,q3=frames
    # Subtract the anchor before differencing to reduce cancellation of the state offset.
    d1,d2,d3=q1-q0,q2-q0,q3-q0
    return (18*d1-9*d2+2*d3)/(6*spacing),(-5*d1+4*d2-d3)/spacing**2


def estimate(system,block,source):
    stem=prefix(system,block,1)
    with np.load(source/(stem+"_observations.npz")) as f:
        states,basis=f["q"],f["basis"]
    start=PLAN["first_index"]+block*PLAN["trajectories"]
    frames=Parallel(n_jobs=4)(delayed(forward_measurements)(system,start+i,basis)
                              for i in range(PLAN["trajectories"]))
    frames=np.stack(frames,axis=1).reshape(4,-1,basis.shape[1])
    np.testing.assert_allclose(frames[0],offset_states(states,0),atol=1e-12,rtol=1e-12)
    dq,ddq=differentiate(frames,DELTA)
    rows=[]
    for rank in PLAN["ranks"]:
        q=frames[0,:,:rank]
        u,r=np.linalg.qr(features(q),mode="reduced")
        s=states[:,:,:rank]
        y_fixed=offset_states(s,PLAN["fixed_horizon_ticks"])
        for m in PLAN["lags"]:
            pred=predictions(q,dq[:,:rank],ddq[:,:rank],y_fixed,m)
            for tick in PLAN["lag_ticks"]:
                h=tick*PLAN["tick"]
                w=np.column_stack([*(features(offset_states(s,-a*tick))[:,:-1]
                                     for a in range(1,m+1)),np.ones(len(q))])
                for arm in ("fixed","coupled"):
                    y=y_fixed if arm=="fixed" else offset_states(s,tick)
                    actual,_,_,wr,cr=observed(u,q,y,w)
                    guess=(r@pred[arm])*h**(m if arm=="fixed" else 2*m)
                    norm=NORM(actual)
                    error=NORM(guess-actual)/norm
                    cosine=np.vdot(guess,actual)/(NORM(guess)*norm)
                    ok=(norm>1e-13*NORM(y-q) and wr==w.shape[1] and cr==r.shape[1]
                        and pred["jet_rank"]==pred["jet_cols"])
                    rows.append(dict(system=system,block=block,rank=rank,lags=m,ticks=tick,
                                     horizon=arm,norm=norm,error=error,cosine=cosine,
                                     predicted=NORM(guess),usable=int(ok and error<=.25 and cosine>=.95)))
    return rows


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results"/"derivative_prediction_v1")
    source=parser.parse_args().output
    meta=json.loads((source/"seal.json").read_text())
    for name,digest in meta["inputs"].items():
        if name.endswith("_observations.npz") and sha(source/name)!=digest:
            raise ValueError(name)
    path=source/"snapshot_scores.csv"
    if path.exists():
        raise FileExistsError(path)
    rows=[]
    for system in PLAN["systems"]:
        for block in range(PLAN["blocks"]):
            rows.extend(estimate(system,block,source))
    write_csv(path,rows)
    print("SNAPSHOT",len(rows),sha(path),flush=True)
