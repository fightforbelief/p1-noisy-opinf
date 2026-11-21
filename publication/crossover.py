"""External two-time-scale test. Predictions for all blocks are sealed before scoring."""
import argparse
import json
import time
from pathlib import Path

from derivative_study import (HERE, ROOT, np, NORM, features, observed, predictions, pod_basis,
                              sha, write_csv)

SPEC=HERE/"crossover_protocol.json"
PLAN=json.loads(SPEC.read_text())
ANCHOR=np.arange(PLAN["first_anchor"],PLAN["last_anchor"]+1,PLAN["stride"])
OFFSETS={0,*PLAN["tau_ticks"],*(-a*h for a in PLAN["lags"] for h in PLAN["h_ticks"])}
TIMES=np.unique(ANCHOR[:,None]+sorted(OFFSETS))


def rhs(x):
    return (np.roll(x,-1,axis=-1)-np.roll(x,2,axis=-1))*np.roll(x,1,axis=-1)-x+PLAN["forcing"]


def second(x,v):
    difference=np.roll(x,-1,axis=-1)-np.roll(x,2,axis=-1)
    dv=np.roll(v,-1,axis=-1)-np.roll(v,2,axis=-1)
    return dv*np.roll(x,1,axis=-1)+difference*np.roll(v,1,axis=-1)-v


def rk4(x,h):
    k1=rhs(x)
    k2=rhs(x+h*k1/2)
    k3=rhs(x+h*k2/2)
    k4=rhs(x+h*k3)
    return x+h*(k1+2*k2+2*k3+k4)/6


def samples(block,ref):
    rng=np.random.default_rng([PLAN["seed"],block])
    x=PLAN["forcing"]+rng.normal(size=(PLAN["trajectories"],PLAN["n"]))
    for _ in range(round(PLAN["spinup"]/PLAN["spinup_step"])):
        x=rk4(x,PLAN["spinup_step"])
    out=np.empty((len(x),len(TIMES),x.shape[1]))
    previous=0
    for i,t in enumerate(TIMES):
        for _ in range((t-previous)*ref):
            x=rk4(x,PLAN["tick"]/ref)
        out[:,i]=x
        previous=t
    return out


def rows(q,offset):
    return q[:,np.searchsorted(TIMES,ANCHOR+offset)].reshape(-1,q.shape[-1])


def cases():
    return [(b,1) for b in range(PLAN["blocks"])]+[(0,2)]


def seal(path):
    path.mkdir(parents=True,exist_ok=False)
    for b,ref in cases():
        field=samples(b,ref)
        anchors=field[:,np.searchsorted(TIMES,ANCHOR)]
        basis,_=pod_basis(list(anchors),max(PLAN["ranks"]))
        q=field@basis
        velocity=rhs(anchors)
        v=(velocity@basis).reshape(-1,basis.shape[1])
        a=(second(anchors,velocity)@basis).reshape(v.shape)
        np.savez(path/f"b{b}r{ref}_states.npz",q=q,basis=basis)
        pred={}
        for rank in PLAN["ranks"]:
            x=rows(q,0)[:,:rank]
            for m in PLAN["lags"]:
                # At c=1 the coefficient from the equal-spacing code is -m R^-1 L'G.
                # Divide by m to retain the unit coefficient for every (h,tau).
                result=predictions(x,v[:,:rank],a[:,:rank],x,m)
                pred[f"r{rank}m{m}"]=result["coupled"]/m
                pred[f"full_r{rank}m{m}"]=int(result["jet_rank"]==result["jet_cols"])
        np.savez(path/f"b{b}r{ref}_prediction.npz",**pred)
    meta=dict(inputs={p.name:sha(p) for p in sorted(path.glob("*.npz"))},
              runner=sha(Path(__file__)),base=sha(HERE/"derivative_study.py"),protocol=sha(SPEC),
              sealed=time.time())
    (path/"seal.json").write_text(json.dumps(meta,indent=2),encoding="utf8")
    print("EXTERNAL_SEAL",sha(path/"seal.json"),flush=True)


def scales(h,tau,m):
    if m==1:
        return h*tau,h*tau
    return .5*h*h*tau*(tau+3*h),.5*h*h*tau*tau


def score(path):
    meta=json.loads((path/"seal.json").read_text())
    assert meta["runner"]==sha(Path(__file__)) and meta["base"]==sha(HERE/"derivative_study.py")
    assert meta["protocol"]==sha(SPEC)
    for name,digest in meta["inputs"].items():
        assert sha(path/name)==digest
    records,coarse=[],{}
    for b,ref in cases():
        with np.load(path/f"b{b}r{ref}_states.npz") as data:
            all_states,basis=data["q"],data["basis"]
        with np.load(path/f"b{b}r{ref}_prediction.npz") as pred:
            for rank in PLAN["ranks"]:
                s=all_states[:,:,:rank]
                q=rows(s,0)
                u,r=np.linalg.qr(features(q),mode="reduced")
                for tick in PLAN["h_ticks"]:
                    past=[]
                    for m in PLAN["lags"]:
                        past.append(features(rows(s,-tick*m))[:,:-1])
                        w=np.column_stack([*past,np.ones(len(q))])
                        direction=r@pred[f"r{rank}m{m}"]
                        for t in PLAN["tau_ticks"]:
                            y=rows(s,t)
                            actual,direct,_,wr,cr=observed(u,q,y,w)
                            norm=NORM(actual)
                            h,tau=tick*PLAN["tick"],t*PLAN["tick"]
                            joint,sep=scales(h,tau,m)
                            cos=np.vdot(direction,actual)/(NORM(direction)*norm)
                            err=NORM(joint*direction-actual)/norm
                            sep_err=NORM(sep*direction-actual)/norm
                            valid=(wr==w.shape[1] and cr==r.shape[0] and
                                   pred[f"full_r{rank}m{m}"] and norm>1e-13*NORM(y-q))
                            ck=(rank,tick,m,t)
                            ref_err=""
                            if b==0:
                                phys=(u@actual)@basis[:,:rank].T
                                if ref==1:
                                    coarse[ck]=phys
                                else:
                                    ref_err=NORM(phys-coarse[ck])/NORM(coarse[ck])
                            records.append(dict(block=b,refinement=ref,rank=rank,lags=m,h=h,tau=tau,
                                                norm=norm,predicted=joint*NORM(direction),cosine=cos,
                                                error=err,separable_error=sep_err,valid=int(valid),
                                                usable=int(valid and err<=.25 and cos>=.95),
                                                direct_error=NORM(direct-actual)/norm,refine_error=ref_err))
    write_csv(path/"scores.csv",records)
    print("EXTERNAL_SCORED",len(records),sha(path/"scores.csv"),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["seal","score"])
    parser.add_argument("--output",type=Path,default=ROOT/"results"/"lag_crossover_v1")
    args=parser.parse_args()
    (seal if args.stage=="seal" else score)(args.output)
