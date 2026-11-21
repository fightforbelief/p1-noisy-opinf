"""Check the complete prospective design and typeset the measured accuracy ranges."""
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
DATA=HERE/"evidence"/"derivative"
OUT=HERE.parent/"manuscript"
PLAN=json.loads((HERE/"derivative_protocol.json").read_text())
SEALED=json.loads((DATA/"seal.json").read_text())
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert digest(HERE/"derivative_protocol.json")==SEALED["protocol"]
assert digest(HERE/"derivative_study.py")==SEALED["runner"]
manifest=json.loads((DATA/"score_manifest.json").read_text())
assert digest(DATA/"seal.json")==manifest["seal"]
assert digest(DATA/"scores.csv")==manifest["scores"]
assert digest(DATA/"snapshot_scores.csv")==manifest["snapshot_scores"]
for name,sha in SEALED["inputs"].items():
    if name.endswith("_prediction.npz"):
        assert digest(DATA/name)==sha
d=pd.read_csv(DATA/"scores.csv")
keys=["system","block","refinement","rank","ticks","lags","horizon"]
cases=[(s,b,1) for s in PLAN["systems"] for b in range(PLAN["blocks"])]
cases += [(s,0,2) for s in PLAN["systems"]]
expected={(*case,*k) for case,k in itertools.product(cases,itertools.product(
    PLAN["ranks"],PLAN["lag_ticks"],PLAN["lags"],["fixed","coupled"]))}
assert set(d[keys].itertuples(index=False,name=None))==expected and len(d)==len(expected)
np.testing.assert_array_equal(d.usable,((d.error<=.25)&(d.cosine>=.95)&(d.full==1)&(d.resolved==1)).astype(int))
# Reconstruct the joint error from separately reported norm ratio and cosine.
ratio=d.predicted/d.norm
np.testing.assert_allclose(d.error**2, 1+ratio**2-2*ratio*d.cosine,atol=1e-12,rtol=3e-7)
base=d[d.refinement==1]
names={"burgers":"Burgers","allen_cahn":"Allen--Cahn"}
def cutoff(group):
    last=0
    for tick in PLAN["lag_ticks"]:
        if group.loc[group.ticks==tick,"usable"].sum()<7:
            break
        last=tick
    return last*PLAN["tick"]

rows=[]
for s in PLAN["systems"]:
    for r in PLAN["ranks"]:
        part=base[(base.system==s)&(base["rank"]==r)]
        ranges=[]
        for arm in ("fixed","coupled"):
            for m in PLAN["lags"]:
                h=cutoff(part[(part.horizon==arm)&(part.lags==m)])
                ranges.append(f"{h:g}" if h else r"\textemdash")
        orders=[]
        for m in PLAN["lags"]:
            g=part[(part.horizon=="coupled")&(part.lags==m)].pivot(index="block",columns="ticks",values="norm")
            orders.append(f"{np.median(np.log2(g[2]/g[1])):.2f}")
        rows.append([names[s],str(r),*ranges,*orders])
(OUT/"ranges.tex").write_text("\n".join(" & ".join(row)+r" \\" for row in rows)+"\n")

fig,axes=plt.subplots(2,2,figsize=(6.5,4.8),sharex=True,sharey=True,layout="constrained")
for row,system in enumerate(PLAN["systems"]):
    for column,arm in enumerate(("fixed","coupled")):
        ax=axes[row,column]
        sample=base[(base.system==system)&(base["rank"]==10)&(base.horizon==arm)]
        for m,color,marker in [(1,"#18578a","o"),(2,"#ae4e24","s")]:
            group=sample[sample.lags==m].groupby("h").error
            median=group.median()
            ax.fill_between(median.index,group.min(),group.max(),color=color,alpha=.12,lw=0)
            ax.loglog(median.index,median,color=color,marker=marker,ms=3,label=f"{m} lag"+("s" if m==2 else ""))
        ax.axhline(.25,color=".3",lw=.8,ls="--"); ax.axhline(1,color=".55",lw=.8,ls=":")
        ax.spines[["top","right"]].set_visible(False)
        if not row: ax.set_title("Fixed horizon" if arm=="fixed" else "Horizon equal to spacing")
        if not column: ax.set_ylabel(names[system].replace("--","–")+"\nRelative matrix error")
        if row: ax.set_xlabel("Lag spacing, h")
        ax.set_xticks([.00025,.001,.005,.025],[".00025",".001",".005",".025"])
axes[0,0].legend(frameon=False,fontsize=8)
fig.savefig(OUT/"prediction.pdf"); plt.close(fig)
print("Ranges and finest-pair observed powers", rows)
for (arm,m),g in base.groupby(["horizon","lags"]):
    print(arm,m,"usable",g.usable.sum(),len(g),"median at finest/coarsest",g.groupby("ticks").error.median().iloc[[0,-1]].tolist())
print("max direct discrepancy",d.direct_error.max(),"max refinement discrepancy",d.refine_error.max(),
      "unresolved",sum(d.resolved==0),"rank failures",sum(d.full==0))

def tex_number(x):
    mantissa, exp=f"{x:.2e}".split("e")
    return mantissa+r"\times10^{"+str(int(exp))+"}"

equal=base[(base.horizon=="fixed")&(base.lags==2)&(base.ticks==20)]
metrics=dict(SequentialError=f"{equal.error.median():.3f}",SequentialCosine=f"{equal.cosine.median():.3f}",
             DirectDiff=tex_number(d.direct_error.max()),RefineDiff=tex_number(d.refine_error.max()))
coupled_equal=base[(base.horizon=="coupled")&(base.lags==2)&(base.ticks==20)]
metrics.update(SequentialRatio=f"{(equal.predicted/equal.norm).median():.3f}",
               CoupledRatio=f"{(coupled_equal.predicted/coupled_equal.norm).median():.3f}",
               AngleError=f"{100*(equal.predicted_angle/equal.angle-1).abs().median():.2f}")
snapshot_file=DATA/"snapshot_scores.csv"
snap=pd.read_csv(snapshot_file)
snap_idx=["system","block","rank","ticks","lags","horizon"]
merged=base.merge(snap,on=snap_idx,suffixes=("_rhs","_snap"),validate="one_to_one")
assert len(merged)==len(base)==len(snap)
np.testing.assert_allclose(merged.norm_rhs,merged.norm_snap,rtol=2e-4,atol=1e-13)
validity=[]
for k,g in snap.groupby(["system","rank","lags","horizon"]):
    a=base[(base.system==k[0])&(base["rank"]==k[1])&(base.lags==k[2])&(base.horizon==k[3])]
    validity.append(cutoff(a)==cutoff(g))
    print("snapshot range",k,cutoff(g))
metrics.update(SnapshotSame=str(sum(validity)),
               SnapshotAgreement=str(int((merged.usable_rhs==merged.usable_snap).sum())),
               SnapshotDelta=f"{(merged.error_rhs-merged.error_snap).abs().median():.3g}")
(OUT/"prospective_numbers.tex").write_text("".join("\\newcommand{\\"+k+"}{"+v+"}\n" for k,v in metrics.items()))
print("snapshot differences",metrics)
