"""Validate archived rows and derive the manuscript tables, numbers and figures."""
from pathlib import Path
import itertools
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "evidence"
OUT = HERE.parent / "manuscript"
OUT.mkdir(exist_ok=True)
CFG = json.loads((HERE/"protocol.json").read_text())
META = json.loads((DATA/"complete.json").read_text())
for path in (DATA/"blocks.csv", HERE/"protocol.json", HERE/"study.py"):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == META[path.name], path
d = pd.read_csv(DATA/"blocks.csv")
idx = ["system", "rank", "spacing", "block"]
expected = set(itertools.product(CFG["systems"], CFG["ranks"], CFG["spacings"], range(CFG["blocks"]), CFG["lags"]))
assert set(d[idx+["lags"]].itertuples(index=False, name=None)) == expected and len(d)==len(expected)
assert np.isfinite(d.select_dtypes("number")).all().all() and (d.w_cols==d.w_rank).all()
assert (d["shift"] <= d.bound+1e-10).all()
p = d.pivot(index=idx, columns="lags", values=["shift", "bound", "test_gap", "test_ols", "test_iv"])
assert (p["bound"][2] <= p["bound"][1]+1e-10).all()
q = pd.read_csv(DATA/"quartic.csv")
historical = pd.read_csv(DATA/"discovery.csv").set_index("r")
noise = pd.read_csv(DATA/"noise.csv")
assert len(noise)==128 and not noise.duplicated(["system","block","rho","lags"]).any()
zero = noise[noise.rho==0].merge(d[(d.spacing==1)&(d["rank"]==10)],on=["system","block","lags"])
assert len(zero)==32
np.testing.assert_allclose(zero.shift_x,zero.shift_y,rtol=1e-6,atol=1e-12)
names = {"burgers":"Burgers", "allen_cahn":"Allen--Cahn"}
rows, macros = [], {}
def tex(x):
    power = int(np.floor(np.log10(abs(x)))) if x else 0
    if -2 <= power <= 2:
        return f"{x:.3g}"
    return rf"{x/10**power:.2f}\!\times\!10^{{{power}}}"
def table(name, rows):
    (OUT/name).write_text("\n".join(" & ".join(row)+r" \\" for row in rows)+"\n")
for system in CFG["systems"]:
    for rank in CFG["ranks"]:
        a = p.xs((system,rank,1),level=idx[:3])
        ratio = a["shift"][2] / a["shift"][1]
        vals = [a["shift"][1].median(), a["shift"][2].median(), ratio.median(), ratio.min(),ratio.max()]
        rows.append([names[system],str(rank)]+["$"+tex(v)+"$" for v in vals])
table("paired.tex",rows)
rows=[]
for system in CFG["systems"]:
    for lag in CFG["lags"]:
        a = d[(d.system==system)&(d["rank"]==10)&(d.spacing==1)&(d.lags==lag)]
        vals=[a.test_gap.median(),a.test_ols.median(),a.test_iv.median(),
              100*(a.test_iv/a.test_ols-1).median()]
        rows.append([names[system],str(lag)]+["$"+tex(v)+"$" for v in vals]+[str(sum(a.test_iv<a.test_ols))+"/8"])
table("forecast.tex",rows)
table("quartic.tex",[[str(int(m)),f"{g.leading.iloc[0]:.3f}"]+[f"{v:.4f}" for v in g.sort_values("h",ascending=False).ratio]
      for m,g in q.groupby("lags")])
rows=[]
for rho,g in noise.groupby("rho"):
    vals=[]
    for system in CFG["systems"]:
        s=g[g.system==system].pivot(index="block",columns="lags",values="shift")
        vals.extend([s[1].median(),s[2].median(),(s[2]/s[1]).median()])
    rows.append([str(rho)]+["$"+tex(v)+"$" for v in vals])
table("noise.tex",rows)
ratios=p["shift"][2]/p["shift"][1]
macros.update(PairCount=len(p),SmallerCount=int((ratios<1).sum()),FullTwo=f"{100*historical.loc[20,'iv2_full_O_rel']:.2f}",
              MatchTwo=f"{100*historical.loc[20,'iv2_window_O_rel']:.3f}",MatchOne=f"{100*historical.loc[20,'iv_window_O_rel']:.2f}",
              SolverGap=tex(historical.loc[20,'iv_solver_gap_rel']))
(OUT/"numbers.tex").write_text("".join(chr(92)+"newcommand{"+chr(92)+k+"}{"+str(v)+"}\n" for k,v in macros.items()))
colors=["#18578a","#ae4e24"]
fig, axes=plt.subplots(1,2,figsize=(6.8,3.0),sharey=True,layout="constrained")
for ax,system in zip(axes,CFG["systems"]):
    for lag,color in zip((1,2),colors):
        group=d[(d.system==system)&(d["rank"]==10)&(d.lags==lag)].groupby("spacing")
        x=np.array(sorted(group.groups))*CFG["dt_obs"]
        med=group.bound.median().to_numpy()
        ax.fill_between(x,group.bound.min(),group.bound.max(),color=color,alpha=.13,linewidth=0)
        ax.loglog(x,med,"o-",color=color,label=f"{lag} lag"+("s" if lag==2 else ""))
        ax.loglog(x,med[0]*(x/x[0])**lag,":",color=color,linewidth=1.5)
    ax.set_xticks(x,[".005",".010",".025"]); ax.minorticks_off()
    ax.set_xlabel("Instrument spacing, h"); ax.set_title(names[system].replace("--","–"))
    ax.grid(axis="y",color=".88"); ax.spines[["top","right"]].set_visible(False)
axes[0].set_ylabel("Worst-case displacement factor")
axes[0].legend(frameon=False); fig.savefig(OUT/"spacing.pdf"); plt.close(fig)
print(f"{len(d)} complete fits, {len(p)} pairs, {len(noise)} noise fits")
