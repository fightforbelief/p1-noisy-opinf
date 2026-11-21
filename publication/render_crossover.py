"""Validate the external matrix forecasts and generate unselected horizon-by-spacing results."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
DATA=HERE/"evidence"/"crossover"
OUT=HERE.parent/"manuscript"
SPEC=json.loads((HERE/"crossover_protocol.json").read_text())
META=json.loads((DATA/"seal.json").read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


digest_manifest=json.loads((DATA/"score_manifest.json").read_text())
assert digest(DATA/"seal.json")==digest_manifest["seal"]
assert digest(DATA/"scores.csv")==digest_manifest["scores"]
for name,expected in META["inputs"].items():
    if name.endswith("_prediction.npz"):
        assert digest(DATA/name)==expected
for f,k in [("crossover.py","runner"),("crossover_protocol.json","protocol"),("derivative_study.py","base")]:
    assert digest(HERE/f)==META[k]

d=pd.read_csv(DATA/"scores.csv")
ref=[(b,1) for b in range(SPEC["blocks"])]+[(0,2)]
expected={(*case,r,m,h,t) for case,r,m,h,t in itertools.product(
    ref,SPEC["ranks"],SPEC["lags"],SPEC["h_ticks"],SPEC["tau_ticks"])}
for col in ("h","tau"):
    d[col+"_tick"]=(d[col]/SPEC["tick"]).round().astype(int)
keys=["block","refinement","rank","lags","h_tick","tau_tick"]
assert len(d)==len(expected) and expected==set(d[keys].itertuples(index=False,name=None))
assert np.isfinite(d[["cosine","error","separable_error","norm","predicted"]]).all().all()
np.testing.assert_array_equal(d.usable,((d.error<=.25)&(d.cosine>=.95)&(d.valid==1)).astype(int))
np.testing.assert_allclose(d[d.lags==1].error,d[d.lags==1].separable_error,atol=1e-13)
b=d.query("refinement==1 and lags==2").copy()
# Invalid rows are failures and cannot silently contribute a favorable finite median.
primary=b.query("h<=.005 and tau<=.005").copy()
primary.loc[primary.valid==0,["error","separable_error"]]=[np.inf,np.inf]
paired=primary.groupby("block")[["error","separable_error"]].median()
passed=(paired.error<=.25)&(paired.error<paired.separable_error)
cells=b.groupby(["h_tick","tau_tick"])[["error","separable_error"]].median()
rows=[]
for h in SPEC["h_ticks"]:
    row=[f"{h*SPEC['tick']:g}"]
    for t in SPEC["tau_ticks"]:
        v=cells.loc[(h,t)]
        row.append(f"{v.error:.3f} / {v.separable_error:.3f}")
    rows.append(" & ".join(row)+r" \\")
(OUT/"crossover_table.tex").write_text("\n".join(rows)+"\n")
macros={"JointPctLo":f"{100*paired.error.min():.2f}","JointPctHi":f"{100*paired.error.max():.2f}",
        "SepMedianPct":f"{100*paired.separable_error.median():.1f}","ExternalPass":str(passed.sum()),"JointLo":f"{paired.error.min():.3f}",
        "JointHi":f"{paired.error.max():.3f}","SepLo":f"{paired.separable_error.min():.3f}",
        "SepHi":f"{paired.separable_error.max():.3f}",
        "ExternalInvalid":str(sum(d.valid==0)),
        "ExternalRefine":f"{d.refine_error.max():.2g}"}
review = pd.read_csv(DATA / "review_blocks.csv").set_index("block").sort_index()
assert review.index.tolist() == list(range(SPEC["blocks"])) and (review.cells == 18).all()
assert np.isfinite(review.to_numpy()).all()
np.testing.assert_allclose(review.joint_median_error, paired.error, rtol=1e-11)
for variable, prefix, multiplier, precision in [
    ("intercept", "ReviewA", 1, 4), ("mixed_coefficient", "ReviewB", 1, 4),
    ("calibrated_median_error", "ReviewCal", 100, 2),
    ("median_increment_share", "ReviewPPM", 1e6, 2),
]:
    values = review[variable] * multiplier
    macros[prefix+"Lo"] = f"{values.min():.{precision}f}"
    macros[prefix+"Hi"] = f"{values.max():.{precision}f}"
(OUT/"external_numbers.tex").write_text("".join("\\newcommand{\\"+k+"}{"+v+"}\n" for k,v in macros.items()))
print("block medians\n",paired,"passes",passed.sum())
print("resolution checks",d.valid.value_counts().to_dict(),"direct",d.direct_error.max(),"refinement",d.refine_error.max())
print("all two-lag medians\n",cells.to_string())
