"""Validate the force evidence and render participant-level pass counts."""
from pathlib import Path
import hashlib
import itertools
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE/"evidence"/"force"
OUT = HERE.parent/"manuscript"
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
seal = json.loads((DATA/"seal.json").read_text())
assert digest(DATA/"observations.npz") == seal["data"]
assert digest(HERE/"force_protocol.json") == seal["protocol"]
assert digest(HERE/"force_study.py") == seal["runner"]
assert digest(DATA/"seal.json") == (DATA/"prediction.sha256").read_text().split()[0]
plan = json.loads((HERE/"force_protocol.json").read_text())
scores = pd.read_csv(DATA/"scores.csv")
expected = set(itertools.product(list(seal["coefficients"]),["raw","lowpass6"],[1,2,3],plan["h_ms"],plan["tau_ms"]))
keys = ["subject","arm","third","h","tau"]
assert len(scores) == len(expected) and expected == set(scores[keys].itertuples(index=False,name=None))
size = (scores.valid == 1) & (scores.load_share >= .01) & (scores.increment_share >= .05)
assert (scores.passed == (size & (scores.error_30<=.25))).all()
for width in [40,50]:
    scores[f"passed_{width}"] = size & (scores[f"error_{width}"]<=.25)
valid = scores.query("valid == 1")
np.testing.assert_allclose(valid.gap_n,valid.load_share*valid.bilateral_n,rtol=1e-12)
np.testing.assert_allclose(valid.gap_n,valid.increment_share*valid.increment_n,rtol=1e-12)
np.testing.assert_allclose(valid.error_30,np.abs(1+np.sign(valid.slope_change)*valid.predicted_n/valid.gap_n),
                           rtol=1e-10,atol=2e-12)
later = scores.query("third > 1")
pair = later.groupby(["h","tau","arm","subject"]).passed.min().groupby(["h","tau","arm"]).sum()
grid = []
for h in plan["h_ms"]:
    row = [str(h)] + [f"{pair[h,t,'raw']} / {pair[h,t,'lowpass6']}" for t in plan["tau_ms"]]
    grid.append(" & ".join(row)+r" \\")
rows = []
for arm,label in [("raw","Recorded"),("lowpass6","6 Hz")]:
    for step in plan["h_ms"]:
        cell = later.query("valid == 1 and arm == @arm and h == @step and tau == @step")
        med = cell[["gap_n","load_share","increment_share","error_30"]].median()
        row = [label, str(step), f"{med.gap_n:.2f}", f"{100*med.load_share:.2f}",
               f"{100*med.increment_share:.1f}", f"{100*med.error_30:.1f}"]
        rows.append(" & ".join(row)+r" \\")
(OUT/"force_table.tex").write_text("\n".join(rows)+"\n")
print("Full two-third pass counts raw / filtered:\n"+"\n".join(grid))
primary = later.query("h == 40 and tau == 40")
for arm,g in primary.groupby("arm"):
    passed = g.groupby("subject").passed.min()
    print(arm,"qualifying subjects",passed[passed==1].index.tolist())
    print("window counts",g.groupby("subject")[["passed","passed_40","passed_50"]].min().sum().to_dict())
    print("Medians of both evaluation thirds over all eligible participants:")
    print(g.query("valid==1")[["gap_n","load_share","increment_share","error_30","separable_error",
                              "forecast_ols_n","forecast_iv_n"]].median().to_string())
    print("eligible segments",g.valid.sum(),"of",len(g),"improvements",sum(g.forecast_iv_n<g.forecast_ols_n))
    print("participant counts passing only materiality, only accuracy, both:",
          [int(pd.Series(a.to_numpy(),index=g.subject).groupby(level=0).min().sum()) for a in
           [(g.load_share>=.01)&(g.increment_share>=.05),g.error_30<=.25,g.passed]])
print("Valid total",scores.valid.sum(),"of",len(scores))
