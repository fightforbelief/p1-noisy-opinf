"""Validate the complete measured-water-level evidence and render the manuscript tables."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "evidence" / "tides"
OUT = HERE.parent / "manuscript"
PLAN = json.loads((HERE/"tide_protocol.json").read_text())
seal = json.loads((DATA/"seal.json").read_text())
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
assert digest(DATA/"seal.json") == (DATA/"prediction.sha256").read_text().split()[0]
assert seal["archive"] == digest(DATA/"observations.tar.xz")
assert seal["protocol"] == digest(HERE/"tide_protocol.json")
assert seal["runner"] == digest(HERE/"tide_study.py")
scores = pd.read_csv(DATA/"scores.csv", dtype={"station":str,"month":str})
keys = ["station","month","lags","h","tau"]
expected = set(itertools.product(PLAN["stations"], PLAN["months"], PLAN["lags"],
                                 PLAN["lag_minutes"], PLAN["horizon_minutes"]))
assert len(scores) == len(expected) and set(scores[keys].itertuples(index=False,name=None)) == expected
criteria = (scores.valid == 1) & (scores.gap_m >= .02) & (scores.share >= .05)
assert (scores.passed == (criteria & (scores.error_60 <= .25))).all()
for width in (42,90):
    scores[f"passed_{width}"] = criteria & (scores[f"error_{width}"] <= .25)
valid = scores.query("valid == 1")
np.testing.assert_allclose(valid.gap_m, valid.share*valid.increment_m, rtol=1e-12)
np.testing.assert_allclose(valid.error_60, np.abs(1+np.sign(valid.slope_change)*valid.predicted_m/valid.gap_m),
                           rtol=1e-10, atol=2e-13)
later = scores[scores.month != PLAN["months"][0]]
primary = later.query("h == 60 and tau == 60")
table = []
for station,name in PLAN["stations"].items():
    row = [name]
    for m in PLAN["lags"]:
        pair = primary[(primary.station == station) & (primary.lags == m)]
        if not pair.valid.all():
            row += [r"\multicolumn{3}{c}{ineligible}"]
            continue
        row += [f"{100*pair.gap_m.min():.2f}", f"{100*pair.share.min():.1f}", f"{100*pair.error_60.max():.1f}"]
    table.append(" & ".join(row) + r" \\")
(OUT/"tide_table.tex").write_text("\n".join(table)+"\n")
passed = later.groupby(["h","tau","lags","station"]).passed.min().groupby(["h","tau","lags"]).sum()
grid = []
for h in PLAN["lag_minutes"]:
    grid.append(" & ".join([str(h), *(f"{passed[h,t,1]} / {passed[h,t,2]}" for t in PLAN["horizon_minutes"])])+r" \\")
(OUT/"tide_grid.tex").write_text("\n".join(grid)+"\n")
print("Worst-month results by station:\n"+"\n".join(table))
print("Both-month counts:\n"+"\n".join(grid))
for m, group in primary.groupby("lags"):
    print("lag",m,"qualifiers",group.groupby("station").passed.min().loc[lambda x:x==1].index.tolist())
    print("window counts",group.groupby("station")[["passed_42","passed","passed_90"]].min().sum().to_dict())
    print("forecast RMSE (cm), median local gap, median error and separable error:",
          100*group[["forecast_ols_m","forecast_iv_m","gap_m","error_60","separable_error"]].median().to_numpy())
    print("forecast improvements",sum(group.forecast_iv_m < group.forecast_ols_m),"of",int(group.valid.sum()))
print("Coverage",scores.valid.sum(),len(scores),min(scores.anchors/scores.slots))
