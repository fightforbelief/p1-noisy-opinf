"""Compare a regenerated two-time-scale run with its keyed archived scores."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compare(archived, regenerated, name, keys):
    def table(folder):
        frame = pd.read_csv(folder / name).set_index(keys).sort_index()
        if not frame.index.is_unique:
            raise ValueError(f"Duplicate keys in {name}")
        return frame
    expected, actual = table(archived), table(regenerated)
    pd.testing.assert_index_equal(actual.index, expected.index)
    pd.testing.assert_index_equal(actual.columns, expected.columns)
    for column in expected:
        if column in ("usable","valid","resolved","full"):
            np.testing.assert_array_equal(actual[column],expected[column])
        else:
            # Roundoff in tiny matrix differences affects the direct cross-check
            # and refinement more than errors of the primary analytic prediction.
            atol = 3e-4 if column=="direct_error" else 3e-8 if column=="refine_error" else 2e-10
            if column in ("norm","predicted"):
                atol = 1e-18
            np.testing.assert_allclose(actual[column],expected[column],rtol=1e-5,atol=atol,
                                       equal_nan=True,err_msg=f"{name}: {column}")
    print(f"Confirmed {len(expected)} keyed records in {name}",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("pde",type=Path)
    p.add_argument("external",type=Path)
    p.add_argument("--reference-pde",type=Path,default=Path(__file__).parent/"evidence"/"derivative")
    p.add_argument("--reference-external",type=Path,default=Path(__file__).parent/"evidence"/"crossover")
    args=p.parse_args()
    primary=["system","block","rank","lags","ticks","horizon"]
    compare(args.reference_pde,args.pde,"scores.csv",["refinement",*primary])
    compare(args.reference_pde,args.pde,"snapshot_scores.csv",primary)
    compare(args.reference_external,args.external,"scores.csv",["block","refinement","rank","lags","h","tau"])
