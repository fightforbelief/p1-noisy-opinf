"""Regenerate every new block and compare keyed numerical outputs with the archive."""
from pathlib import Path
import numpy as np
import pandas as pd

from study import CFG, HERE, block_rows
from noise_check import records
from verify_limit import calculate


def compare(filename, rows, keys):
    expected = pd.read_csv(HERE/"evidence"/filename).set_index(keys).sort_index()
    actual = pd.DataFrame(rows).set_index(keys).sort_index()
    pd.testing.assert_index_equal(expected.index, actual.index, exact=False)
    assert expected.index.is_unique and actual.index.is_unique
    assert set(actual) == set(expected)
    for column in expected:
        np.testing.assert_allclose(actual[column], expected[column], rtol=2e-6, atol=2e-12,
                                   err_msg=f"{filename}: {column}")
    print(f"Verified {len(expected)} rows of {filename}", flush=True)


if __name__ == "__main__":
    blocks = (row for system in CFG["systems"] for block in range(CFG["blocks"])
              for row in block_rows(system, block))
    compare("blocks.csv", blocks, ["system","block","rank","spacing","lags"])
    compare("noise.csv", records(), ["system","block","rho","lags"])
    compare("quartic.csv", calculate(), ["lags","h"])
