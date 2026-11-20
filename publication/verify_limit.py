"""Manufactured smooth feature with a nonzero leading term for one to three lags."""
import csv
import math
from pathlib import Path

import numpy as np

from study import ROOT, instrument_basis


def calculate():
    t = np.linspace(-1, 1, 81)
    z = np.column_stack([np.ones_like(t), t**4])
    u, r = np.linalg.qr(z)
    for m in (1, 2, 3):
        derivatives = [math.factorial(4)/math.factorial(4-j)*t**(4-j) for j in range(m+1)]
        jet = instrument_basis(np.column_stack([z[:, 0], *derivatives[:-1]]))
        last = np.column_stack([np.zeros_like(t), derivatives[-1]])
        leading = np.linalg.norm((last-jet@(jet.T@last))@np.linalg.inv(r), 2)
        for h in (0.05, 0.02, 0.01, 0.005):
            # The divided differences below span the same lagged columns and avoid
            # cancellation by evaluating each quartic coefficient analytically.
            basis = [np.ones_like(t)]
            for k in range(m):
                col = np.zeros_like(t)
                for degree in range(k, 5):
                    weight = sum((-1)**j * math.comb(k, j) * (-(j+1))**degree
                                 for j in range(k+1))
                    col += math.comb(4, degree)*weight*h**(degree-k)*t**(4-degree)
                basis.append(col)
            w = instrument_basis(np.column_stack(basis))
            cross = w.T@u
            tangent = np.linalg.norm(u-w@cross, 2)/np.linalg.svd(cross, compute_uv=False)[-1]
            yield dict(lags=m, h=h, tangent=tangent, leading=leading,
                       scaled=tangent/h**m, ratio=tangent/h**m/leading)


if __name__ == "__main__":
    output = ROOT / "results" / "lag_geometry_v1" / "quartic.csv"
    rows = list(calculate())
    with output.open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
