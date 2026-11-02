# E4 pre-run note

Pilot directory: `results/e4_pilot/`. It is not the formal result directory.

The pilot used S2, basis=noisy, n_train=30, n_val=20, n_test=50, rho=0.05, replicates 0 and 1, ranks 4, 6, 8, 10, 12 and 16, and estimators ols, ridge, mc, iv2. Sigma2 for mc and iv2 is est_trailing. `runs.csv` has 48 data rows. `failures.json` is empty. `meta.json` records `pass: true`.

Timing on HANQING-WIN, 15 workers:

- pilot wall time: 35.62768879998475 s
- estimated full-grid wall time: 116.62390450015664 s

The estimate is below the 4 hour gate. The formal grid stays at R=10. No rank, rho, or estimator is dropped. S2's replicate floor is 10, so the 04 §1.6 halving step is not available.

Rules fixed before the formal run:

- One full-order noisy training draw and one independent noisy validation draw are shared by all six ranks and all four estimators inside a `(rho, replicate)`.
- POD is one rank-16 SVD. Each candidate rank uses the leading columns. The pilot truncation gap is below 1e-8.
- Reference operator for each rank is OLS on the clean training trajectories projected onto that prefix. Operators are not compared across ranks.
- Rank aggregation is the median of the ten replicate validation medians. Ties take the smaller rank.
- `r_star_val` uses clean validation initial conditions and clean full-order validation states. `r_star_noisyval` uses noisy validation initial conditions and noisy validation states. `r_star_test` is descriptive and is not an input to either selection.
- Ridge lambda is chosen in reduced coordinates by the existing 13-lambda protocol. Rank selection then uses full-order prediction error.
- SNR uses the uncentered second moment of the clean training coordinates on the fixed noisy basis, compared with the one est_trailing value. Empty sets stay empty. The plotted aggregate is the lower central median of nonempty raw ranks.
- Divergence threshold is `1e3 * max |clean projected training state|` on that rank, shared by the four estimators.

Pytest before this note: 123 passed, 0 failed, using `--basetemp` at `%LOCALAPPDATA%\e4-pytest-tmp`. The smoke test now checks that `results/e3/runs.csv` is unchanged, because the accepted E3 archive is present. The default pytest cache directory remains inaccessible on this machine.

Formal E4 must be launched only after the commit that contains this note, and only from that committed source.
