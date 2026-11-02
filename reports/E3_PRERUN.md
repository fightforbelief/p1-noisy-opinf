# E3 pre-run note

Pilot directory: `results/e3_pilot/`. It is not the formal result directory.

The pilot used S2, r=8, n_train=30, n_val=20, n_test=50, rho=0.05, replicates 0 and 1, both POD bases, seven estimators, and three sigma2 sources where an estimator needs sigma2. `runs.csv` has 60 data rows. `failures.json` is empty. `meta.json` records `pass: true`.

Timing on HANQING-WIN, 15 workers:

- pilot wall time: 17.28277450008318 s
- mean time of one `(rho, replicate)` job: 16.967052199994214 s
- estimated full-grid wall time: 135.7364175999537 s

The estimate is below the 4 hour gate, so the formal grid stays at R=20 and keeps `est_gd`. No rho or estimator is dropped.

Pilot checks that passed:

- noise is added to the full-order state before POD
- clean basis matches the shared `V_clean` hash `f8f795b73cce74eaff1a57a02f5557d4b6e746e80be8548ebd571330b1d55616`
- noisy-basis hashes differ by replicate and differ from the clean basis
- each basis has its own OLS reference; the two references are not interchangeable
- sigma2 sources do not change the basis, the training noise, or the reference
- oracle matches that basis reference
- ridge is selected once per basis and replicate
- one blowup threshold is shared by every estimator and sigma2 source inside a `(rho, basis, replicate)`
- per-trajectory `rollout_details.npz` reproduces the rollout summaries
- saved operators, summaries, bootstrap intervals, ridge scores, and sigma2 relative errors pass `audit_directory`

Threshold rule used by E3: `1e3 * max |clean projected training state|` on that basis. The multiplier is the approved E2 factor. Because a noisy basis changes coordinates, its threshold can differ by replicate. The clean-basis threshold in this pilot is the same for replicates 0 and 1: `13167.147110481143`.

Pytest after the E3 tests: 115 passed, 0 failed, using `--basetemp` at `%LOCALAPPDATA%\e3-pytest-tmp`. The default pytest temp directory `pytest-of-hanqi` is not accessible on this machine; that is an environment permission, not an E3 failure. The 106 existing tests remain in the suite.

Formal E3 must be launched only after the commit that contains this note, and only from that committed source.
