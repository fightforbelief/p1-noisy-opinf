# Lagged instruments in operator inference

**A two-time-scale law for lagged operator inference on smooth trajectories**

Yunjie Guo and Hanqing Yang · Independent Researcher, Suzhou, China

[Read the paper](manuscript/paper.pdf) · [Editable LaTeX](manuscript/paper.tex) · [Measured data](publication/evidence) · [Experiment protocol](publication/protocol.json)

For smooth trajectories, the spacing between lagged instruments and the forward prediction horizon jointly determine how much two-stage least squares changes fitted values. For two equally spaced lags, the signed leading matrix has scalar factor **h²τ(τ+3h)/2**. Ignoring the mixed term gives one quarter of the leading magnitude at τ = h. The paper proves a uniform two-parameter expansion and tests signed predictions on held-back viscous Burgers, Allen–Cahn and Lorenz–96 ensembles. It also reports a clean snapshot-only derivative check, a closed-vector-field exception and a measurement-noise sensitivity study.

On the prespecified external small-step region, block-median relative errors in the predicted displacement were 2.36%–2.74% for the joint law versus about 85.7% for the separable approximation. The protocols and pre-score prediction digests are versioned. [PDE protocol](publication/derivative_protocol.json) · [PDE predictions and scores](publication/evidence/derivative) · [External validation protocol](publication/crossover_protocol.json) · [Lorenz–96 predictions and scores](publication/evidence/crossover)

A post hoc comparison recalibrated the separable formula on seven Lorenz–96 blocks at a time. Its median error in the omitted block was 69.45%–69.50%. The fitted differences themselves had block medians of 4.26–6.15 parts per million of the state increment. The revised derivation credits the known sketched-regression angle bound, gives an analytic extension for deficient derivative matrices, and checks the unequal-scale formula with exact rational fits. [Block-level diagnostics](publication/evidence/crossover/review_blocks.csv)

Measured water levels show a material one-lag displacement at six of eight enrolled gauges in both later months. The two-lag criteria were met at zero tide stations and by zero participants across the measured-force grid, including the predefined offline low-pass target. At 40 ms the recorded force gap reached a median 17.06 N, but its magnitude was substantially overpredicted. [Tide scores](publication/evidence/tides/scores.csv) · [Force scores](publication/evidence/force/scores.csv)

## Reproduce the paper

Use Python 3.12 from the project root. The recorded project environment is Windows with Python 3.12.10. For an existing project environment, skip its creation. The simulation uses four worker processes with single-threaded BLAS. From a fresh checkout in PowerShell:

~~~powershell
py -3.12 -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install -r requirements.txt
$env:PYTHONPATH = 'src'
& $py -m pytest -q tests publication
& $py publication/audit.py
~~~

The audit regenerates both PDE ensembles, the noisy-input sensitivity study and the manufactured quartic example. It checks the complete row keys and numeric columns against the published tables. Cross-platform numerical libraries can differ slightly. Relative and absolute comparison tolerances are in the audit script.

To save fresh CSVs in a new checkout, run the commands below in order. They create files under `results/lag_geometry_v1/` and refuse to overwrite corresponding files. The recorded primary run took about 40 seconds on the study machine.

~~~powershell
& $py publication/study.py
& $py publication/verify_limit.py
& $py publication/noise_check.py
~~~

To regenerate the prospective predictions and new scores on Windows, use a fresh output directory for each study. Each seal records the prediction archive before score calculation. The snapshot-only check uses the PDE study directory. The two protocols and scoring functions specify the independent block identifiers, grid and acceptance rule. Full-order states are generated deterministically. Compact signed forecasts and measured scores ship with this repository.

~~~powershell
& $py publication/derivative_study.py seal --output results/replay-pde
& $py publication/derivative_study.py score --output results/replay-pde
& $py publication/snapshot_check.py --output results/replay-pde
& $py publication/crossover.py seal --output results/replay-l96
& $py publication/crossover.py score --output results/replay-l96
& $py publication/review_crossover.py results/replay-l96 results/review-blocks.csv
& $py publication/check_replay.py results/replay-pde results/replay-l96
~~~

The PDE scores have 896 primary evaluations across 16 blocks and 112 temporal-refinement comparisons reusing two of those blocks. The auxiliary snapshot study repeats the same 896 evaluations. External scores contain 512 primary evaluations across eight independent blocks and 64 refinements. Multiple ranks, horizons and spacings within a block share data. The seal metadata fingerprints the calculation's original inputs. Re-compressed NumPy archives and time stamps can differ across platforms; numeric comparisons and matrix scores are the reproducibility criteria.

The measured applications use the archived inputs. The NOAA archive contains verified station-month responses with their request URLs. The PhysioNet archive contains time and left/right total-force columns from twelve public control records. Its [attribution and license](publication/evidence/force/ATTRIBUTION.md) cover the extracted database. To replay both transfers from the root, choose fresh output directories.

~~~powershell
& $py publication/tide_study.py calibrate publication/evidence/tides/observations.tar.xz results/replay-tides
& $py publication/tide_study.py score publication/evidence/tides/observations.tar.xz results/replay-tides
& $py publication/force_study.py calibrate publication/evidence/force/observations.npz results/replay-force
& $py publication/force_study.py score publication/evidence/force/observations.npz results/replay-force
~~~

To rebuild all manuscript tables and figures from the published data, run the following from the project root, then compile in the manuscript directory. Tectonic is a separate LaTeX tool. The submission source files are flat in that directory.

~~~powershell
& $py publication/render_results.py
& $py publication/render_derivatives.py
& $py publication/render_crossover.py
& $py publication/render_tides.py
& $py publication/render_force.py
Set-Location manuscript
tectonic paper.tex
~~~

## Evidence and conventions

| File under publication/evidence | Unit |
|---|---|
| blocks.csv | One clean fit per equation, independent trajectory block, rank, lag spacing and lag count |
| noise.csv | One exploratory noisy fit per equation, block, noise level and lag count |
| quartic.csv | One comparison of the measured angular factor with the analytic leading coefficient |
| discovery.csv | Seven ranks from the earlier 100-trajectory Burgers diagnostic that motivated the study |
| complete.json | Completion record and SHA-256 hashes from the primary Windows run |

In the clean table, `shift` is the fitted-value change divided by the matched OLS residual norm, `bound` is the tangent of the largest principal angle, and `coefficient` is the relative Frobenius change in raw POD coordinates. `test_gap`, `test_ols` and `test_iv` are normalized by the held-out response norm. Blocks are independent within each equation. Fits within a block reuse trajectories. The sensitivity run adds noise to the reduced coordinates using a frozen clean basis.

The historical P1 protocols, E0–E5 reports and code are retained. E5 Part 1 and the subsequent common-window diagnostic supplied the discovery example. The original E5 noise arm and E6–E8 were not run as part of this paper. The [revised scientific report](reports/PROSPECTIVE_VALIDATION.md) documents the two-time-scale results. The dated [original closeout](reports/PUBLICATION_CLOSEOUT.md) explains the change of question and distinguishes the evidence streams.
