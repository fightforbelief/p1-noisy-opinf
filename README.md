# Lagged instruments in operator inference

**How closely do lagged instruments reproduce least squares in operator inference?**

Yunjie Guo and Hanqing Yang · Independent researchers, Suzhou, China

[Read the paper](manuscript/paper.pdf) · [Editable LaTeX](manuscript/paper.tex) · [Measured data](publication/evidence) · [Experiment protocol](publication/protocol.json)

The paper characterizes how smooth lagged features approach the current feature space in two-stage least squares. It derives the leading coefficient for both the worst-case change in fitted values and the change for a given response. In two clean PDE ensembles, adding a second lag reduced the realized displacement in all 144 matched block comparisons. At one percent measurement noise, the median two-lag to one-lag ratio approached 0.97 in both equations. These comparisons quantify agreement with ordinary least squares. Held-out one-step errors are reported separately.

## Reproduce the paper

Use Python 3.12 from the project root. The recorded project environment is Windows with Python 3.12.10. For an existing project environment, skip its creation. The simulation uses four worker processes with single-threaded BLAS. From a fresh checkout in PowerShell:

~~~powershell
py -3.12 -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install -r requirements.txt
$env:PYTHONPATH = 'src'
& $py -m pytest -q tests publication/test_study.py
& $py publication/audit.py
~~~

The audit regenerates both PDE ensembles, the noisy-input sensitivity study and the manufactured quartic example. It checks the complete row keys and numeric columns against the published tables. Cross-platform numerical libraries can differ slightly. Relative and absolute comparison tolerances are in the audit script.

To save fresh CSVs in a new checkout, run the commands below in order. They create files under `results/lag_geometry_v1/` and refuse to overwrite corresponding files. The recorded primary run took about 40 seconds on the study machine.

~~~powershell
& $py publication/study.py
& $py publication/verify_limit.py
& $py publication/noise_check.py
~~~

To rebuild all manuscript tables and the figure from the published data, run the following from the project root, then compile in the manuscript directory. Tectonic is a separate LaTeX tool. The submission source files are flat in that directory.

~~~powershell
& $py publication/render_results.py
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

The historical P1 protocols, E0–E5 reports and code are retained. E5 Part 1 and the subsequent common-window diagnostic supplied the discovery example. The original E5 noise arm and E6–E8 were not run as part of this paper. The dated [closeout](reports/PUBLICATION_CLOSEOUT.md) explains the change of question and distinguishes the evidence streams.
