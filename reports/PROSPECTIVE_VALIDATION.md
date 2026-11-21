# Two-time-scale validation · 21 November 2026

The revision asks whether the formula can predict a measured change between two fitted regressions, including its direction and magnitude, before either lagged regression is fit. Baseline evidence and the original hypotheses remain described in [the original closeout](PUBLICATION_CLOSEOUT.md).

## What changed

The original fixed-response expansion follows the angle between current and lagged feature spaces. When a future state is itself selected from the same dictionary, the future response residual also approaches the lag space. The manuscript proves a signed leading term for a moving horizon with distinct, fixed positive lag multipliers, then proves a remainder uniform in the ratio of two independently shrinking time scales. For two consecutive lags the scalar factor is h²τ(τ+3h)/2. A Riccati example illustrates how physical closure can remove the nominal leading term.

The related-work discussion identifies the separate contributions of classical projection geometry, small-delay derivative coordinates and interpolation remainders. The paper claims the two-time-scale response expansion and its numerical validation. It makes no priority claim for any of those ingredients.

## Sequence of evidence

| Stage | Design and chronology | Result unit |
|---|---|---|
| Original work | Historical E5 discovery, then separately designed Burgers and Allen–Cahn ensembles | Regression agreement, held-out error and measurement-noise sensitivity |
| Prospective PDE check | Protocol commit 6eb5692. Eighteen forecasts, including two time-step refinements, were sealed before scoring. Exact-byte seal commit 9b96809 | Signed displacement on matched anchor rows for two response arms |
| Snapshot-only sensitivity | Same PDE ensembles and scoring rule. Derivatives estimated from three extra forward observations at times disjoint from all scored times | Offline clean-data check sharing the primary blocks |
| External crossover validation | Protocol commit 166bbf8 after the PDE results. Lorenz–96 model unseen in the original project. External seal recorded at 1953a77 before scoring | Paired relative matrix error of the joint and separable formulas |
| Analytic and numerical controls | Quartic curves, unequal forward/backward steps, a closed Riccati field, independent direct fits, and halved integration steps | Algebraic identities, response sign, orders, conditioning and temporal resolution |

The internal seals establish the sequence of computation. They are not external public preregistrations. The simulation protocol fixes the random streams, train-only POD, grid, signs, numerical resolution floor and accuracy criteria. Full score files retain all grid cells and refinements. Eight independent ensembles per system support the block summaries. Individual grid rows within each ensemble are dependent.

## Interpretation

At equal forward and backward steps of 0.005 on the two PDEs, the fixed-horizon two-lag formula predicted a median 0.247 times the observed displacement norm. The coupled formula predicted 1.003 times that norm on the exact same rows. The signed matrix errors include orientation. With extra clean snapshots all 896 pass/fail decisions and all 16 usable range endpoints matched the equation-derivative calculations. This sensitivity analysis shares trajectories with the primary calculation and requires additional acquisition times.

Small-lag agreement with ordinary least squares does not identify the physical operator or validate an instrument's statistical exogeneity. High observational noise breaks the smooth-time argument. The tested success regions describe these grids. The manuscript distinguishes those regions from the order theorem and from long-horizon prediction.

## External result

All eight Lorenz–96 blocks met the predeclared joint-formula criterion in the small-step region. The within-block median signed-matrix relative errors ranged from 2.36% to 2.74% for the joint formula. The comparator omitting the mixed term gave approximately 85.7%. All 576 rows, including 64 refinements, passed rank and resolution checks. Errors across the entire 4-by-4 step grid are reported in the manuscript and CSV. These are errors predicting the difference between fitted maps, not errors predicting the underlying future state.

## Reproduction check

A fresh run in the Windows project environment regenerated the forecasts and scores for all three studies. The replay checker matched 1,008 PDE records, 896 snapshot records and 576 external records to the archived keyed results. The project test suite passed all 160 tests.

## Referee sensitivity analysis

These checks were designed after the external results were known. The primary protocol and scores remain at their original digests. The extra analysis uses all 18 primary two-lag evaluations per Lorenz–96 block. It fits the two coefficients of the leading polynomial using squared relative matrix errors, then evaluates a freely rescaled separable formula by leaving one whole block out. Across the eight blocks the fitted mixed coefficient ranges from 2.9913 to 2.9977 (theory 3), and the recalibrated separable formula gives held-out block medians of 69.45% to 69.50%. The measured fit difference is small in state units. Its block medians are 4.26 to 6.15 parts per million of the reduced state increment.

Each block contributes one row to [the archived sensitivity table](../publication/evidence/crossover/review_blocks.csv). The six unequal-scale rational-arithmetic checks in [the exact controls](../publication/test_uniform_exact.py) use independently fitted normal equations. A seventh control demonstrates why two analytic lag spaces can have a limiting direction absent from the derivative matrix. These manufactured checks cover ratios approaching zero and infinity. No application-level noise robustness or improvement in forecasting is inferred from them.

The Windows project environment reproduced the eight-row diagnostic table byte for byte from the separate full-trajectory replay. All 167 tests passed, including the seven new exact controls. The 13-page submission PDF also matched a standalone rebuild from the submission archive page by page in text and rendered pixels.

## Measurements of coastal water level

Protocol 6cb28da fixed the sites, months, lag counts, grid and materiality criteria before the fits. Commit 13baaff recorded the Windows SHA-256 of the January derivative coefficients before the July and October scoring run. Boston, The Battery, Charleston, Key West, Seattle and Honolulu qualified with one lag at the prespecified one-hour step and horizon. Galveston and San Francisco did not reach the January coverage requirement and remain in the eight-site denominator. No station qualified for two lags in both evaluation months anywhere on the nine-point grid. A fresh Windows replay reproduced the complete 432-row file byte for byte, including ineligible rows. [Observed scores](../publication/evidence/tides/scores.csv) · [Protocol](../publication/tide_protocol.json) · [NOAA data service](https://api.tidesandcurrents.noaa.gov/api/prod/)

## Measurements of vertical foot force

The tide findings preceded protocol 11d2ce2. Its participating records, filters, time grid, calibration split and thresholds were chosen before scoring any force records. Commit 982f0f7 fixed the first-third coefficient digest. All 648 design cells were eligible. At the primary two-lag setting of 40 ms, 12/12 participants passed both materiality thresholds in both later thirds for recorded force and 11/12 for its offline low-pass component. No participant passed the accuracy threshold in both thirds in either arm at this setting. No participant qualified on both size and accuracy for either arm anywhere on the complete nine-point grid. At 20 ms the filtered component had median error 13.5%, but its median gap was only 1.27 N. The application concerns reduced-model agreement and has no clinical endpoint. [Measured scores](../publication/evidence/force/scores.csv) · [Protocol](../publication/force_protocol.json) · [Data and attribution](../publication/evidence/force/ATTRIBUTION.md)

Both measurement studies ran from the formal Windows project environment. The final suite passed all 170 tests. Fresh output directories reproduced all 432 tide and 648 force score rows byte for byte. The two measurement studies are additional evidence after the earlier Lorenz–96 validation; the force protocol was developed after seeing the tide results.
