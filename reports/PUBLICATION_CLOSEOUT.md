# P1 journal manuscript · 20 November 2026

The original noisy-operator benchmark yielded mixed conclusions. In E2, correction reduced coefficient error on better-sampled designs and amplified it on poorly conditioned small samples. E4 partially supported the rank-selection hypothesis. In E5, the expected decline in operator-target discrepancy with increasing POD rank did not appear. Two-lag estimates had been compared to an OLS fit that retained two extra starting observations per trajectory.

The journal manuscript follows the common-window finding. It holds the response and anchor times fixed, derives a sharp residual amplification factor and a small-spacing expansion, and tests the latter against an analytic quartic dictionary. The earlier Burgers diagnostic remains discovery evidence. The independent clean study uses new initial conditions, an additional PDE, an internally frozen grid and held-out trajectories. Noise sensitivity was added after the clean comparison. No claim about consistency, uniform rank selection or improved long-horizon forecasting is made.

## Verification and scope

- Windows project environment, Python 3.12.10. Historical regression suite passed 142 tests. With the original three study checks the suite passed 145. The final suite passed all 148 tests in 31.28 seconds, including direct-fit checks of the signed response-specific leading matrix for one, two and three lags.
- Complete clean grid, 288 fits in 16 trajectory blocks. Every fit had the intended numerical rank and satisfied its sharp bound. All 144 paired realized displacements decreased. Multiple ranks and spacings reuse each block.
- Additional checks, 128 noise fits and 12 manufactured cases. The zero-noise sensitivity arm agrees with the primary values at the same rank and spacing.
- The Windows replay regenerated all 288 primary fits, 128 sensitivity fits and 12 manufactured cases. Every keyed numeric column agreed with the evidence bundle at the published tolerances. A fresh Git clone rebuilt all tables and the seven-page PDF. The rendered pages matched pixel-for-pixel.
- Scope closed for this manuscript. The original E5 Part 2 and E6–E8 concern additional scientific questions. Their absence is recorded in the README and the earlier protocols remain available.
- Proposed venue: Numerical Algorithms. Its scope covers theoretical analysis, implementation and numerical behavior of algorithms. See the [scope](https://link.springer.com/journal/11075/aims-and-scope) and [submission instructions](https://link.springer.com/journal/11075/submission-guidelines).
- Author information, funding and competing interests follow the corresponding author's instructions. Both authors should review the proofs, numerical interpretation and computational-assistance statement before approving a journal submission.
