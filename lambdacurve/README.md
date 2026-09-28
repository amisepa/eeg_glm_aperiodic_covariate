# lambdacurve

Does a result about oscillatory (periodic) power depend on how the aperiodic
background was removed? This package answers that for a regression slope or a
within-subject contrast, and provides the other reusable methods of the paper:
aperiodic fits by Whittle deviance, a log-free estimator of the coupling
exponent, and a null for group statistics run through your own pipeline.

## λ, the λ-curve and λ*

With `a` the periodic and `b` the aperiodic power in a band, the coupling
exponent λ sets how the two combine, `a = c · b^λ`, with `c` the intrinsic
strength of the oscillation. Removing the background assumes a λ: subtracting
it in linear power (IRASA) assumes λ = 0, dividing it out (specparam, dB
baselines) assumes λ = 1. Periodic power under an assumed λ is
`y = ln a − λ ln b`, so any linear effect on `y` is linear in λ:

    s(λ) = s_a − λ s_b

where `s_a` and `s_b` are the same effect computed on `ln a` and on `ln b`.
The **λ-curve** is `s(λ)` with its bootstrap interval over a grid of λ. It
changes sign at the **crossover** `λ* = s_a / s_b`. If λ* lies outside [0, 1]
the conclusion holds under either rule; if it lies inside, the conclusion
depends on the assumed λ. λ* is large whenever the background barely changes
with the predictor, so look at `s_b` first: when its interval includes zero
(`lam_star_bounded` is False) λ* is unbounded and its interval says nothing,
while the effects at λ = 0 and 1 still do.

## What it does not do

It does not tell you λ. λ cannot be recovered from the shape of a single
spectrum, and none of the open designs we tried identify it. The package shows
whether a conclusion depends on λ, and reports the effect at λ = 0 and λ = 1
and across the curve rather than under one assumed value. The log-free
estimator returns a λ-hat, but it is biased by anything that moves the rhythm
together with the background (eye state, arousal, artefact, peak power
leaking into the background fit); calibrate it by simulation under your own
design and check the instrument before reading it against 0 and 1.

## Example

```python
import numpy as np
from lambdacurve import lambda_curve, verdict

rng = np.random.default_rng(1)
n = 300
age = rng.uniform(6, 18, n)
sex = rng.integers(0, 2, n).astype(float)
b = np.exp(1.0 - 0.10 * age + rng.normal(0, 0.4, n))     # background falls with age
a = np.exp(0.5 - 0.05 * age + rng.normal(0, 0.5, n))     # alpha falls, but less

# between subjects: age slope of ln a - lambda ln b, adjusted for sex
r = lambda_curve(np.log(a), np.log(b), x=age, covariates=sex)
print(r)               # effects at lambda = 0 and 1, lambda* with CI and HDI
print(verdict(r))      # 'reverses': falls at lambda = 0, rises at lambda = 1

# within subjects: condition 2 minus condition 1, per subject
a1, b1 = rng.gamma(4, 0.5, 40), rng.gamma(9, 1.0, 40)
a2, b2 = a1 * np.exp(rng.normal(0.8, 0.3, 40)), b1 * np.exp(rng.normal(0.3, 0.2, 40))
w = lambda_curve(np.log(a2) - np.log(a1), np.log(b2) - np.log(b1))
print(w.lam_star, w.hdi, w.p_cross_in_01)
print(w.effect(0.5))   # estimate and 95% interval at any lambda
```

Band power from a spectrum, keeping `a <= 0` visible:

```python
from lambdacurve import band_power
bp = band_power(P, f, band=(8, 12), fit_range=(2, 40), censor=[(6, 16)])
# bp["a"] can be <= 0. lambda_curve refuses non-finite ln a unless
# dropna=True, and then reports n_dropped: dropping a <= 0 selects on a
# quantity that grows with the background.
```

## Contents

| function | what |
|---|---|
| `fit_aperiodic`, `ap_eval`, `fit_mask` | fixed, plateau, knee, knee_plateau models, fitted by Whittle deviance |
| `band_power`, `peak_frequency` | total, background `b` (plateau excluded) and periodic `a = total − b − plateau` in a band |
| `lambda_curve` → `LambdaCurve` | `s_a`, `s_b`, λ*, pairs-bootstrap CI, Bayesian-bootstrap HDI, share of draws with λ* in [0, 1], the curve, `effect(λ)` |
| `verdict` | holds under both / reverses / depends on lambda / null under both, from the intervals at λ = 0 and 1 |
| `coupling_two_conditions` | log-free λ from two conditions, split-half band powers, subject bootstrap |
| `coupling_levels` | log-free λ from fluctuations within units (epochs within a participant) |
| `matched_null`, `spectral_noise` | null distribution of any group statistic, through your own per-participant code, with as many simulated participants as real ones |

Notes:

- Censor a wide window around the peak (6-16 Hz for alpha) when fitting the
  background, and the region of the harmonic if it falls in the fit range.
  Flexible models (knee) absorb peaks: use them to report aperiodic
  parameters, not for band power in a coupling analysis.
- A peak-free control band, where the answer must be undefined, is the
  specificity check for any coupling estimate.
- `matched_null` simulates one null participant per real participant (from
  that participant's own template), so the null has the sample's size and
  composition. The null of a correlation between group maps moves with n.
- The bootstrap uses numpy's `default_rng(seed)` in the same order as the
  research scripts, so `seed=0` reproduces their tables.

## Install and test

    pip install .            # or: pip install -e .[test]
    pytest                   # about 15 s

Python ≥ 3.9, numpy, scipy. A MATLAB version of `lambda_curve` and `verdict`
is `code/lib/oof_lambda_curve.m` in the same repository.

## Licence

GPL-3.0, as the repository (see LICENSE).
