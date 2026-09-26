# eeg_glm_aperiodic_covariate

Separating periodic (oscillatory) from aperiodic (1/f) activity in EEG power
spectra with a generalized linear model.

Removing the aperiodic background from a spectrum always involves an
assumption about how it combines with the oscillations. Subtracting it in
linear power (as IRASA does) assumes the two are added. Subtracting it in log
power (as specparam does) assumes the oscillation scales with the
background. This repository treats that assumption as a parameter, the
coupling exponent λ:

    P(f) = L(f) + Σ aₙ Gₙ(f) L(f)^λ

where L is the aperiodic component and Gₙ are peak shapes. λ = 0 is additive
and λ = 1 multiplicative. For band power this reduces to log a = log c +
λ log b, so λ is the coefficient of the aperiodic covariate in a Gamma GLM
(log link) of periodic power.

The code covers:

- a MATLAB library of aperiodic estimators (full, censored and robust
  regression, Theil–Sen, a specparam port, IRASA, a knee+plateau Whittle fit)
  and of the coupling model;
- simulations with known ground truth: an estimator benchmark, the effect
  of the assumed λ on condition contrasts, and calibration of the λ estimate;
- a Python pipeline for the Healthy Brain Network resting-state EEG
  (eyes open vs eyes closed, about 2,800 participants), including
  specificity controls, alpha power against age, and a topographic test
  with its null simulation.

The accompanying paper is in preparation; a preprint will be linked here.

## Main findings so far

- λ cannot be recovered from a single spectrum, only from how the
  background varies across trials, epochs or conditions.
- Assuming the wrong λ distorts condition effects. In a simulated contrast
  in which the background steepens after the event, the recovered change
  is 69% too large when the truth is additive but the background is divided
  out (λ = 1), and 72% too small when the truth is multiplicative but the
  background is subtracted (λ = 0).
- Censored regression is the least biased of the simple aperiodic
  estimators.
- In HBN (2,396 participants aged 5-22 after quality control), eyes-closed
  alpha power decreases with age when the background is removed additively
  (λ = 0, -5.0% per year) and increases when it is removed as specparam
  does (λ = 1, +4.8% per year). The sign changes at λ ≈ 0.52.
- The eyes-open vs eyes-closed contrast does not identify λ: eye closure
  also changes arousal, ocular and muscle activity, and no test available
  in these data separates those from coupling.

## Layout

    code/
      lib/                       MATLAB functions (oof_*): model, synthesis,
                                 Welch PSD, aperiodic estimators, coupling fit
      sim01_estimator_benchmark.m  estimator bias, reliability, coupling estimate
      sim02_ersp_contrast.m        condition contrasts under additive vs
                                   multiplicative coupling
      test_synth.m, test_lambda_identify.m   checks of synthesis and
                                   identifiability
      hbn_extract_psd.py         download HBN-EEG and compute Welch PSDs
      hbn_fit.py                 aperiodic fits and band power per subject
      hbn_lambda.py              coupling estimate and (p, q) tests
      hbn_controls.py            window and band specificity controls
      hbn_kp.py, ap_models.py    knee+plateau aperiodic model
      hbn_age_alpha.py           alpha vs age under each separation rule
      hbn_age_alpha_robust.py    quality control; age slopes across
                                 estimators; crossover λ*
      hbn_age_alpha_bands.py     fixed band, specparam and scalp-gain checks
      hbn_topography*.py         topographic test (fitted and fit-free)
      spatial_neff.py            effective number of channels for map
                                 correlations
      sim_*.py                   calibration curves and null simulations
      identifiability_power.py   data needed to identify λ from one spectrum
      joint_fit.py               joint aperiodic + peak fit
      figures/make_figures.py    Figures 1-5 and Supplementary Figures 1-2
    results/                     group-level outputs (CSV)
    archive/                     superseded first simulations (see its README)

## Running

MATLAB with the Statistics and Machine Learning, Optimization and Signal
Processing toolboxes:

    matlab -batch "run('code/test_synth.m')"
    matlab -batch "run('code/sim01_estimator_benchmark.m')"
    matlab -batch "run('code/sim02_ersp_contrast.m')"
    matlab -batch "scenario='flatten'; run('code/sim02_ersp_contrast.m')"

Python 3.10+ with numpy, scipy, pandas, statsmodels, scikit-learn, mne,
specparam, matplotlib and h5py:

    # 1. PSDs from OpenNeuro (HBN-EEG releases 1-11: ds005505-ds005512 and
    #    ds005514-ds005516; there is no ds005513);
    #    output directory set by HBN_OUT (default ./hbn_psd)
    python code/hbn_extract_psd.py ds005505 ds005506 --workers 6
    # 2. fits and coupling estimates
    python code/hbn_fit.py
    python code/hbn_lambda.py
    python code/hbn_controls.py
    python code/hbn_kp.py --workers 8
    python code/hbn_age_alpha.py
    python code/hbn_age_alpha_robust.py      # writes the QC flags used below
    python code/hbn_age_alpha_bands.py
    # 3. topographic analyses and their null
    python code/hbn_topography.py --workers 8
    python code/hbn_topography.py --workers 8 --fit-range 4,40 --tag _f4-40
    #    likewise 2,30 (_f2-30), and 5,40 with --censor 7,16 (_f5-40)
    python code/hbn_topography_flanks.py
    python code/spatial_neff.py
    python code/sim_topography_leakage.py --workers 8 \
        --out results/sim_topography_null.csv \
        --out-flanks results/sim_topography_null_flanks.csv
    #    repeat with --harmonic 0 and with --iaf-shift -0.3
    # 4. calibration and identifiability
    python code/sim_calibration.py
    python code/identifiability_power.py
    # 5. figures
    python code/figures/make_figures.py figures

Per-subject derivatives are not included, because they contain participant
age and sex. All of them can be regenerated from the public data. The one
per-subject file included, results/hbn_qc_flags.csv, holds only the
quality-control flags.

## Data

Healthy Brain Network EEG (Shirazi et al., 2024), available on OpenNeuro.

## Context

This work is part of the sccn/OneOverF collaboration on periodic/aperiodic
separation (https://github.com/sccn/OneOverF).

## References

Alday, P. M. (2019). How much baseline correction do we need in ERP research?
Extended GLM model can replace baseline correction while lifting its limits.
Psychophysiology, 56(12), e13451.

Donoghue, T., et al. (2020). Parameterizing neural power spectra into periodic
and aperiodic components. Nature Neuroscience, 23(12), 1655–1665.

Gyurkovics, M., Clements, G. M., Low, K. A., Fabiani, M., & Gratton, G.
(2021). The impact of 1/f activity and baseline correction on the results and
interpretation of time–frequency analyses of EEG/MEG data: A cautionary tale.
NeuroImage, 237, 118192.

Kałamała, P., Clements, G. M., Gyurkovics, M., et al. (2026). How to improve
the reliability of aperiodic parameter estimates in M/EEG: A method
comparison. Psychophysiology, 63(3), e70272.

Shirazi, S. Y., et al. (2024). HBN-EEG: The FAIR implementation of the Healthy
Brain Network (HBN) electroencephalography dataset. bioRxiv,
10.1101/2024.10.03.615261.

Wen, H., & Liu, Z. (2016). Separating fractal and oscillatory components in
the power spectrum of neurophysiological signal. Brain Topography, 29, 13–26.
