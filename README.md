# eeg_glm_aperiodic_covariate
Conventional baseline correction in EEG time–frequency analysis can spuriously mix oscillatory changes with broadband (aperiodic) shifts when prestimulus baseline aperiodic parameters differ. This simple simulation illustrates this effect and apply a GLM-based solution that includes the aperiodic offset and exponent as covariates to address the issue. 


### Figure 1 — Illustration of the baseline correction bias

![Figure 1: Baseline correction bias](https://github.com/amisepa/eeg_glm_aperiodic_covariate/blob/main/fig1.png?raw=true)

The figure shows mean baseline vs. post-stimulus spectra for a simulated dataset with both a true alpha decrease and broadband offset increase. In **linear units** (left), the absolute change is visible as a dip in alpha power; in **dB units** (right), multiplicative normalization distorts the apparent alpha change because the broadband component differs between baseline and post.


### Figure 2 — GLM with aperiodic covariates

![Figure 2: GLM solution](https://github.com/amisepa/eeg_glm_aperiodic_covariate/blob/main/fig2.png?raw=true)

The figure shows how including baseline aperiodic offset, exponent, and alpha power as covariates in a GLM recovers the *true* oscillatory change without bias from broadband shifts. Unlike the standard dB baseline correction, the GLM approach yields estimates that are independent of baseline alpha power.


### Figure 3 — GLM diagnostics

![Figure 3: GLM diagnostics](https://github.com/amisepa/eeg_glm_aperiodic_covariate/blob/main/fig3.png?raw=true)

Diagnostic plot showing GLM coefficients (β) for baseline aperiodic offset, exponent, and baseline alpha power across frequencies. 

Left panel: model fitted to changes in linear power.  Right panel: model fitted to changes in dB power.  

The coefficients reveal how baseline spectral parameters influence apparent changes and why the standard dB approach can misattribute broadband shifts to oscillatory changes.
