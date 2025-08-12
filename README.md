# eeg_glm_aperiodic_covariate
Conventional baseline correction in EEG time–frequency analysis can spuriously mix oscillatory changes with broadband (aperiodic) shifts when prestimulus baseline aperiodic parameters differ (Gyurkovics et al., 2021; Donoghue et al., 2020). This simple simulation illustrates this effect and applies a GLM-based solution that includes the aperiodic offset and exponent as covariates to address the issue (Wen & Liu, 2016; Alday, 2019).

### Figure 1 — Illustration of the baseline correction bias

![Figure 1](https://github.com/amisepa/eeg_glm_aperiodic_covariate/blob/main/fig1.png?raw=true&v=2)

The figure shows mean baseline vs. post-stimulus spectra for a simulated dataset with both a true alpha decrease and broadband offset increase. In **linear units** (left), the absolute change is visible as a dip in alpha power; in **dB units** (right), multiplicative normalization distorts the apparent alpha change because the broadband component differs between baseline and post.


### Figure 2 — GLM with aperiodic covariates

![Figure 2](https://github.com/amisepa/eeg_glm_aperiodic_covariate/blob/main/fig2.png?raw=true&v=2)


The figure shows how including baseline aperiodic offset, exponent, and alpha power as covariates in a GLM recovers the *true* oscillatory change without bias from broadband shifts. Unlike the standard dB baseline correction, the GLM approach yields estimates that are independent of baseline alpha power.


### Figure 3 — GLM diagnostics

![Figure 3](https://github.com/amisepa/eeg_glm_aperiodic_covariate/blob/main/fig3.png?raw=true&v=2)

Diagnostic plot showing GLM coefficients (β) for **bsl offset**, **exponent**, and **alpha power** across frequencies.  
Left panel: model fitted to changes in linear power.  
Right panel: model fitted to changes in dB power.  
These coefficients explain how baseline spectral properties influence apparent changes and why the GLM adjustment removes broadband bias.

## References

Gyurkovics, M., Clements, G. M., Low, K. A., Fabiani, M., & Gratton, G. (2021). The impact of 1/f activity and baseline correction on the results and interpretation of time–frequency analyses of EEG/MEG data: A cautionary tale. NeuroImage, 237, 118192.

Donoghue, T., Haller, M., Peterson, E. J., Varma, P., Sebastian, P., Gao, R., ... & Voytek, B. (2020). Parameterizing neural power spectra into periodic and aperiodic components. Nature Neuroscience, 23(12), 1655–1665.

Wen, H., & Liu, Z. (2016). Separating fractal and oscillatory components in the power spectrum of neurophysiological signal. Brain Topography, 29, 13–26.

Alday, P. M. (2019). How much baseline correction do we need in ERP research? Extended GLM model can replace baseline correction while lifting its limits. Psychophysiology, 56(12), e13451.
