# eeg_glm_aperiodic_covariate
Conventional baseline correction in EEG time, frequency, or time-frequency analysis can spuriously mix oscillatory changes with broadband (aperiodic) shifts when prestimulus baseline aperiodic parameters differ (Gyurkovics et al., 2021; Donoghue et al., 2020). This simple simulation illustrates this effect and applies a GLM-based solution that includes the aperiodic offset and exponent as covariates to address the issue (Wen & Liu, 2016; Alday, 2019).


## References

Gyurkovics, M., Clements, G. M., Low, K. A., Fabiani, M., & Gratton, G. (2021). The impact of 1/f activity and baseline correction on the results and interpretation of time–frequency analyses of EEG/MEG data: A cautionary tale. NeuroImage, 237, 118192.

Donoghue, T., Haller, M., Peterson, E. J., Varma, P., Sebastian, P., Gao, R., ... & Voytek, B. (2020). Parameterizing neural power spectra into periodic and aperiodic components. Nature Neuroscience, 23(12), 1655–1665.

Wen, H., & Liu, Z. (2016). Separating fractal and oscillatory components in the power spectrum of neurophysiological signal. Brain Topography, 29, 13–26.

Alday, P. M. (2019). How much baseline correction do we need in ERP research? Extended GLM model can replace baseline correction while lifting its limits. Psychophysiology, 56(12), e13451.
