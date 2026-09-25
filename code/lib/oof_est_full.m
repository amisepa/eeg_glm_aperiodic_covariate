function out = oof_est_full(P, f)
% "Full regression" baseline of Kalamala et al. (2026): OLS of log10 power on
% log10 frequency over the whole analysis range, nothing excluded and no peak
% model. Equivalent to specparam with max_n_peaks = 0 up to the optimiser.
out = oof_est_ols(P, f, false(size(f)));
end
