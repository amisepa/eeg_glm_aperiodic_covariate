function out = oof_est_censored(P, f, censor)
% Censored regression as defined by Kalamala, Clements, Gyurkovics et al.
% (2026, Psychophysiology 63, e70272): regress log10 power on log10 frequency
% over the analysis range with a FIXED frequency window excluded from every
% spectrum, chosen a priori as the range where scalp periodic activity is
% expected. Their default is to censor 6-16 Hz within a 2-33 Hz fit range.
%
% The point of censoring the same window everywhere is reliability: unlike
% specparam's data-driven peak detection, the number and identity of the
% fitted points never varies across epochs, channels or participants, so
% between-spectrum differences cannot come from differences in how many
% peaks happened to be detected.
%
% P      : N x nf linear power spectra
% f      : 1 x nf frequencies (Hz), already restricted to the fit range
% censor : [lo hi] window to exclude (default [6 16])
%
% See also oof_est_full (their "full regression" baseline) and
% oof_est_selfcensor (an iterative variant that is NOT their method).

if nargin < 3 || isempty(censor), censor = [6 16]; end
out = oof_est_ols(P, f, f >= censor(1) & f <= censor(2));
end
