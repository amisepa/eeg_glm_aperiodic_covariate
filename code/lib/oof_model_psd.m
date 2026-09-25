function [S, parts] = oof_model_psd(f, ap, pk, lambda)
% Generalized periodic/aperiodic spectral model with a coupling exponent.
%
%   S(f) = L(f) + sum_n a_n * G_n(f) * L(f)^lambda
%
% lambda = 0  -> purely ADDITIVE coupling. Peak amplitudes a_n are absolute
%                power (uV^2/Hz); this is the model implied by IRASA's linear
%                subtraction (Wen & Liu 2016) and by Gyurkovics et al. (2021).
% lambda = 1  -> purely MULTIPLICATIVE coupling, S = L*(1 + sum a_n G_n).
%                Peak amplitudes are relative (dimensionless). This is the
%                model implied by specparam/FOOOF, whose Gaussians are
%                additive in log10 power (Donoghue et al. 2020), and by dB /
%                divisive baseline correction.
% 0 < lambda < 1 -> partial coupling.
%
% ap : struct with fields offset, exponent, and optional knee
% pk : struct array with fields cf, bw, amp  (amp may be [] / absent -> no peak)
% Returns S and, in parts, the aperiodic and periodic contributions.

if nargin < 4 || isempty(lambda), lambda = 0; end
if ~isfield(ap,'knee'), ap.knee = 0; end
f = f(:).';
L = oof_aperiodic(f, ap.offset, ap.exponent, ap.knee);
Pper = zeros(1, numel(f));
if ~isempty(pk)
    for n = 1:numel(pk)
        if isempty(pk(n).amp) || pk(n).amp == 0, continue; end
        Pper = Pper + pk(n).amp .* oof_peak(f, pk(n).cf, pk(n).bw);
    end
end
Pper = Pper .* (L.^lambda);
S = L + Pper;
parts = struct('aperiodic', L, 'periodic', Pper);
end
