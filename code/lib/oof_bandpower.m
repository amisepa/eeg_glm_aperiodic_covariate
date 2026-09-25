function [ap, per] = oof_bandpower(fit, f, band)
% Band-average aperiodic and periodic power implied by a fitted model.
%   fit  : output of oof_fit_coupling. fit.peaks{i} holds the raw
%          coefficients a of G(f)*L(f)^lambda, and fit.lambda the exponent.
%   band : [flo fhi] in Hz
% Returns N x 1 vectors in the PSD units of the data (uV^2/Hz).
f = f(:).';
m = f >= band(1) & f <= band(2);
N = numel(fit.offset);
lambda = 0; if isfield(fit,'lambda') && ~isempty(fit.lambda), lambda = fit.lambda; end
ap = zeros(N,1); per = zeros(N,1);
for i = 1:N
    kn = 0; if isfield(fit,'knee') && ~isempty(fit.knee), kn = fit.knee(i); end
    L  = oof_aperiodic(f, fit.offset(i), fit.exponent(i), kn);
    Pp = zeros(1, numel(f));
    if isfield(fit,'peaks') && ~isempty(fit.peaks{i})
        pk = fit.peaks{i};
        for n = 1:size(pk,1)
            Pp = Pp + pk(n,2)*exp(-0.5*((f-pk(n,1))./pk(n,3)).^2);
        end
        Pp = Pp .* (L.^lambda);
    end
    ap(i)  = mean(L(m));
    per(i) = mean(Pp(m));
end
end
