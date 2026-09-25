function out = oof_est_kneeplateau(P, f, keep, model)
% Aperiodic fit with a knee and an additive plateau, by Whittle deviance.
%
%   fixed         L(f) = 10^b / f^chi
%   plateau       L(f) = 10^b / f^chi + p
%   knee          L(f) = 10^b / (k + f^chi)                 (Donoghue 2020)
%   knee_plateau  L(f) = 10^b / (k + f^chi) + p             (default)
%
% WHY. A single power law is the wrong model for real scalp spectra, and
% under it the signed residual P - L_hat is a systematic function of the
% background. That is one of the two mechanisms which made a coupling
% estimate return lambda ~ 0.89 in a PEAK-FREE control band, where nothing
% can couple. Fitted on HBN posterior spectra (2-55 Hz, 6-16 Hz censored,
% n = 40), the median residual across 2.5-5 / 30-38 / 45-55 Hz moves from
% -51.6% / -13.7% / -8.5% under a power law to -8.0% / -0.8% / +2.0% here.
%
% The knee flattens the LOW-frequency end: below f_knee = k^(1/chi) the
% spectrum is flat, above it falls as f^-chi, so the apparent slope increases
% with frequency. The plateau is an additive instrumental noise floor
% (amplifier noise plus residual myogenic activity) which flattens the HIGH
% end. A plateau WITHOUT a knee buys nothing on these data: the two are only
% useful together.
%
% NOTE ON THE EXPONENT. In knee mode chi is the asymptotic high-frequency
% slope, which is steeper than the broadband slope of a fixed-mode fit
% (median 3.5 against 2.1 on the same spectra). Knee-mode and fixed-mode
% exponents are not comparable, and published values are almost always
% fixed-mode.
%
% NOTE ON THE PLATEAU AND COUPLING. The plateau is instrumental, not neural,
% so an oscillation should not be treated as coupled to it. When feeding
% these estimates into a coupling analysis, use the NEURAL aperiodic power
% 10^b/(k+f^chi) as the background, not the total including p. Negligible in
% the alpha band, decisive above 30 Hz.
%
% Fitting minimises the Whittle deviance  D = sum_f [log mu + P/mu], which is
% the Gamma log-likelihood of Welch estimates up to a constant, rather than
% least squares on log10 power.
%
% P     : N x nf linear power spectra
% f     : 1 x nf frequencies (Hz)
% keep  : logical mask over f of frequencies to FIT (default all). Pass
%         ~(f>=6 & f<=16) for the censored fit.
% model : 'fixed' | 'plateau' | 'knee' | 'knee_plateau' (default)
%
% out   : struct with offset, exponent, knee, plateau, knee_freq, dev (N x 1)

if nargin < 3 || isempty(keep),  keep  = true(size(f)); end
if nargin < 4 || isempty(model), model = 'knee_plateau'; end
f = f(:).'; keep = logical(keep(:).');
hasK = any(strcmp(model, {'knee','knee_plateau'}));
hasP = any(strcmp(model, {'plateau','knee_plateau'}));
np = 2 + hasK + hasP;

ff = f(keep);
N  = size(P,1);
off = zeros(N,1); ex = zeros(N,1); kn = zeros(N,1);
pl  = zeros(N,1); dv = zeros(N,1);

o = optimoptions('fmincon','Display','off','Algorithm','interior-point', ...
                 'MaxIterations',600,'MaxFunctionEvaluations',6000, ...
                 'OptimalityTolerance',1e-10,'StepTolerance',1e-12);

for k = 1:N
    PP = P(k,keep);
    b0 = [ones(numel(ff),1) log10(ff(:))] \ log10(PP(:));
    binit = b0(1); chiinit = -b0(2);

    lo = [binit-5, 0.05]; hi = [binit+5, 6];
    if hasK, lo(end+1) = -5;  hi(end+1) = 8;  hi(1) = binit+8; end %#ok<AGROW>
    if hasP
        lo(end+1) = log10(max(min(PP)*1e-4, 1e-18)); %#ok<AGROW>
        hi(end+1) = log10(max(min(PP)*2,    1e-17)); %#ok<AGROW>
    end

    % try no knee and a knee near the low edge of the fit range
    if hasK, lkset = [-4, chiinit*log10(max(ff(1)*1.5,1.5))]; else, lkset = 0; end
    best = inf; bestth = [];
    for lk = lkset
        th0 = [binit chiinit];
        if hasK, th0 = [th0 lk]; th0(1) = binit + max(lk,0); end %#ok<AGROW>
        if hasP, th0 = [th0 log10(max(min(PP)*0.3, 1e-18))]; end %#ok<AGROW>
        th0 = min(max(th0, lo), hi);
        obj = @(th) whittle(th, ff, PP, hasK, hasP);
        th  = fmincon(obj, th0, [],[],[],[], lo, hi, [], o);
        d   = obj(th);
        if d < best, best = d; bestth = th; end
    end

    off(k) = bestth(1); ex(k) = bestth(2); dv(k) = best;
    q = 2;
    if hasK, q = q+1; kn(k) = 10^bestth(q); end
    if hasP, q = q+1; pl(k) = 10^bestth(q); end
end

out = struct('offset',off, 'exponent',ex, 'knee',kn, 'plateau',pl, ...
             'dev',dv, 'model',{model}, ...
             'knee_freq', (kn>0).*kn.^(1./max(ex,eps)));
end

% ---------------------------------------------------------------------
function D = whittle(th, f, P, hasK, hasP)
b = th(1); chi = th(2); q = 2; k = 0; p = 0;
if hasK, q = q+1; k = 10^th(q); end
if hasP, q = q+1; p = 10^th(q); end
mu = max(10.^b ./ (k + f.^chi) + p, realmin);
D  = sum(log(mu) + P./mu);
end
