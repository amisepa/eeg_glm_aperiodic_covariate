function out = oof_fit_coupling(P, f, lambda, init, opt)
% Fit the generalized periodic/aperiodic model at a FIXED coupling exponent
%
%       mu_i(f) = L_i(f) + sum_n a_in * G(f; cf_in, bw_in) * L_i(f)^lambda
%       L_i(f)  = 10^off_i / (knee_i + f^chi_i)
%
% by minimizing the Whittle deviance, which is the correct log-likelihood
% (up to a constant) for Welch/periodogram power estimates, whose sampling
% distribution is Gamma with mean mu and shape K:
%
%       D(theta) = sum_f [ log mu(f) + Phat(f)/mu(f) ]
%
% Fitting in this space -- rather than by least squares on log10 power, as
% specparam does, or on linear power, as IRASA's subtraction implies --
% removes the arbitrary choice of error space: the heteroscedasticity of
% spectral estimates (SD proportional to the mean) is handled by the
% likelihood instead of by a variance-stabilizing transform that silently
% fixes lambda.
%
% P      : N x nf linear power spectra
% f      : 1 x nf frequencies (Hz)
% lambda : scalar coupling exponent held fixed during the fit
% init   : struct array (1 x N) or scalar struct with fields
%            offset, exponent, knee, peaks (npk x 3 = [cf amp_abs bw])
%          amp_abs is ABSOLUTE peak power at the peak centre; it is
%          rescaled internally so the initial height matches at any lambda.
%          On output, peaks holds the RAW coefficient a of G(f)*L(f)^lambda
%          and peaks_abs holds the absolute peak power a*L(cf)^lambda.
% opt    : struct, fields knee (false), maxiter (400), display ('off')
%
% out    : struct with offset, exponent, knee (N x 1), peaks (1 x N cell),
%          dev (N x 1 Whittle deviance), mu (N x nf fitted spectra),
%          lambda, and Ktilde (moment estimate of the Gamma shape).

if nargin < 5, opt = struct; end
useknee = getdef(opt, 'knee', false);
maxiter = getdef(opt, 'maxiter', 400);
disp_   = getdef(opt, 'display', 'off');

f = f(:).'; nf = numel(f); N = size(P,1);
if numel(init) == 1 && N > 1, init = repmat(init, 1, N); end

o = optimoptions('fmincon', 'Display', disp_, 'Algorithm', 'interior-point', ...
                 'MaxIterations', maxiter, 'MaxFunctionEvaluations', 50*maxiter, ...
                 'OptimalityTolerance', 1e-8, 'StepTolerance', 1e-10, ...
                 'SpecifyObjectiveGradient', false);

off = zeros(N,1); ex = zeros(N,1); kn = zeros(N,1);
dev = zeros(N,1); mu = zeros(N,nf); pk = cell(1,N); pkabs = cell(1,N);

for k = 1:N
    ini = init(min(k, numel(init)));
    npk = size(ini.peaks, 1);

    % ---- pack: [off, chi, (log10 knee), (cf, log10 amp, bw) x npk] ----
    L0 = oof_aperiodic(f, ini.offset, ini.exponent, ini.knee);
    th0 = [ini.offset, ini.exponent];
    lo  = [ini.offset-3, 0.05];
    hi  = [ini.offset+3, 6];
    if useknee
        th0 = [th0, log10(max(ini.knee, 1e-3))]; %#ok<AGROW>
        lo  = [lo, -4]; hi = [hi, 6]; %#ok<AGROW>
    end
    for n = 1:npk
        cf0 = ini.peaks(n,1); bw0 = ini.peaks(n,3);
        % rescale the absolute amplitude so the peak height is preserved
        Lcf = interp1(f, L0, cf0, 'linear', 'extrap');
        a0  = max(ini.peaks(n,2), 1e-12) / max(Lcf, eps)^lambda;
        th0 = [th0, cf0, log10(a0), bw0]; %#ok<AGROW>
        lo  = [lo, max(cf0-4, f(1)), log10(a0)-4, 0.25]; %#ok<AGROW>
        hi  = [hi, min(cf0+4, f(end)), log10(a0)+4, 8]; %#ok<AGROW>
    end

    obj = @(th) whittle_dev(th, f, P(k,:), lambda, npk, useknee);
    th  = fmincon(obj, th0, [], [], [], [], lo, hi, [], o);

    [dev(k), mu(k,:)] = obj(th);
    off(k) = th(1); ex(k) = th(2);
    if useknee, kn(k) = 10^th(3); end
    q = 2 + useknee;
    pp = zeros(npk,3); pa = zeros(npk,3);
    Lk = oof_aperiodic(f, off(k), ex(k), kn(k));
    for n = 1:npk
        cf = th(q+1); a = 10^th(q+2); bw = th(q+3); q = q + 3;
        Lcf = interp1(f, Lk, cf, 'linear', 'extrap');
        pp(n,:) = [cf, a, bw];              % raw coefficient of G(f)*L(f)^lambda
        pa(n,:) = [cf, a*Lcf^lambda, bw];   % absolute peak power at cf
    end
    pk{k} = pp; pkabs{k} = pa;
end

% moment estimate of the Gamma shape K from the standardized residuals
r = P ./ max(mu, eps);
Ktilde = 1 / max(var(r(:)), eps);

out = struct('offset', off, 'exponent', ex, 'knee', kn, 'peaks', {pk}, ...
             'peaks_abs', {pkabs}, 'dev', dev, 'mu', mu, 'lambda', lambda, ...
             'Ktilde', Ktilde);
end

% ---------------------------------------------------------------------
function [D, mu] = whittle_dev(th, f, Phat, lambda, npk, useknee)
off = th(1); chi = th(2);
if useknee, kn = 10^th(3); q = 3; else, kn = 0; q = 2; end
L = oof_aperiodic(f, off, chi, kn);
Pper = zeros(1, numel(f));
for n = 1:npk
    cf = th(q+1); a = 10^th(q+2); bw = th(q+3); q = q + 3;
    Pper = Pper + a * exp(-0.5*((f-cf)./bw).^2);
end
mu = L + Pper .* (L.^lambda);
mu = max(mu, realmin);
D  = sum(log(mu) + Phat./mu);
end

function v = getdef(s, name, dflt)
if isfield(s, name) && ~isempty(s.(name)), v = s.(name); else, v = dflt; end
end
