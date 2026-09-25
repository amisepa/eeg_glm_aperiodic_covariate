function out = oof_est_specparam(P, f, opt)
% Faithful MATLAB port of the specparam / FOOOF fitting procedure
% (Donoghue et al., 2020; fooof 1.1 reference implementation).
%
% NOTE ON THE MODEL. specparam fits Gaussians that are ADDITIVE IN log10
% POWER on top of the aperiodic component:
%       log10 P(f) = ap(f) + sum_n N(f; cf_n, h_n, sd_n)
% Exponentiating, P(f) = L(f) * 10^(sum_n N(f)), i.e. the periodic part
% MULTIPLIES the aperiodic part. In the coupling family of oof_model_psd
% this is lambda = 1. Peak heights are therefore relative (log10 power
% ratios), not absolute power, and they inherit the aperiodic level whenever
% the true coupling is additive. out.peak_abs converts heights to absolute
% power at the peak centre for comparison with additive-model estimands.
%
% P   : N x nf linear power spectra
% f   : 1 x nf frequencies (Hz), already restricted to the fit range
% opt : struct, fields (defaults in brackets)
%         peak_width_limits [0.5 12]  min/max FWHM-equivalent width, Hz
%         max_n_peaks       [6]
%         min_peak_height   [0]       log10 power
%         peak_threshold    [2]       in SD of the flattened spectrum
%         aperiodic_mode    ['fixed'] or 'knee'
%
% out : struct with offset, exponent, knee (N x 1) and peaks (1 x N cell,
%       each n_peaks x 3 = [cf height_log10 sd]), plus peak_abs.

if nargin < 3, opt = struct; end
pwl    = getdef(opt, 'peak_width_limits', [0.5 12]);
maxnp  = getdef(opt, 'max_n_peaks', 6);
mph    = getdef(opt, 'min_peak_height', 0);
pthr   = getdef(opt, 'peak_threshold', 2);
apmode = getdef(opt, 'aperiodic_mode', 'fixed');
bw_std_edge = 1.0; gauss_overlap_thresh = 0.75;

f = f(:).'; lf = log10(f);
fres = median(diff(f));
N = size(P,1);
off = zeros(N,1); ex = zeros(N,1); kn = zeros(N,1);
peaks = cell(1,N); peak_abs = cell(1,N);
knee = strcmpi(apmode, 'knee');

apfun = @(p, ff) ap_eval(p, ff, knee);
lb = -inf(1, 2+knee); ub = inf(1, 2+knee);
o = optimoptions('lsqcurvefit', 'Display', 'off', ...
                 'MaxFunctionEvaluations', 5000, 'MaxIterations', 1000, ...
                 'FunctionTolerance', 1e-10, 'StepTolerance', 1e-10);

for k = 1:N
    lp = log10(P(k,:));

    % ---- 1. initial aperiodic fit -----------------------------------
    p0  = init_ap_guess(lp, lf, knee);
    ap0 = lsqcurvefit(apfun, p0, f, lp, lb, ub, o);

    % ---- 2. robust aperiodic fit: refit on points not above the fit --
    flat = lp - apfun(ap0, f);
    mask = flat <= 0;
    if sum(mask) >= 4
        ap0 = lsqcurvefit(apfun, ap0, f(mask), lp(mask), lb, ub, o);
    end

    % ---- 3. greedy peak guesses on the flattened spectrum -----------
    flatspec = lp - apfun(ap0, f);
    fi = flatspec; guess = zeros(0,3);
    for n = 1:maxnp
        [mx, mi] = max(fi);
        if mx <= pthr*std(fi) || mx <= mph, break; end
        gsd = halfheight_std(fi, mi, fres, pwl);
        guess = [guess; f(mi) mx gsd]; %#ok<AGROW>
        fi = fi - gauss_eval(guess(end,:), f);
    end
    guess = drop_edge(guess, f, bw_std_edge);
    guess = drop_overlap(guess, gauss_overlap_thresh);

    % ---- 4. joint fit of all guessed Gaussians ----------------------
    if ~isempty(guess)
        ng  = size(guess,1);
        glb = [guess(:,1) - 2*bw_std_edge*guess(:,3), zeros(ng,1), repmat(pwl(1)/2, ng, 1)];
        gub = [guess(:,1) + 2*bw_std_edge*guess(:,3),  inf(ng,1), repmat(pwl(2)/2, ng, 1)];
        gp  = lsqcurvefit(@(p, ff) gauss_eval(reshape(p, [], 3), ff), guess(:), ...
                          f, flatspec, glb(:), gub(:), o);
        gp  = reshape(gp, [], 3);
    else
        gp = zeros(0,3);
    end

    % ---- 5. refit aperiodic on the peak-removed spectrum ------------
    lp_rm = lp - gauss_eval(gp, f);
    ap1   = lsqcurvefit(apfun, ap0, f, lp_rm, lb, ub, o);

    off(k) = ap1(1); ex(k) = ap1(end);
    if knee, kn(k) = ap1(2); end
    peaks{k} = gp;
    if isempty(gp)
        peak_abs{k} = zeros(0,3);
    else
        Lcf = 10.^apfun(ap1, gp(:,1).');
        peak_abs{k} = [gp(:,1), (10.^gp(:,2) - 1).*Lcf(:), gp(:,3)];
    end
end
out = struct('offset', off, 'exponent', ex, 'knee', kn, ...
             'peaks', {peaks}, 'peak_abs', {peak_abs});
end

% ---------------------------------------------------------------------
function v = getdef(s, name, dflt)
if isfield(s, name) && ~isempty(s.(name)), v = s.(name); else, v = dflt; end
end

function y = ap_eval(p, ff, knee)
if knee
    y = p(1) - log10(p(2) + ff.^p(3));
else
    y = p(1) - p(2)*log10(ff);
end
end

function p0 = init_ap_guess(lp, lf, knee)
ex0 = abs((lp(end)-lp(1))/(lf(end)-lf(1)));
if knee, p0 = [lp(1) 0 ex0]; else, p0 = [lp(1) ex0]; end
end

function y = gauss_eval(gp, ff)
y = zeros(1, numel(ff));
for n = 1:size(gp,1)
    y = y + gp(n,2)*exp(-0.5*((ff-gp(n,1))./gp(n,3)).^2);
end
end

function gsd = halfheight_std(fi, mi, fres, pwl)
hh = 0.5*fi(mi);
le = find(fi(1:mi) <= hh, 1, 'last');
ri = find(fi(mi:end) <= hh, 1, 'first');
ds = [];
if ~isempty(le), ds(end+1) = mi - le; end
if ~isempty(ri), ds(end+1) = ri - 1;  end
if isempty(ds)
    gsd = mean(pwl)/2;
else
    fwhm = min(ds)*2*fres;
    gsd  = fwhm/(2*sqrt(2*log(2)));
end
gsd = min(max(gsd, pwl(1)/2), pwl(2)/2);
end

function g = drop_edge(g, f, bw_std_edge)
if isempty(g), return; end
keep = (g(:,1) - bw_std_edge*g(:,3) > f(1)) & (g(:,1) + bw_std_edge*g(:,3) < f(end));
g = g(keep,:);
end

function g = drop_overlap(g, thr)
if size(g,1) < 2, return; end
g = sortrows(g, 1);
drop = false(size(g,1),1);
for i = 1:size(g,1)-1
    if abs(g(i+1,1)-g(i,1)) < thr*max(g(i,3), g(i+1,3))
        if g(i,2) < g(i+1,2), drop(i) = true; else, drop(i+1) = true; end
    end
end
g = g(~drop,:);
end
