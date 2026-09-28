function out = oof_lambda_curve(la, lb, x, covariates, opt)
% Effect on ln a - lambda*ln b as a function of the coupling exponent.
%
% With a the periodic and b the aperiodic power in a band, periodic power
% under an assumed lambda is y = ln a - lambda*ln b: lambda = 0 subtracts
% the background in linear power (as IRASA does), lambda = 1 divides it out
% (as specparam and dB baselines do). Any linear effect on y is linear in
% lambda,
%
%       s(lambda) = s_a - lambda*s_b
%
% with s_a and s_b the same effect computed on ln a and on ln b, and it
% changes sign at the crossover lambda* = s_a/s_b. A crossover outside
% [0 1] means the conclusion holds under either rule; one inside [0 1]
% means it depends on the assumption. lambda* is large whenever the
% background barely changes with the predictor, so look at s_b first: when
% its interval includes zero, lambda* is unbounded and its CI and HDI are
% not informative (out.lam_star_bounded is false).
%
% Between subjects: la, lb are ln a and ln b per subject and x the
% predictor; the effect is the least-squares coefficient of x, adjusted
% for the covariates. Within subjects: x is empty and la, lb are paired
% differences (condition 2 - condition 1) of ln a and ln b; the effect is
% the mean difference.
%
% ln a does not exist for a <= 0. Non-finite rows raise an error unless
% opt.dropna is true, because dropping them selects on the periodic
% estimate, which grows with the background. Report out.n_dropped.
%
% la, lb     : N x 1
% x          : N x 1 predictor, [] for a within-subject contrast
% covariates : N x k nuisance covariates, [] for none
% opt        : struct, fields grid (0:0.05:1), nboot (2000), seed (0),
%              dropna (false), draws ([]: drawn here; or a struct with
%              idx, N x nboot row indices, and w, N x nboot weights
%              summing to one per column, to reuse given draws)
%
% out : struct with n, n_dropped, design, s_a, s_b, lam_star, ci (pairs-
%       bootstrap percentile interval of lam_star), hdi (Bayesian-bootstrap
%       95% highest-density interval), p_cross_in_01 (share of Bayesian
%       draws with lam_star in [0 1]), s_a_ci, s_b_ci, lam_star_bounded,
%       grid, curve (estimate, lower, upper for each grid value), lam0 and
%       lam1 ([estimate lower upper] at lambda = 0 and 1) and verdict:
%       'holds under both' (both intervals exclude zero, same sign),
%       'reverses' (both exclude zero, opposite signs), 'depends on lambda'
%       (only one does) or 'null under both'.
%
% Same computation as lambdacurve.lambda_curve (Python, lambdacurve/ in
% this repository). Percentiles interpolate as numpy's do, so the same
% draws give the same intervals; MATLAB's own draws differ from numpy's.

if nargin < 3, x = []; end
if nargin < 4, covariates = []; end
if nargin < 5, opt = struct; end
grid_  = getdef(opt, 'grid', 0:0.05:1);
nboot  = getdef(opt, 'nboot', 2000);
seed   = getdef(opt, 'seed', 0);
dropna = getdef(opt, 'dropna', false);
draws  = getdef(opt, 'draws', []);

la = la(:); lb = lb(:);
if numel(la) ~= numel(lb), error('la and lb must have the same length'); end
if isempty(x) && ~isempty(covariates), error('covariates need a predictor x'); end
between = ~isempty(x);
if isvector(covariates) && numel(covariates) == numel(la), covariates = covariates(:); end
X = ones(numel(la), 1);
if between, X = [X, x(:), covariates]; end
if size(X, 1) ~= numel(la), error('x and covariates need one row per subject'); end
ok = isfinite(la) & isfinite(lb) & all(isfinite(X), 2);
n_dropped = sum(~ok);
if n_dropped > 0 && ~dropna
    error('oof_lambda_curve:nonfinite', ['%d of %d rows are not finite (ln a of ' ...
          'a <= 0?). Dropping them selects on the periodic estimate; set ' ...
          'opt.dropna = true to drop them anyway and report n_dropped.'], ...
          n_dropped, numel(ok));
end
Y = [la(ok) lb(ok)];
X = X(ok, :);
n = size(Y, 1);
if n < size(X, 2) + 2, error('too few complete rows'); end

if between
    c = X \ Y;  s = c(2, :);
else
    s = (ones(1, n) / n) * Y;
end
s_a = s(1); s_b = s(2);

% pairs bootstrap and Bayesian bootstrap (Dirichlet(1,...,1) weights)
if isempty(draws)
    rs = RandStream('twister', 'Seed', seed);
else
    nboot = size(draws.idx, 2);
end
boot = zeros(nboot, 2); bayes = zeros(nboot, 2);
for k = 1:nboot
    if isempty(draws)
        i = randi(rs, n, n, 1);
        e = -log(rand(rs, n, 1));
        w = e / sum(e);
    else
        i = draws.idx(:, k); w = draws.w(:, k);
    end
    if between
        c = X(i, :) \ Y(i, :);          boot(k, :) = c(2, :);
        WX = X .* w;
        c = (X' * WX) \ (WX' * Y);      bayes(k, :) = c(2, :);
    else
        boot(k, :) = mean(Y(i, :), 1);
        bayes(k, :) = w' * Y;
    end
end
ls_boot  = boot(:, 1) ./ boot(:, 2);
ls_bayes = bayes(:, 1) ./ bayes(:, 2);

grid_ = grid_(:);
curve = zeros(numel(grid_), 3);
for g = 1:numel(grid_)
    curve(g, :) = effect(grid_(g));
end

out.n = n;
out.n_dropped = n_dropped;
if between, out.design = 'between'; else, out.design = 'within'; end
out.s_a = s_a;
out.s_b = s_b;
if s_b ~= 0, out.lam_star = s_a / s_b; else, out.lam_star = NaN; end
out.ci  = pct(ls_boot(~isnan(ls_boot)), [2.5 97.5]);
out.hdi = hdi(ls_bayes, 0.95);
out.p_cross_in_01 = mean(ls_bayes >= 0 & ls_bayes <= 1);
out.s_a_ci = pct(boot(:, 1), [2.5 97.5]);
out.s_b_ci = pct(boot(:, 2), [2.5 97.5]);
out.lam_star_bounded = out.s_b_ci(1) > 0 || out.s_b_ci(2) < 0;
out.grid  = grid_;
out.curve = curve;
out.lam0  = effect(0);
out.lam1  = effect(1);
out.verdict = verdict(out.lam0, out.lam1);

    function r = effect(lam)
        r = [s_a - lam * s_b, pct(boot(:, 1) - lam * boot(:, 2), [2.5 97.5])];
    end
end

function v = verdict(e0, e1)
sig0 = e0(2) > 0 || e0(3) < 0;
sig1 = e1(2) > 0 || e1(3) < 0;
if sig0 && sig1
    if (e0(2) > 0) == (e1(2) > 0), v = 'holds under both'; else, v = 'reverses'; end
elseif sig0 || sig1
    v = 'depends on lambda';
else
    v = 'null under both';
end
end

function v = pct(x, p)
% Percentiles with linear interpolation between order statistics, as
% numpy.percentile computes them (MATLAB's prctile interpolates differently).
x = sort(x(:));
m = numel(x);
v = nan(1, numel(p));
if m == 0, return; end
for j = 1:numel(p)
    h  = (m - 1) * (p(j) / 100);
    lo = floor(h);
    t  = h - lo;
    a  = x(min(lo, m - 1) + 1);
    b  = x(min(lo + 1, m - 1) + 1);
    d  = b - a;
    if t >= 0.5, v(j) = b - d * (1 - t); else, v(j) = a + d * t; end
end
end

function iv = hdi(x, mass)
% Shortest interval containing mass of the finite values of x.
x = sort(x(isfinite(x)));
m = numel(x);
if m < 2, iv = [NaN NaN]; return; end
k = floor(mass * m);
[~, i] = min(x(k+1:m) - x(1:m-k));
iv = [x(i), x(i+k)];
end

function v = getdef(s, name, dflt)
if isfield(s, name) && ~isempty(s.(name)), v = s.(name); else, v = dflt; end
end
