function out = oof_lambda_pop(a, b, aI, bI, nboot)
% Population estimator of the coupling exponent.
%
% Write the absolute periodic power of spectrum i as
%
%       a_i = c_i * b_i^lambda
%
% where b_i is the aperiodic power in the same band and c_i is the intrinsic
% strength of the oscillatory generator. If c_i is independent of the
% background b_i -- the assumption the field already makes whenever it asks
% for "background-invariant" oscillatory estimates -- then
%
%       log a_i = log c_i + lambda * log b_i
%
% and lambda is just the log-log regression slope of periodic power on
% aperiodic power across trials, epochs or subjects. lambda = 0 means the
% oscillation is additive (absolute power is the background-invariant
% estimand); lambda = 1 means it is multiplicative (relative power is).
%
% Any existing specparam or IRASA output table is enough to compute it.
%
% CAUTION. a_i and b_i are both estimated from the same spectrum, so their
% errors are negatively correlated (power assigned to the peak is power not
% assigned to the background). That biases the naive slope downwards. Pass
% independent split-half replicates aI, bI (e.g. odd vs even epochs) to get
% the instrumented estimate
%
%       lambda_IV = cov(log a, log bI) / cov(log b, log bI)
%
% which is consistent because the two splits' errors are independent.
%
% a, b   : N x 1 periodic and aperiodic band power (same split)
% aI, bI : N x 1 the same quantities from an independent split ([] to skip)
% nboot  : bootstrap replicates for the CI (default 2000)

if nargin < 3, aI = []; end
if nargin < 4, bI = []; end
if nargin < 5 || isempty(nboot), nboot = 2000; end

ok = a > 0 & b > 0;
if ~isempty(bI), ok = ok & bI > 0 & aI > 0; end
a = a(ok); b = b(ok);
if ~isempty(bI), aI = aI(ok); bI = bI(ok); end
la = log(a); lb = log(b);

slope      = @(x,y) sum((x-mean(x)).*(y-mean(y))) / sum((x-mean(x)).^2);
theilsen   = @(x,y) ts_slope(x,y);
out.lambda_ols     = slope(lb, la);
out.lambda_robust  = theilsen(lb, la);

if ~isempty(bI)
    lbI = log(bI); laI = log(aI);
    cxy = @(x,y) mean((x-mean(x)).*(y-mean(y)));
    % symmetric two-split instrument: average both directions
    num = 0.5*(cxy(lbI, la) + cxy(lb, laI));
    den = cxy(lb, lbI);
    out.lambda_iv = num/den;
    out.reliability = den / sqrt(var(lb,1)*var(lbI,1));
else
    out.lambda_iv = NaN; out.reliability = NaN;
end

% bootstrap over observations
n = numel(la);
bs = nan(nboot, 3);
for r = 1:nboot
    idx = randi(n, n, 1);
    bs(r,1) = slope(lb(idx), la(idx));
    bs(r,2) = ts_slope(lb(idx), la(idx));
    if ~isempty(bI)
        cxy = @(x,y) mean((x-mean(x)).*(y-mean(y)));
        num = 0.5*(cxy(lbI(idx), la(idx)) + cxy(lb(idx), laI(idx)));
        bs(r,3) = num / cxy(lb(idx), lbI(idx));
    end
end
out.ci_ols    = prctile(bs(:,1), [2.5 97.5]);
out.ci_robust = prctile(bs(:,2), [2.5 97.5]);
out.ci_iv     = prctile(bs(:,3), [2.5 97.5]);
out.n         = n;
end

function s = ts_slope(x, y)
n = numel(x);
if n > 300                       % subsample pairs for speed on large N
    m = 300; sel = randperm(n, m); x = x(sel); y = y(sel); n = m;
end
[i, j] = find(triu(true(n),1));
d = x(j) - x(i);
keep = abs(d) > eps;
s = median((y(j(keep)) - y(i(keep))) ./ d(keep));
end
