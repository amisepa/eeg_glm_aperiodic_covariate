function out = oof_profile_lambda(P, f, init, grid_, opt)
% Profile the Whittle deviance over the coupling exponent lambda.
%
% For each lambda on the grid, every spectrum is refitted with lambda held
% fixed (oof_fit_coupling) and the deviances are summed. The minimum gives
% lambda_hat; the likelihood-ratio interval is the set of lambda whose
% scaled deviance lies within chi2inv(1-alpha,1)/(2*K) of the minimum, where
% K is the Gamma shape of the spectral estimates (the effective number of
% independent Welch segments), estimated from the residual dispersion.
%
% lambda = 0 : purely additive coupling  (IRASA's linear subtraction)
% lambda = 1 : purely multiplicative     (specparam's log-space Gaussians,
%                                         dB / divisive baseline correction)
%
% P, f, init : as for oof_fit_coupling
% grid_      : lambda values to profile (default 0:0.1:1.2)
% opt        : passed through, plus alpha (0.05) and refine (true) for a
%              parabolic refinement of the minimum

if nargin < 4 || isempty(grid_), grid_ = 0:0.1:1.2; end
if nargin < 5, opt = struct; end
alpha  = getdef(opt, 'alpha', 0.05);
refine = getdef(opt, 'refine', true);

ng = numel(grid_);
dev = zeros(ng,1); Kt = zeros(ng,1); fits = cell(ng,1);
for g = 1:ng
    fits{g} = oof_fit_coupling(P, f, grid_(g), init, opt);
    dev(g)  = sum(fits{g}.dev);
    Kt(g)   = fits{g}.Ktilde;
end

[~, gmin] = min(dev);
lam_hat = grid_(gmin);
if refine && gmin > 1 && gmin < ng
    % parabolic interpolation through the three lowest points
    x = grid_(gmin-1:gmin+1).'; y = dev(gmin-1:gmin+1);
    c = [x.^2 x ones(3,1)] \ y;
    if c(1) > 0, lam_hat = -c(2)/(2*c(1)); end
end

K  = Kt(gmin);
thr = dev(gmin) + chi2inv(1-alpha, 1)/(2*K);
in  = dev <= thr;
if any(in)
    ci = [grid_(find(in,1,'first')) grid_(find(in,1,'last'))];
else
    ci = [NaN NaN];
end

% likelihood-ratio tests against the two canonical models
[~, i0] = min(abs(grid_ - 0)); [~, i1] = min(abs(grid_ - 1));
lrt = @(d0) max(2*K*(d0 - dev(gmin)), 0);
out = struct('lambda', lam_hat, 'ci', ci, 'grid', grid_(:), 'dev', dev, ...
             'K', K, 'fits', {fits}, ...
             'lrt_additive',       lrt(dev(i0)), ...
             'p_additive',         1 - chi2cdf(lrt(dev(i0)), 1), ...
             'lrt_multiplicative', lrt(dev(i1)), ...
             'p_multiplicative',   1 - chi2cdf(lrt(dev(i1)), 1), ...
             'best_fit', fits{gmin});
end

function v = getdef(s, name, dflt)
if isfield(s, name) && ~isempty(s.(name)), v = s.(name); else, v = dflt; end
end
