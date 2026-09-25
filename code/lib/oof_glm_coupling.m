function out = oof_glm_coupling(a, b, cond, trial, lambda_fixed)
% Coupling GLM for periodic band power.
%
% Stage 1 (lambda): with cond and trial omitted, or with lambda_fixed empty
% and a single condition supplied, fit
%
%       a_i ~ Gamma,   log E[a_i] = beta0 + lambda * log b_i
%
% The slope on log aperiodic power IS the coupling exponent: lambda = 0 is
% an additive oscillation (absolute power is the background-invariant
% estimand), lambda = 1 a multiplicative one (relative power is).
%
% Stage 2 (effect): with cond supplied and lambda_fixed given, fit
%
%       log E[a_ik] = beta0 + delta*cond_k + u_i,  offset = lambda*log b_ik
%
% so that delta is the condition effect on the INTRINSIC oscillatory
% strength, free of the background. lambda is held fixed at its stage-1
% value because the within-trial variation in log b is small and its
% estimation error is shared with the outcome.
%
% A Gamma family with a log link is used rather than OLS on log(a) because
% Welch power estimates are Gamma distributed: the log link models the log
% of the MEAN, so no Jensen bias is introduced by logging noisy estimates,
% and the variance-proportional-to-mean-squared structure is the right one.
%
% a, b   : N x 1 (stage 1) or (N*K) x 1 (stage 2) band power, a > 0
% cond   : (N*K) x 1 condition code, [] for stage 1
% trial  : (N*K) x 1 trial/subject id, [] for no random effect
% lambda_fixed : scalar for stage 2, [] for stage 1

if nargin < 3, cond  = []; end
if nargin < 4, trial = []; end
if nargin < 5, lambda_fixed = []; end

ok = a > 0 & b > 0 & isfinite(a) & isfinite(b);
a = a(ok); b = b(ok);
if ~isempty(cond),  cond  = cond(ok);  end
if ~isempty(trial), trial = trial(ok); end
lb = log(b);

if isempty(lambda_fixed)
    % ---- stage 1: estimate lambda ---------------------------------
    m = fitglm(lb, a, 'linear', 'Distribution', 'gamma', 'Link', 'log');
    out.lambda    = m.Coefficients.Estimate(2);
    out.lambda_se = m.Coefficients.SE(2);
    out.lambda_ci = [out.lambda - 1.96*out.lambda_se, out.lambda + 1.96*out.lambda_se];
    out.p_additive       = 2*normcdf(-abs(out.lambda    )/out.lambda_se);
    out.p_multiplicative = 2*normcdf(-abs(out.lambda - 1)/out.lambda_se);
    out.model = m;
    out.n = numel(a);
else
    % ---- stage 2: condition effect at fixed lambda ------------------
    off = lambda_fixed * lb;
    T = table(a, categorical(cond), 'VariableNames', {'a','cond'});
    T.offs = off;
    if isempty(trial)
        m = fitglm(T, 'a ~ 1 + cond', 'Distribution', 'gamma', 'Link', 'log', ...
                   'Offset', T.offs);
        cf = m.Coefficients;
    else
        T.trial = categorical(trial);
        m = fitglme(T, 'a ~ 1 + cond + (1|trial)', 'Distribution', 'gamma', ...
                    'Link', 'log', 'Offset', T.offs);
        cf = table(m.Coefficients.Estimate, m.Coefficients.SE, m.Coefficients.pValue, ...
                   'VariableNames', {'Estimate','SE','pValue'});
        cf.Properties.RowNames = m.CoefficientNames;
    end
    out.delta    = cf.Estimate(end);
    out.delta_se = cf.SE(end);
    out.delta_ci = [out.delta - 1.96*out.delta_se, out.delta + 1.96*out.delta_se];
    out.p        = cf.pValue(end);
    out.lambda   = lambda_fixed;
    out.model    = m;
    out.n        = numel(a);
end
end
