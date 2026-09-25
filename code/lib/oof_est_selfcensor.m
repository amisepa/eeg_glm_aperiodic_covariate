function out = oof_est_selfcensor(P, f, niter)
% CAUTION: this is NOT the censored regression of Kalamala et al. (2026).
% Theirs excludes a FIXED, a-priori frequency window (see oof_est_censored).
% This variant instead refits iteratively on whichever points currently lie at
% or below the fit, to convergence. It is included only because it is an
% obvious-looking idea that fails badly: each pass keeps the lower half of the
% residuals, so the fit walks downwards, and in our simulations it
% underestimates aperiodic band power by about 0.28 dex (~47%) and displaces
% the downstream coupling estimate from 0 to 0.45.
if nargin < 3 || isempty(niter), niter = 10; end
N = size(P,1);
lf = log10(f(:)); X0 = [ones(numel(lf),1) lf];
off = zeros(N,1); ex = zeros(N,1);
for k = 1:N
    lp = log10(P(k,:)).';
    b  = X0 \ lp;
    for it = 1:niter
        keep = lp <= X0*b;
        if sum(keep) < 4, break; end
        bn = X0(keep,:) \ lp(keep);
        if max(abs(bn-b)) < 1e-8, b = bn; break; end
        b = bn;
    end
    off(k) = b(1); ex(k) = -b(2);
end
out = struct('offset', off, 'exponent', ex, 'knee', zeros(N,1));
end
