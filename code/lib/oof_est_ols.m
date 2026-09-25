function out = oof_est_ols(P, f, exclude, fitonly)
% Log-log OLS aperiodic fit.
%   exclude : logical mask over f of frequencies to DROP (default none)
%   fitonly : if true, `exclude` is instead the mask of frequencies to KEEP
% Returns offset in log10 power units and a positive exponent.
if nargin < 3 || isempty(exclude), exclude = false(size(f)); end
if nargin < 4 || isempty(fitonly), fitonly = false; end
if fitonly, m = logical(exclude); else, m = ~logical(exclude); end
lf = log10(f(m)).';  X = [ones(sum(m),1) lf];
LP = log10(P(:,m)).';
B  = X \ LP;
out = struct('offset', B(1,:).', 'exponent', -B(2,:).', 'knee', zeros(size(P,1),1));
end
