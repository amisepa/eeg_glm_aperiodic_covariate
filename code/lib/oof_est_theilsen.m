function out = oof_est_theilsen(P, f)
% Theil-Sen estimator in log-log space: the exponent is the median of the
% slopes of all frequency pairs, the offset the median of the residual
% intercepts. Proposed in this form by E. Masherov (sccn/OneOverF #12) as a
% tuning-parameter-free estimator; it is the classical Theil (1950) / Sen
% (1968) estimator, with a 29.3% asymptotic breakdown point.
N  = size(P,1);
lf = log10(f(:));
nf = numel(lf);
[ii, jj] = find(triu(true(nf), 1));
dlf = lf(jj) - lf(ii);
off = zeros(N,1); ex = zeros(N,1);
for k = 1:N
    lp = log10(P(k,:)).';
    s  = median((lp(jj) - lp(ii)) ./ dlf);
    ex(k)  = -s;
    off(k) = median(lp - s*lf);
end
out = struct('offset', off, 'exponent', ex, 'knee', zeros(N,1));
end
