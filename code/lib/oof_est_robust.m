function out = oof_est_robust(P, f)
% Log-log fit with iteratively reweighted bisquare (Tukey) weights, as
% recommended by Gao et al. (2017) and benchmarked by Kalamala et al. (2026).
N = size(P,1);
off = zeros(N,1); ex = zeros(N,1);
lf = log10(f(:));
for k = 1:N
    b = robustfit(lf, log10(P(k,:)).', 'bisquare');
    off(k) = b(1); ex(k) = -b(2);
end
out = struct('offset', off, 'exponent', ex, 'knee', zeros(N,1));
end
