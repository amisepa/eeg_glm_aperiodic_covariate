% Verify that the "recovery" statistic in simulation_v3.m is the OLS
% intercept of a design whose non-constant columns are all z-scored, and is
% therefore identically mean(y) regardless of the aperiodic estimation
% method. Run: matlab -batch "run('archive/verify_v3_artifact.m')"
rng(42);
f = 2:0.5:40; N = 200;
alpha_mu = 10; alpha_bw = 1.5; alpha_change_true = 0.25;
off0 = 0.8 + 0.6*rand(N,1);
exp0 = 1.3 + 0.3*randn(N,1);
A0   = 0.5 + 0.15*randn(N,1);
off1 = off0 .* (1.2 + 0.4*rand(N,1));
exp1 = exp0 - (0.1 + 0.2*rand(N,1));
A1   = A0 + alpha_change_true;
toSpec_add = @(off,expo,A) off.*(f.^(-expo)) + A.*exp(-0.5*((f-alpha_mu)/alpha_bw).^2);
P0 = zeros(N,numel(f)); P1 = P0;
for k = 1:N
    P0(k,:) = toSpec_add(off0(k),exp0(k),A0(k)) .* exp(0.05*randn(1,numel(f)));
    P1(k,:) = toSpec_add(off1(k),exp1(k),A1(k)) .* exp(0.05*randn(1,numel(f)));
end
amask = f>=8 & f<=12;
a0 = mean(P0(:,amask),2); a1 = mean(P1(:,amask),2);
y  = a1 - a0;

fprintf('\nmean(y) [raw ERSP mean]          = %.6f\n', mean(y));
fprintf('true alpha change                = %.6f\n\n', alpha_change_true);

% Three deliberately different "estimators" of the aperiodic offset:
%   (a) the correct one, (b) pure noise, (c) a constant-ish garbage value
ests = struct('name',{'OLS excl 8-12 Hz','pure random noise','true off0 (oracle)'}, ...
              'off',{[],[],[]},'exp',{[],[],[]});
lf = log10(f(~amask)).'; X = [ones(numel(lf),1) lf];
oh = zeros(N,1); eh = zeros(N,1);
for k = 1:N
    b = X \ log10(P0(k,~amask)).';
    oh(k) = 10^b(1); eh(k) = -b(2);
end
ests(1).off = oh;            ests(1).exp = eh;
ests(2).off = randn(N,1);    ests(2).exp = randn(N,1);
ests(3).off = off0;          ests(3).exp = exp0;

for i = 1:numel(ests)
    Xg = [ones(N,1) zscore(ests(i).off) zscore(ests(i).exp) zscore(a0)];
    b  = Xg \ y;
    fprintf('%-20s  intercept = %.10f   (intercept - mean(y) = %.3e)\n', ...
        ests(i).name, b(1), b(1)-mean(y));
    % the metric v3 reports as "background dependency"
    r = corr(ests(i).off, y);
    % the correlation of a genuinely GLM-adjusted per-trial estimate
    resid = y - Xg(:,2:end)*b(2:end);
    r_adj = corr(ests(i).off, resid);
    fprintf('%-20s  corr(off_hat, raw y) = %+.3f   corr(off_hat, adjusted y) = %+.3e\n\n', ...
        '', r, r_adj);
end
fprintf('corr(TRUE off0, raw y) = %+.3f  <-- the real background dependence\n', corr(off0,y));
