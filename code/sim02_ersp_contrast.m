% SIM 2 -- The Gyurkovics scenario, and what the GLM coefficient means.
%
% Setup: every trial has the SAME intrinsic oscillatory change between a
% baseline and a post-event window, while the aperiodic background differs
% across trials AND shifts between windows. Two background shifts:
%   scenario = 'steepen'  exponent up by 0.10-0.20, offset unchanged: the
%                         post-stimulus steepening reported by Gyurkovics
%                         et al. (2022); lowers the background at alpha
%   scenario = 'flatten'  offset up by 0.10-0.18, exponent down by
%                         0.10-0.20; raises the background at alpha
% Set `scenario` in the workspace before running (default 'steepen').
% We compare five ways of quantifying the oscillatory change:
%
%   raw     dPow    = mean(P1) - mean(P0)              in the band
%   dB      ddB     = 10*log10(mean(P1)/mean(P0))
%   sub     da      = a1 - a0        (aperiodic removed, absolute)
%   rel     dloga   = log(a1) - log(a0)
%   GLM     dlog a_i = beta0 + lambda * dlog b_i + e_i
%
% The last line is the GLM Cedric proposed, with two changes: the outcome is
% the periodic part (not raw band power), and the covariate coefficient is
% not a nuisance -- it IS the coupling exponent lambda of the generative
% model. beta0 is the background-invariant oscillatory effect.
%
% Writes results/sim02_<scenario>.mat

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'lib'));
outdir = fullfile(fileparts(here), 'results');
if ~exist(outdir,'dir'), mkdir(outdir); end
rng(303);
if ~exist('scenario', 'var'), scenario = 'steepen'; end

srate = 500; dur = 20; nsamp = srate*dur;   % 20 s per window
fmod = 0.25:0.25:120; frng = [2 45]; band = [8 12];
N = 150;
lam_set = [0 1];

S = struct();
for li = 1:numel(lam_set)
    lt = lam_set(li);

    % ---- per-trial ground truth ------------------------------------
    off0 = 0.4 + 1.4*rand(N,1);            % background varies across trials
    exp0 = 1.30 + 0.25*randn(N,1);
    switch scenario                      % background shift, trial specific
        case 'steepen'
            off1 = off0;
            exp1 = exp0 + (0.10 + 0.10*rand(N,1));
        case 'flatten'
            off1 = off0 + (0.10 + 0.08*rand(N,1));
            exp1 = exp0 - (0.10 + 0.10*rand(N,1));
        otherwise
            error('unknown scenario %s', scenario);
    end
    cf = 10; bw = 1.5;

    % The intrinsic oscillator strength must be drawn INDEPENDENTLY of the
    % trial's own background, otherwise the coupling is built in by the
    % simulation rather than by lambda. Scale by a single reference level so
    % that SNR is comparable across lambda.
    Lref = 10^median(off0) / cf^median(exp0);
    c0 = 0.8 * Lref^(1-lt) * 10.^(0.20*randn(N,1));      % intrinsic strength
    dlogc_true = log(1.60);                              % SAME +60% on every trial
    c1 = c0 * exp(dlogc_true);

    P0 = []; P1 = [];
    for i = 1:N
        a0s = struct('offset', off0(i), 'exponent', exp0(i), 'knee', 0);
        a1s = struct('offset', off1(i), 'exponent', exp1(i), 'knee', 0);
        s0 = oof_model_psd(fmod, a0s, struct('cf',cf,'bw',bw,'amp',c0(i)), lt);
        s1 = oof_model_psd(fmod, a1s, struct('cf',cf,'bw',bw,'amp',c1(i)), lt);
        x0 = oof_synth_signal(s0, fmod, srate, nsamp);
        x1 = oof_synth_signal(s1, fmod, srate, nsamp);
        [p0, fp] = oof_welch(x0, srate, 4, 0.5, frng);
        [p1, ~ ] = oof_welch(x1, srate, 4, 0.5, frng);
        if isempty(P0), P0 = zeros(N,numel(fp)); P1 = P0; end
        P0(i,:) = p0; P1(i,:) = p1;
    end
    m = fp >= band(1) & fp <= band(2);

    % ---- aperiodic estimation (OLS excluding 6-14 Hz) ---------------
    e0 = oof_est_ols(P0, fp, fp>=6 & fp<=14);
    e1 = oof_est_ols(P1, fp, fp>=6 & fp<=14);
    b0 = zeros(N,1); b1 = zeros(N,1);
    for i = 1:N
        b0(i) = mean(oof_aperiodic(fp(m), e0.offset(i), e0.exponent(i), 0));
        b1(i) = mean(oof_aperiodic(fp(m), e1.offset(i), e1.exponent(i), 0));
    end
    tot0 = mean(P0(:,m),2); tot1 = mean(P1(:,m),2);
    a0 = tot0 - b0; a1 = tot1 - b1;
    ok = a0 > 0 & a1 > 0;

    % ---- the five estimators ----------------------------------------
    dPow  = tot1 - tot0;
    ddB   = 10*log10(tot1./tot0);
    da    = a1 - a0;
    dloga = log(a1) - log(a0);
    dlogb = log(b1) - log(b0);

    % lambda is estimated from the BASELINE LEVEL relation across trials,
    % where the leverage is the 1.4-decade spread of backgrounds. It is NOT
    % estimated from the within-trial change: the true variance of dlog b is
    % tiny and its estimation error is shared with the outcome, so that
    % regression is a textbook errors-in-variables trap.
    Xl  = [ones(sum(ok),1) log(b0(ok))];
    bl  = Xl \ log(a0(ok));
    lam_hat = bl(2);
    se  = sqrt(sum((log(a0(ok))-Xl*bl).^2)/(sum(ok)-2) * diag(inv(Xl'*Xl)));
    % background-invariant contrast: the change in intrinsic strength,
    % log c = log a - lambda*log b
    dlogc_hat = (log(a1(ok)) - lam_hat*log(b1(ok))) - ...
                (log(a0(ok)) - lam_hat*log(b0(ok)));
    beta0 = mean(dlogc_hat);

    % Gamma GLM with a log link: lambda from the baseline levels, then the
    % condition effect with lambda*log b carried as a fixed offset.
    g1 = oof_glm_coupling(a0(ok), b0(ok));
    idx  = find(ok);
    astack = [a0(ok); a1(ok)];
    bstack = [b0(ok); b1(ok)];
    cstack = [zeros(sum(ok),1); ones(sum(ok),1)];
    tstack = [idx; idx];
    g2 = oof_glm_coupling(astack, bstack, cstack, tstack, g1.lambda);

    % background dependence of each estimator, against the TRUE baseline
    % background level (not an estimate of it)
    lb0 = log(10.^off0 ./ cf.^exp0);
    dep = @(v) corr(lb0(ok), v(ok));

    fprintf('\n===== true lambda = %.1f, true dlog c = %+.3f (x%.2f) =====\n', ...
        lt, dlogc_true, exp(dlogc_true));
    fprintf('%-28s %12s %14s\n', 'estimator', 'mean effect', 'r with background');
    fprintf('%-28s %12.4f %14.3f\n', 'raw band power change',  mean(dPow(ok)),  dep(dPow));
    fprintf('%-28s %12.4f %14.3f\n', 'dB change',              mean(ddB(ok)),   dep(ddB));
    fprintf('%-28s %12.4f %14.3f\n', 'periodic, absolute',     mean(da(ok)),    dep(da));
    fprintf('%-28s %12.4f %14.3f\n', 'periodic, log ratio',    mean(dloga(ok)), dep(dloga));
    fprintf('%-28s %12.4f %14.3f   lambda_hat = %.3f (SE %.3f)\n', ...
        'lambda-GLM (plug-in log)', beta0, corr(lb0(ok), dlogc_hat), lam_hat, se(2));
    fprintf('%-28s %12.4f %14s   lambda_hat = %.3f (SE %.3f)\n', ...
        'lambda-GLM (Gamma, log link)', g2.delta, '-', g1.lambda, g1.lambda_se);
    fprintf('   target for the log-ratio estimators: %+.3f\n', dlogc_true);

    S(li).lambda_true = lt; S(li).lam_hat = lam_hat; S(li).beta0 = beta0;
    S(li).se = se; S(li).dlogc_true = dlogc_true;
    S(li).dPow = dPow; S(li).ddB = ddB; S(li).da = da; S(li).dloga = dloga;
    S(li).dlogb = dlogb; S(li).lb0 = lb0; S(li).ok = ok;
    S(li).dlogc_hat = dlogc_hat; S(li).g1 = g1; S(li).g2 = g2;
    S(li).a0 = a0; S(li).a1 = a1; S(li).b0 = b0; S(li).b1 = b1;
end
save(fullfile(outdir, ['sim02_' scenario '.mat']), 'S', 'lam_set', 'scenario', '-v7.3');
fprintf('\nsaved %s\n', fullfile(outdir, ['sim02_' scenario '.mat']));
