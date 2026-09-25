% First identifiability check for the coupling exponent.
%
% Simulate spectra at a known lambda with the intrinsic oscillatory strength
% drawn INDEPENDENTLY of the aperiodic background, then try to recover
% lambda two ways:
%   (1) spectral-shape estimator  : Whittle profile likelihood (oof_profile_lambda)
%   (2) population estimator      : log-log slope of periodic on aperiodic
%                                   band power across spectra (oof_lambda_pop)
%
% Run: matlab -batch "run('code/test_lambda_identify.m')"

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'lib'));
rng(7);

srate = 500; dur = 60; nsamp = srate*dur;
fmod  = 0.25:0.25:120;
frng  = [2 45];
band  = [8 12];
N     = 40;
lam_true_set = [0 0.5 1];

fprintf('\n%-6s | %-28s | %-34s\n', 'true', 'Whittle profile', 'population log-log slope');
fprintf('%s\n', repmat('-', 1, 78));

for lt = lam_true_set
    % ---- ground truth, background and oscillator independent -----------
    offset = 0.6 + 1.2*rand(N,1);          % 1.2 decades of background range
    expo   = 1.2 + 0.35*randn(N,1);
    cint   = 10.^(-0.7 + 0.25*randn(N,1)); % intrinsic strength, independent of offset
    cf     = 10 + 0.8*randn(N,1);
    bw     = 1.4 + 0.2*rand(N,1);

    P = zeros(N, 0); fpsd = [];
    Ph = cell(N,1);
    for i = 1:N
        ap = struct('offset', offset(i), 'exponent', expo(i), 'knee', 0);
        % amplitude is defined so that the ABSOLUTE peak power at cf equals
        % cint(i) * L(cf)^lt  -> raw coefficient is cint(i)
        pk = struct('cf', cf(i), 'bw', bw(i), 'amp', cint(i));
        S  = oof_model_psd(fmod, ap, pk, lt);
        x  = oof_synth_signal(S, fmod, srate, nsamp);
        [p, fpsd] = oof_welch(x, srate, 4, 0.5, frng);
        Ph{i} = p;
    end
    P = cell2mat(Ph);

    % ---- initialisation from the specparam port -----------------------
    sp = oof_est_specparam(P, fpsd, struct('max_n_peaks', 3, 'peak_width_limits', [1 8]));
    init = repmat(struct('offset',0,'exponent',1,'knee',0,'peaks',zeros(1,3)), 1, N);
    for i = 1:N
        pkabs = sp.peak_abs{i};
        if isempty(pkabs)
            pkabs = [10 0.1*10^sp.offset(i)/10^sp.exponent(i) 1.5];
        else
            [~, ia] = min(abs(pkabs(:,1) - 10));   % keep the alpha-range peak
            pkabs = pkabs(ia,:);
        end
        init(i) = struct('offset', sp.offset(i), 'exponent', sp.exponent(i), ...
                         'knee', 0, 'peaks', pkabs);
    end

    % ---- (1) Whittle profile over lambda ------------------------------
    pr = oof_profile_lambda(P, fpsd, init, 0:0.125:1.25, struct('alpha',0.05));

    % ---- (2) population estimator, from a lambda = 0 fit --------------
    f0 = oof_fit_coupling(P, fpsd, 0, init);
    [apw, perw] = oof_bandpower(f0, fpsd, band);

    % independent split-half for the instrument
    Podd = zeros(N, numel(fpsd)); Peve = zeros(N, numel(fpsd));
    for i = 1:N
        % re-derive two independent halves by resynthesising is not valid;
        % instead split the Welch segments of the same signal
        Podd(i,:) = P(i,:); Peve(i,:) = P(i,:);
    end
    pop = oof_lambda_pop(perw, apw, [], [], 500);

    fprintf('%-6.2f | %5.3f  [%5.3f %5.3f]  p0=%.3g p1=%.3g | OLS %5.3f [%5.3f %5.3f]  TS %5.3f\n', ...
        lt, pr.lambda, pr.ci(1), pr.ci(2), pr.p_additive, pr.p_multiplicative, ...
        pop.lambda_ols, pop.ci_ols(1), pop.ci_ols(2), pop.lambda_robust);
end
