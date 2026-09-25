% SIM 1 -- Aperiodic estimator benchmark with ground truth.
%
% Kalamala, Clements, Gyurkovics et al. (2026) compared aperiodic estimators
% on two real data sets using odd-even reliability, outlier frequency and
% effect size, and recommended censored regression (a fixed 6-16 Hz window
% excluded from every spectrum). Having no ground truth, they could not
% measure bias. This simulation adds that, keeps their metrics, and adds the
% downstream quantity that matters for separation: the coupling estimate.
%
% Reported per estimator:
%   exp bias / RMSE   error in the aperiodic exponent
%   off bias          error in the offset (log10 power at 1 Hz)
%   log b bias        error in aperiodic power inside the alpha band, in dex
%   reliability       correlation of estimates between two independent halves
%                     of the same recording (their odd-even metric)
%   pos slope %       their outlier metric: fraction of positive slopes
%   lambda_hat        population coupling estimate; should equal lambda_true
%
% Writes results/sim01.mat

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'lib'));
outdir = fullfile(fileparts(here), 'results');
if ~exist(outdir, 'dir'), mkdir(outdir); end
rng(2026);

srate = 500; dur = 120; nsamp = srate*dur;
fmod  = 0.25:0.25:120;
frng  = [2 45];
band  = [8 12];
N     = 120;
lam_set  = [0 0.5 1];
rel_peak = 0.8;

names = { ...
  'Full regression', ...
  'Censored 6-16 Hz', ...
  'OLS excl 8-12 Hz', ...
  'Robust (bisquare)', ...
  'Theil-Sen', ...
  'specparam 1 peak', ...
  'specparam 3 peaks', ...
  'specparam 6 peaks', ...
  'IRASA', ...
  'Joint Whittle', ...
  'Self-censoring (iter.)'};
nE = numel(names);

R = struct();
for li = 1:numel(lam_set)
    lt = lam_set(li);
    fprintf('\n================ true lambda = %.2f ================\n', lt);

    offset = 0.4 + 1.4*rand(N,1);
    expo   = 1.2 + 0.30*randn(N,1);
    cf_i   = 10 + 0.6*randn(N,1);
    bw_i   = 1.5 + 0.15*randn(N,1);
    Lref   = 10^median(offset) / 10^median(expo);      % L at 10 Hz, median params
    cint   = rel_peak * Lref^(1-lt) * 10.^(0.25*randn(N,1));

    X = zeros(nsamp, N);
    for i = 1:N
        api = struct('offset', offset(i), 'exponent', expo(i), 'knee', 0);
        pk  = struct('cf', cf_i(i), 'bw', bw_i(i), 'amp', cint(i));
        X(:,i) = oof_synth_signal(oof_model_psd(fmod, api, pk, lt), fmod, srate, nsamp);
    end
    h = nsamp/2;
    [P,  fp] = oof_welch(X,          srate, 4, 0.5, frng);
    [Pa, ~ ] = oof_welch(X(1:h,:),   srate, 4, 0.5, frng);
    [Pb, ~ ] = oof_welch(X(h+1:end,:), srate, 4, 0.5, frng);
    m = fp >= band(1) & fp <= band(2);

    % ---- ground truth band quantities -------------------------------
    b_true = zeros(N,1); a_true = zeros(N,1); relh = zeros(N,1);
    for i = 1:N
        api = struct('offset', offset(i), 'exponent', expo(i), 'knee', 0);
        pk  = struct('cf', cf_i(i), 'bw', bw_i(i), 'amp', cint(i));
        [~, parts] = oof_model_psd(fp, api, pk, lt);
        b_true(i) = mean(parts.aperiodic(m));
        a_true(i) = mean(parts.periodic(m));
        Lcf = oof_aperiodic(cf_i(i), offset(i), expo(i), 0);
        relh(i) = cint(i)*Lcf^lt / Lcf;
    end
    fprintf('relative peak height: median %.2f, range %.2f-%.2f\n', ...
        median(relh), min(relh), max(relh));
    fprintf('ground-truth log-log slope of periodic on aperiodic = %.3f\n', ...
        slope_(log(b_true), log(a_true)));

    % ---- run every estimator on the full and the two half recordings --
    E = cell(nE,3);
    for s = 1:3
        switch s, case 1, PP = P; case 2, PP = Pa; case 3, PP = Pb; end
        E{1,s}  = oof_est_full(PP, fp);
        E{2,s}  = oof_est_censored(PP, fp, [6 16]);
        E{3,s}  = oof_est_ols(PP, fp, fp>=8 & fp<=12);
        E{4,s}  = oof_est_robust(PP, fp);
        E{5,s}  = oof_est_theilsen(PP, fp);
        E{6,s}  = oof_est_specparam(PP, fp, struct('max_n_peaks',1,'peak_width_limits',[2 12]));
        E{7,s}  = oof_est_specparam(PP, fp, struct('max_n_peaks',3,'peak_width_limits',[2 12]));
        E{8,s}  = oof_est_specparam(PP, fp, struct('max_n_peaks',6,'peak_width_limits',[2 12]));
        E{11,s} = oof_est_selfcensor(PP, fp);
    end
    % IRASA needs the time series
    ir  = oof_est_irasa(X,            srate, frng, 1.1:0.05:1.9, 4);
    ira = oof_est_irasa(X(1:h,:),     srate, frng, 1.1:0.05:1.9, 4);
    irb = oof_est_irasa(X(h+1:end,:), srate, frng, 1.1:0.05:1.9, 4);
    E{9,1} = ir; E{9,2} = ira; E{9,3} = irb;
    % joint Whittle, initialised from specparam(3)
    for s = 1:3
        switch s, case 1, PP = P; case 2, PP = Pa; case 3, PP = Pb; end
        sp = E{7,s};
        init = repmat(struct('offset',0,'exponent',1,'knee',0,'peaks',zeros(1,3)), 1, N);
        for i = 1:N
            pa = sp.peak_abs{i};
            if isempty(pa)
                pa = [10, 0.3*10^sp.offset(i)*10^(-sp.exponent(i)), 1.5];
            else
                [~, ia] = min(abs(pa(:,1) - 10)); pa = pa(ia,:);
                pa(2) = max(pa(2), 1e-9);
            end
            init(i) = struct('offset', sp.offset(i), 'exponent', sp.exponent(i), ...
                             'knee', 0, 'peaks', pa);
        end
        E{10,s} = oof_fit_coupling(PP, fp, 0, init);
    end

    % ---- score ------------------------------------------------------
    res = struct('name', names);
    for e = 1:nE
        ee = E{e,1};
        if e == 9
            fpe = ir.f; me = fpe >= band(1) & fpe <= band(2);
            bhat = mean(ir.Pfractal(:,me), 2);
            ahat = mean(ir.Pmixed(:,me), 2) - bhat;
        elseif e == 10
            [bhat, ahat] = oof_bandpower(ee, fp, band);
        else
            bhat = bandap_(ee, fp, m);
            ahat = mean(P(:,m), 2) - bhat;
        end
        res(e).exp_bias = mean(ee.exponent - expo);
        res(e).exp_rmse = sqrt(mean((ee.exponent - expo).^2));
        res(e).off_bias = mean(ee.offset - offset);
        res(e).b_logbias = mean(log10(bhat) - log10(b_true));
        res(e).pos_slope = 100*mean(ee.exponent <= 0);
        res(e).rel_exp = corr(E{e,2}.exponent, E{e,3}.exponent);
        res(e).rel_off = corr(E{e,2}.offset,   E{e,3}.offset);
        ok = ahat > 0;
        g  = oof_glm_coupling(ahat(ok), bhat(ok));
        res(e).lambda = g.lambda; res(e).lambda_se = g.lambda_se;
        res(e).lambda_err = g.lambda - lt;
        res(e).n_valid = sum(ok);
    end

    fprintf('\n%-23s %8s %8s %8s %9s %7s %7s %8s %16s\n', 'estimator', ...
        'expBias','expRMSE','offBias','log b dex','rel_ex','rel_of','pos%','lambda_hat (SE)');
    for e = 1:nE
        fprintf('%-23s %8.3f %8.3f %8.3f %9.3f %7.3f %7.3f %7.1f  %6.3f (%.3f)\n', ...
            res(e).name, res(e).exp_bias, res(e).exp_rmse, res(e).off_bias, ...
            res(e).b_logbias, res(e).rel_exp, res(e).rel_off, res(e).pos_slope, ...
            res(e).lambda, res(e).lambda_se);
    end

    R(li).lambda_true = lt; R(li).res = res; R(li).f = fp; R(li).band = band;
    R(li).truth = struct('offset',offset,'exponent',expo,'cf',cf_i,'bw',bw_i, ...
                         'cint',cint,'b_true',b_true,'a_true',a_true,'relh',relh);
end
save(fullfile(outdir,'sim01.mat'), 'R', 'names', 'lam_set', '-v7.3');
fprintf('\nsaved %s\n', fullfile(outdir,'sim01.mat'));

function b = bandap_(ee, fp, m)
N = numel(ee.offset); b = zeros(N,1);
for i = 1:N
    kn = 0; if isfield(ee,'knee') && ~isempty(ee.knee), kn = ee.knee(i); end
    L = oof_aperiodic(fp, ee.offset(i), ee.exponent(i), kn);
    b(i) = mean(L(m));
end
end

function s = slope_(x, y)
s = sum((x-mean(x)).*(y-mean(y))) / sum((x-mean(x)).^2);
end
