% Diagnostic: (a) is lambda identifiable from the SHAPE of one spectrum?
%             (b) does the population log-log estimator work with an oracle
%                 aperiodic fit, and then with estimated aperiodic fits?
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'code', 'lib'));
rng(11);

fmod = 0.25:0.25:120; frng = [2 45]; band = [8 12];
ap = struct('offset', 1.0, 'exponent', 1.4, 'knee', 0);

%% (a) shape degeneracy: best Gaussian (lambda=0) approximation to a
%     lambda=1 periodic component
f = frng(1):0.25:frng(2);
L = oof_aperiodic(f, ap.offset, ap.exponent, 0);
cf = 10; bw = 1.5; a = 0.2;
target = a * oof_peak(f, cf, bw) .* L;              % lambda = 1 periodic part
obj = @(p) sum((p(1)*oof_peak(f, p(2), p(3)) - target).^2);
p = fminsearch(obj, [max(target) cf bw], optimset('Display','off','TolX',1e-10,'TolFun',1e-14));
resid = p(1)*oof_peak(f, p(2), p(3)) - target;
fprintf('(a) lambda=1 periodic part approximated by a plain Gaussian:\n');
fprintf('    best-fit  amp %.4g  cf %.3f Hz  bw %.3f Hz\n', p(1), p(2), p(3));
fprintf('    residual RMS / peak height = %.4f%%\n', 100*sqrt(mean(resid.^2))/max(target));
fprintf('    peak-relative max |residual| = %.4f%%\n\n', 100*max(abs(resid))/max(target));

%% (b) population estimator
srate = 500; dur = 120; nsamp = srate*dur; N = 120;
for lt = [0 0.5 1]
    offset = 0.4 + 1.4*rand(N,1);
    expo   = 1.2 + 0.30*randn(N,1);
    cint   = 10.^(-0.7 + 0.25*randn(N,1));
    P  = zeros(N, 0); P1 = []; P2 = [];
    for i = 1:N
        api = struct('offset', offset(i), 'exponent', expo(i), 'knee', 0);
        pk  = struct('cf', 10, 'bw', 1.5, 'amp', cint(i));
        S   = oof_model_psd(fmod, api, pk, lt);
        x   = oof_synth_signal(S, fmod, srate, nsamp);
        [pa, fp] = oof_welch(x(1:nsamp/2),     srate, 4, 0.5, frng);
        [pb, ~ ] = oof_welch(x(nsamp/2+1:end), srate, 4, 0.5, frng);
        [pf, ~ ] = oof_welch(x,                srate, 4, 0.5, frng);
        if isempty(P), P = zeros(N, numel(fp)); P1 = P; P2 = P; end
        P(i,:) = pf; P1(i,:) = pa; P2(i,:) = pb;
    end
    m = fp >= band(1) & fp <= band(2);

    % --- oracle aperiodic -------------------------------------------------
    bo = zeros(N,1); ao = zeros(N,1);
    for i = 1:N
        Li = oof_aperiodic(fp, offset(i), expo(i), 0);
        bo(i) = mean(Li(m)); ao(i) = mean(P(i,m)) - bo(i);
    end
    po = oof_lambda_pop(ao, bo, [], [], 400);

    % --- estimated aperiodic (censored regression), band power by residual -
    est  = oof_est_censored(P,  fp);
    est1 = oof_est_censored(P1, fp);
    est2 = oof_est_censored(P2, fp);
    bp = @(e, PP) deal_band(e, PP, fp, m);
    [be, ae] = bp(est,  P);
    [b1, a1] = bp(est1, P1);
    [b2, a2] = bp(est2, P2);
    pe = oof_lambda_pop(ae, be, [], [], 400);
    pv = oof_lambda_pop(a1, b1, a2, b2, 400);

    fprintf('(b) true lambda = %.2f\n', lt);
    fprintf('    oracle aperiodic      : OLS %6.3f [%6.3f %6.3f]   TS %6.3f\n', ...
        po.lambda_ols, po.ci_ols(1), po.ci_ols(2), po.lambda_robust);
    fprintf('    censored fit, naive   : OLS %6.3f [%6.3f %6.3f]   TS %6.3f\n', ...
        pe.lambda_ols, pe.ci_ols(1), pe.ci_ols(2), pe.lambda_robust);
    fprintf('    censored fit, split IV: IV  %6.3f [%6.3f %6.3f]   rel %.3f\n\n', ...
        pv.lambda_iv, pv.ci_iv(1), pv.ci_iv(2), pv.reliability);
end

function [b, a] = deal_band(e, PP, fp, m)
N = size(PP,1); b = zeros(N,1); a = zeros(N,1);
for i = 1:N
    Li = oof_aperiodic(fp, e.offset(i), e.exponent(i), 0);
    b(i) = mean(Li(m)); a(i) = mean(PP(i,m)) - b(i);
end
end
