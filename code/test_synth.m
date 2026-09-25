% Check that oof_synth_signal + oof_welch recover the target PSD, and that
% the generalized model reduces to the additive / multiplicative forms.
addpath(fullfile(fileparts(mfilename('fullpath')),'lib'));
rng(1);
srate = 500; dur = 120; nsamp = srate*dur;
fmod = 0.5:0.25:80;
ap = struct('offset', 1.0, 'exponent', 1.4, 'knee', 0);
pk = struct('cf', 10, 'bw', 1.5, 'amp', 0.30);

for lambda = [0 1]
    S = oof_model_psd(fmod, ap, pk, lambda);
    nrep = 20; Pacc = 0;
    for r = 1:nrep
        x = oof_synth_signal(S, fmod, srate, nsamp);
        [P, f] = oof_welch(x, srate, 4, 0.5, [1 80]);
        Pacc = Pacc + P;
    end
    Pacc = Pacc/nrep;
    Starg = interp1(log10(fmod), log10(S), log10(f), 'linear', 'extrap');
    err = mean(abs(log10(Pacc) - Starg));
    fprintf('lambda=%g : mean |log10 PSD error| over 1-80 Hz = %.4f dex\n', lambda, err);
end

% closed-form checks
S0 = oof_model_psd(fmod, ap, pk, 0);
S1 = oof_model_psd(fmod, ap, pk, 1);
L  = oof_aperiodic(fmod, ap.offset, ap.exponent, 0);
G  = oof_peak(fmod, pk.cf, pk.bw);
fprintf('additive form max abs diff       = %.3e\n', max(abs(S0 - (L + pk.amp*G))));
fprintf('multiplicative form max abs diff = %.3e\n', max(abs(S1 - L.*(1 + pk.amp*G))));
