function x = oof_synth_signal(S, f, srate, nsamp)
% Realize a stationary Gaussian process whose expected one-sided PSD is S.
%
% Frequency-domain synthesis: draw circularly-symmetric complex Gaussian
% Fourier coefficients with variance set so that the (rectangular-window)
% periodogram of the output is an unbiased estimate of S, impose Hermitian
% symmetry, and invert.
%
% S      : 1 x nf target one-sided PSD in uV^2/Hz
% f      : 1 x nf frequency grid in Hz, strictly increasing, f(1) > 0
% srate  : sampling rate in Hz
% nsamp  : output length in samples
% x      : nsamp x 1 time series in uV

nf_half = floor(nsamp/2);
fgrid   = (0:nf_half).' * (srate/nsamp);

% log-log interpolation of the target, held constant outside [f(1) f(end)]
lS = interp1(log10(f(:)), log10(S(:)), log10(max(fgrid, eps)), 'linear', NaN);
lS(fgrid <  f(1))   = log10(S(1));
lS(fgrid >  f(end)) = log10(S(end));
Starget = 10.^lS;
Starget(fgrid == 0) = 0;                 % no DC

% E|X_k|^2 = S(f_k) * srate * nsamp / 2  gives an unbiased one-sided
% periodogram  P_k = 2|X_k|^2 / (srate*nsamp).
sigma = sqrt(Starget * srate * nsamp / 2);

X = zeros(nsamp, 1);
kk = 2:nf_half+1;                        % 1-based indices of f > 0
X(kk) = sigma(kk) .* (randn(numel(kk),1) + 1i*randn(numel(kk),1)) / sqrt(2);
if mod(nsamp,2) == 0
    % Nyquist bin must be real; variance is preserved with a real draw
    X(nf_half+1) = sigma(nf_half+1) * randn;
    X(nf_half+2:end) = conj(X(nf_half:-1:2));
else
    X(nf_half+2:end) = conj(X(nf_half+1:-1:2));
end
x = real(ifft(X));
x = x(:);
end
