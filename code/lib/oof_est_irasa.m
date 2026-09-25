function out = oof_est_irasa(x, srate, frange, hset, winsec)
% IRASA on time-domain signals (Wen & Liu, 2016) -- the real algorithm, not
% a frequency-axis approximation: resample by h and by 1/h, take the
% geometric mean of the two PSDs, then the median over h. Oscillatory peaks
% are displaced in opposite directions by the two resamplings and cancel;
% the self-similar (fractal) part does not.
%
% NOTE ON THE MODEL. IRASA recovers the oscillatory PSD by LINEAR
% SUBTRACTION, Sosc = Smixed - Sfractal, i.e. it assumes purely ADDITIVE
% coupling (lambda = 0 in oof_model_psd) -- the opposite structural
% assumption to specparam. Peak amplitudes are absolute power.
%
% x       : nsamp x ntrials time series
% srate   : sampling rate (Hz)
% frange  : [flo fhi] range of interest; the returned grid is restricted to
%           [flo*hmax, fhi/hmax] to avoid resampling edge effects
% hset    : resampling factors (default 1.1:0.05:1.9)
% winsec  : Welch window length in seconds (default 4)

if nargin < 4 || isempty(hset),   hset   = 1.1:0.05:1.9; end
if nargin < 5 || isempty(winsec), winsec = 4;            end
hmax = max(hset);
fev  = [frange(1)*hmax, frange(2)/hmax];

[Pmix, f] = oof_welch(x, srate, winsec, 0.5, fev);   % ntrials x nf
nt = size(Pmix,1); nf = numel(f);
Pg = zeros(numel(hset), nt, nf);

for ih = 1:numel(hset)
    h = hset(ih);
    [nu, de] = rat(h, 1e-6);
    xu = resample(x, nu, de);
    xd = resample(x, de, nu);
    Pu = interp_psd(xu, srate, winsec, f);
    Pd = interp_psd(xd, srate, winsec, f);
    Pg(ih,:,:) = sqrt(max(Pu, eps) .* max(Pd, eps));
end
Pfrac = reshape(median(Pg, 1), nt, nf);

lf = log10(f(:)); X = [ones(nf,1) lf];
B  = X \ log10(Pfrac).';
out = struct('offset', B(1,:).', 'exponent', -B(2,:).', 'knee', zeros(nt,1), ...
             'f', f, 'Pmixed', Pmix, 'Pfractal', Pfrac, 'Posc', Pmix - Pfrac);
end

function Pi = interp_psd(y, srate, winsec, f)
% PSD of the resampled series, expressed back on the original frequency grid
[Py, fy] = oof_welch(y, srate, winsec, 0.5, []);
Pi = zeros(size(Py,1), numel(f));
for i = 1:size(Py,1)
    Pi(i,:) = interp1(fy, Py(i,:), f, 'linear', 'extrap');
end
end
