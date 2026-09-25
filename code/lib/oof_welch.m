function [P, f] = oof_welch(x, srate, winsec, overlap, frange)
% Welch PSD with a Hann window, one-sided, uV^2/Hz. Column-wise on x.
if nargin < 3 || isempty(winsec),  winsec  = 2;    end
if nargin < 4 || isempty(overlap), overlap = 0.5;  end
if nargin < 5, frange = []; end
nwin = round(winsec*srate);
[P, f] = pwelch(x, hann(nwin), round(overlap*nwin), nwin, srate);
if ~isempty(frange)
    keep = f >= frange(1) & f <= frange(2);
    P = P(keep,:); f = f(keep);
end
P = P.'; f = f(:).';      % rows = signals, cols = frequencies
end
