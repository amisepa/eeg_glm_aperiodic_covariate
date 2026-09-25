function L = oof_aperiodic(f, offset, expo, knee)
% Aperiodic PSD, Lorentzian form (Donoghue et al. 2020).
%   L(f) = 10^offset / (knee + f^expo)
% knee = 0 gives the pure power law 10^offset * f^(-expo).
% offset is in log10 power units (as specparam reports it).
if nargin < 4 || isempty(knee), knee = 0; end
f = f(:).';
L = (10.^offset) ./ (knee + f.^expo);
end
