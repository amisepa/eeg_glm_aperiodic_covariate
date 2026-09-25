function G = oof_peak(f, cf, bw)
% Unit-height Gaussian peak kernel on the frequency axis.
%   bw is the Gaussian standard deviation in Hz.
f = f(:).';
G = exp(-0.5*((f-cf)./bw).^2);
end
