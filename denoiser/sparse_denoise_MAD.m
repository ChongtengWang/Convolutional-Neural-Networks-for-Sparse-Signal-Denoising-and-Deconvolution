function x_hat = sparse_denoise_MAD(y)
% x_hat = sparse_denoise_MAD(x)
%
% INPUT
%   y : input signal 
%
% OUTPUT
%   x_hat : output
%
% Median absolute deviation (MAD)
% See 
% https://en.wikipedia.org/wiki/Median_absolute_deviation
%
% MAD is a robust estimate of deviation of a data set

mad_y = median(abs(y));

sigma_w = 1.4826 * mad_y;

soft = @(x, T) max(x - T, 0) + min(x + T, 0);

T = 1.2 * sigma_w;

x_hat = soft(y, T);



