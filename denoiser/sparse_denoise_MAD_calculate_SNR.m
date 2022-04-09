function [sigma_w, SNR_x_hat, SNR_y, MSE] = sparse_denoise_MAD_calculate_SNR(N, rho, sigma_x)
% [sigma_w, SNR_x_hat, SNR_y] = SNsparse_denoise_MAD_calculate_SNRR_MAD(N, rho, sigma_x)
%
% INPUT
%   N : length of signal
%   rho : sparsity level ( 0 < rho < 1 )
%   sigma_x : signal standard deviation (scalar)
%
% OUTPUT
%   sigma_w : noise standard deviations
%   SNR_x_hat : SNR of estimated signal
%   SNR_y : SNR of noisy signal
%
% Notes
%  rho = 0.1 means that 10% of the signal is non-zero

%     rng(1)

%% Set parameters

Nr = 1000;     % Number of realizations

min_sigma_w = 0.1;  % min noise std range
max_sigma_w = 2.0;  % max noise std range
num_sigmas_w = 20;  % number of noise std points in the range

%% Initialization

sigma_w = logspace(log10(min_sigma_w), log10(max_sigma_w), num_sigmas_w);

SNR_y_vals = nan(num_sigmas_w, Nr);
SNR_x_hat_vals = nan(num_sigmas_w, Nr);
MSE_vals = nan(num_sigmas_w, Nr);
% [noise sigmas, realizations]

%% Computation

for i = 1:num_sigmas_w
    % loop over noise levels
    fprintf('progress = %f\n', i/num_sigmas_w);

    for j = 1:Nr
        % loop over realizations

        x = sparse_signal(N, rho, sigma_x);
        y = x + sigma_w(i) * randn(N, 1);

        x_hat = sparse_denoise_MAD(y);  % signal estimation

        SNR_x_hat_vals(i, j) = SNR(x_hat, x);
        SNR_y_vals(i, j) = SNR(y, x);
        MSE_vals(i, j) = mean((x_hat-x).^2);
    end
end


%% Average across realizations

SNR_x_hat = mean(SNR_x_hat_vals, [2]);
SNR_y = mean(SNR_y_vals, [2]);
MSE = mean(MSE_vals, [2]);

