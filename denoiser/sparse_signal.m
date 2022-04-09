function x = sparse_signal(N, rho, sigma_x)
% x = sparse_signal(L, rho, sigma_x)
% Generates a sparse signal following a Gaussian sparse prior
%
% INPUT
%    L: signal length
%    rho: sparsity level
%    sigma_x: Gaussian standard deviation
%
% OUTPUT
%    x: sparse signal

nonz = ceil(rho * N);
x = zeros(N, 1);
k = randperm(N);
%x(k(1:nonz)) = sigma_x * randn(nonz, 1);
x(k(1:nonz)) = 20 * (rand(nonz, 1)-0.5);