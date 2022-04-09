clear
clc

%% Create the CNN by training or using the trained parameters.

threshold1 = 2.4;
threshold2 = 4.8;
rho = 1;                % rho is the ratio between output and input signal.
l = 37;            % l is the length of the filters in the second layer.
training_sigma = 2;     % The standard deviation of the Gaussian noise in the training data is between 0 and training_sigma.
training_num = 60000;   % training_num is the number of the training signals.
training_type = 1;      % 1 means Uniform and 2 means Gaussian.
istrain_flag = false;   % istrain_flag can determine if training a new CNN or directly using the trained parameters.
h1 = create_denoiser(l,rho,threshold1,threshold2,training_type,istrain_flag);
% h1 = create_denoiser(l,rho,threshold1,threshold2,training_type,istrain_flag,training_num,training_sigma);


%% Create input signal (noisy signal) and ground truth (pure signal).

% N is the total length of the pure sparse signal.
% K is the number of non-zeros in the pure sparse signal.
% As a result, 1-K/N determines the sparsity of the pure signal.

K = 25;
N = 500;


%% Choose the type of the input signal

signal_type = 'noisy';
% signal_type = 'pure_signal';
% signal_type = 'pure_impulse';
% signal_type = 'pure_noise';

sigma = 0.5;

switch signal_type

    case 'noisy'

        groundtruth = zeros(1, N);

        index_random = randperm(N);
        index = index_random(1:K);

        groundtruth(index)  = 10*2*(rand(1,K) - 0.5);
%         groundtruth(index)  = 10*randn(1,K);

        input = groundtruth + sigma*randn(1,N);
        
    case 'pure_signal'

        groundtruth = zeros(1, N);

        index_random = randperm(N);
        index = index_random(1:K);

        groundtruth(index)  = 10*2*(rand(1,K) - 0.5);
        input = groundtruth;

    case 'pure_impulse'

        groundtruth = zeros(1, N);
        groundtruth(N/2) = 10;
        input = groundtruth + randn(1,N);

    case 'pure_noise'

        groundtruth = zeros(1, N);
        input = groundtruth + randn(1,N);
        
end


%% Apply the CNN and compute the loss (mean square error).

output = CNN(input, h1);
loss = mean((output - groundtruth).^2)
SNR = 10*log10(mean(groundtruth.^2)/loss)

%% Display groundtruth, input signal and output signal.

figure(1)
clf 
subplot(3,1,1)
plot(groundtruth)
title('pure signal (S[n])');
ylim([-10,10])
subplot(3,1,2)
plot(input)
title('noisy signal (x[n])');
ylim([-10,10])
subplot(3,1,3)
plot(output)
title('output signal (y[n])');
ylim([-10,10])

%% Plot the input signal and output signal in the same figure.

n = 1:N;
figure(2)
clf
stem(n, input)
hold on
plot(n, output, 'ro')
legend('Input (x[n])', 'Output (y[n])')
ylim([-10,10])

%% Plot the groundtruth and output signal in the same figure.

figure(3)
clf
stem(n, groundtruth)
hold on 
stem(n, output)

legend('True', 'Output')

%% Plot the output signal v.s. input signal and display the thresholds.

figure(4)
plot(input, output, 'o')
hold on;
line([threshold1 threshold1], [-10 10],'Color','magenta','LineStyle','--')
line([threshold2 threshold2], [-10 10],'Color','red','LineStyle','--')
line([-threshold1 -threshold1], [-10 10],'Color','magenta','LineStyle','--')
line([-threshold2 -threshold2], [-10 10],'Color','red','LineStyle','--')
xlabel('Input')
ylabel('Output')
grid
axis equal square
line([-10 10], [-10 10])

%% Display the layers of the CNN

figure(5)
[r,c,~] = size(h1{1});
for i=1:1:r
    for j=1:1:c
    subplot(r,c,c*i-(c-j))
    stem(flip(squeeze(h1{1}(i,j,:))))
    end
end

figure(6)
[r,c,~] = size(h1{2});
for i=1:1:r
    for j=1:1:c
    subplot(r,c,c*i-(c-j))
    stem(flip(squeeze(h1{2}(i,j,:))))
    hold on
    plot(flip(squeeze(h1{2}(i,j,:))))
    hold off
    end
end
figure(7)
[r,c,~] = size(h1{3});
for i=1:1:r
    for j=1:1:c
    subplot(r,c,c*i-(c-j))
    stem(flip(squeeze(h1{3}(i,j,:))))
    end
end