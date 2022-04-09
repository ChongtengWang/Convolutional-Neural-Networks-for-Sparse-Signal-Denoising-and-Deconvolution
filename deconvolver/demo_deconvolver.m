clc;
clear;
close all;
%% Create the convolution filter and load the proposed CNN.

r = 0.9;                             % Define filter
om = 0.95;
a = [1 -2*r*cos(om) r^2];
b = [1 r*cos(om)];
h = filter(b, a, [zeros(1,38) 1 zeros(1,40)]);

load('sin23_2.mat');

deconvolver{1} = double(conv1);
deconvolver{2} = double(conv2);
deconvolver{3} = double(conv3);
deconvolver{4} = double(conv4);
deconvolver{5} = double(conv5);

%% Choose the type of the input signal.

signal_type = 'noisy';
% signal_type = 'pure_impulse';
% signal_type = 'pure_noise';
% signal_type = 'test_signal';


K = 25;
N = 500;
sigma = 1;


switch signal_type

    case 'noisy'

        groundtruth = zeros(1, N);
        index_random = randperm(N);
        index = index_random(1:K);
        groundtruth(index) = 10*2*(rand(1,K) - 0.5);
        after_conv = conv(groundtruth,h,'same');
        input = after_conv + sigma*randn(1,N);



    case 'pure_impulse'

        groundtruth = zeros(1,500);
        groundtruth(300) = 10;
        groundtruth(200) = -10;
        after_conv = conv(groundtruth,h,'same');
        input = after_conv+ sigma*randn(1,N);

    case 'pure_noise'

        groundtruth = zeros(1, N);
        after_conv = conv(groundtruth,h,'same');
        input = after_conv + sigma*randn(1,N);
        
    case 'test_signal'
        
        groundtruth = zeros(1, N);
        groundtruth(40:15:500) = 25;
        groundtruth(48:15:500) = -25;

        after_conv = conv(groundtruth,h,'same');
        input = after_conv + sigma*randn(1,N);
%         input = after_conv;

        
end

%% Choose the output of which layer is displayed.

% operation = 'first';
% operation = 'second';
% operation = 'third';
% operation = 'forth';
operation = 'total';

switch operation
    case 'first'
        l1 = layer(input,deconvolver{1});
        figure()
        subplot(2,1,1)
        plot(1:N, l1(1,:))
        title('channel 1')
        subplot(2,1,2)
        plot(1:N, l1(2,:))
        title('channel 2')
        figure()
        plot(groundtruth)
        figure()
        stem(input)   
        
    case 'second'
        l1 = layer(input,deconvolver{1});
        l2 = conv1d_withplot(l1,deconvolver{2});     
        
    case 'third'
        l1 = layer(input,deconvolver{1});
        l2 = layer(l1,deconvolver{2});
        l3 = conv1d_withplot(l2,deconvolver{3});
        
    case 'forth'
        l1 = layer(input,deconvolver{1});
        l2 = layer(l1,deconvolver{2});
        l3 = layer(l2,deconvolver{3});
        l4 = conv1d_withplot(l3,deconvolver{4});
        
    case 'total'
        
        output = CNN(input,deconvolver);
        MSE = mean((output - groundtruth).^2)
        SNR = 10*log10(mean(groundtruth.^2)/loss)
        %Plot the signals
        figure()
        subplot(4,1,1)
        plot(groundtruth);
        title('pure signal')
        subplot(4,1,2)
        plot(after_conv);
        title('signal after convolution')
        %ylim([-30,30])
        subplot(4,1,3)
        plot(input)
        title('input signal')
        subplot(4,1,4)
        plot(output)
        %ylim([-30,30])
        title('output signal')
               
end

%% Plot the groundtruth and output signal in the same figure.

n = 1:500;
figure()
clf
stem(n, groundtruth)
hold on 
stem(n, output)

legend('True', 'Output')

%% Plot the output signal v.s. input signal.

figure()
plot(groundtruth, after_conv, 'ro')
hold on;
plot(input, output, 'bo')
hold off;
xlabel('Input')
ylabel('Output')
legend('after_conv','deconv')
grid
axis equal square
% line([-10 10], [-10 10])

%% Display the layers of the CNN.

figure()
[r,c,~] = size(deconvolver{1});
for i=1:1:r
    for j=1:1:c
    subplot(r,c,c*i-(c-j))
    stem(flip(squeeze(deconvolver{1}(i,j,:))))
    hold on
    plot(flip(squeeze(deconvolver{1}(i,j,:))))
    hold off
    %xlim([0,18])
    %xlim([0,10])
    end
end
figure()
[r,c,~] = size(deconvolver{2});
for i=1:1:r
    for j=1:1:c
    subplot(r,c,c*i-(c-j))
    stem(flip(squeeze(deconvolver{2}(i,j,:))))
    hold on
    plot(flip(squeeze(deconvolver{2}(i,j,:))))
    hold off
    %xlim([0,18])
    %xlim([0,10])
    end
end
figure()
[r,c,~] = size(deconvolver{3});
for i=1:1:r
    for j=1:1:c
    subplot(r,c,c*i-(c-j))
    stem(flip(squeeze(deconvolver{3}(i,j,:))))
    hold on
    plot(flip(squeeze(deconvolver{3}(i,j,:))))
    hold off
    %xlim([0,18])
    %xlim([0,10])
    end
end
figure()
[r,c,~] = size(deconvolver{4});
for i=1:1:r
    for j=1:1:c
    subplot(r,c,c*i-(c-j))
    stem(flip(squeeze(deconvolver{4}(i,j,:))))
    hold on
    plot(flip(squeeze(deconvolver{4}(i,j,:))))
    hold off
    %xlim([0,18])
    %xlim([0,10])
    end
end

figure()
[r,c,~] = size(deconvolver{5});
for i=1:1:r
    for j=1:1:c
    subplot(r,c,c*i-(c-j))
    stem(flip(squeeze(deconvolver{5}(i,j,:))))
    hold on
    plot(flip(squeeze(deconvolver{5}(i,j,:))))
    hold off
    %xlim([0,18])
    %xlim([0,10])
    end
end

 