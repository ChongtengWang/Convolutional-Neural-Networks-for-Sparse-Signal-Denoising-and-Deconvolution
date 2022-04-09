# Denoiser CNN

## Demo for denoiser CNN

In this [demo](/denoiser/demo_denoiser.m), a denoiser CNN is created by the current trained parameters. The denoiser CNN is applied to the input signal, which can be a pure sparse signal, a pure Gaussian noise or a noisy signal. You can decide the type of the input signal as well as the sparsity of the signal and the noise level of the Gaussian noise by setting different parameters in the demo program.

The MSE between the output signal and the groundtruth as well as the SNR of the output signal can be calculated. Also, the groundtruth, input signal and output signal will be displayed in different subplots to visually show the denoising results.

In addition, the output signal v.s. groundtruth and the output signal v.s. the input signal will be displayed to visually show the firm thresholding. Finally, the filters in the denoiser CNN will be plotted.

The core command of applying the denoiser CNN to the input signal is output = CNN(input, h1); Therefore, you can create your own test signal and test the denoiser CNN on the test signal to evaluate the performance. You can also change the training parameters in the demo as the comment describes to create a new denoiser CNN based on different training data.

## 
