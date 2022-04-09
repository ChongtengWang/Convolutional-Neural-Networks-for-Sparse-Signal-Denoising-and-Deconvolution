# Denoising CNN

## Demo for denoising CNN

In this [demo](/denoiser/demo_denoiser.m), a denoising CNN is created by the current trained parameters. The denoising CNN is applied to the input signal, which can be a pure sparse signal, a pure Gaussian noise or a noisy signal. You can decide the type of the input signal as well as the sparsity of the signal and the noise level of the Gaussian noise by setting different parameters in the demo program.

The MSE between the output signal and the groundtruth as well as the SNR of the output signal can be calculated. Also, the groundtruth, input signal and output signal will be displayed in different subplots to visually show the denoising results.

In addition, the output signal v.s. groundtruth and the output signal v.s. the input signal will be displayed to visually show the firm thresholding. Finally, the filters in the denoising CNN will be plotted.

The core command of applying the denoising CNN to the input signal is output = CNN(input, h1); Therefore, you can create your own test signal and test the denoising CNN on the test signal to evaluate the performance. You can also change the training parameters in the demo as the comment describes to create a new denoising CNN based on different training data.

## Performance metric for the denoising CNN

In this [program](/denoiser/figures.m), the denoising CNN is evaluated through MSE and SNR metrics. 

First, the structure and filters of the denoising CNN are shown. Then the denoising CNN is applied to different types of input signals, and the output signals are displayed compared to the input signals and the grountruth. What's more, the MSE and SNR of the denoising CNN are calculated and plotted with different sparsities and noise levels to show how the sparsity and noise level influence the performance of the CNN. Finally, The MSE and SNR performance of the denoising CNN are compared with other CNNs and MAD-soft denoising method under multiple sparsities and noise levels to show that the proposed denoising CNN can achieve the best performance under most circumstances.

You can change the parameters in the program to change the test data to evaluate the denoising CNN under different conditions. You can also add other denoising methods and compare the denoising CNN with those methods and evaluate their SNR and MSE. Because the test data is generated randomly, there may be slight differences between the outputs of the program and the results in the thesis.
