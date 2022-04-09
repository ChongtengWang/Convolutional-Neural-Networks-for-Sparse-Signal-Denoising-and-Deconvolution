# Deconvolution CNN

## Demo for deconvolution CNN

In this [demo](/deconvolver/demo_deconvolver.m), a deconvolution CNN is loaded from .mat files. The deconvolution CNN is applied to the input signal, which can be a pure sparse signal after convolution, a pure Gaussian noise or a sparse signal after convolution and corrupted by Gaussian noise. You can decide the type of the input signal as well as the sparsity of the signal and the noise level of the Gaussian noise by setting different parameters in the demo program. The convolution filter should be fixed, because the deconvolution CNN was trained based on the known convolution filter.

The MSE between the output signal and the groundtruth as well as the SNR of the output signal can be calculated. Also, the groundtruth, groundtruth after convolution, input signal and output signal will be displayed in different subplots to visually show the deconvolution results.

In addition, you can choose to show the input and output signals of any layer, which helps visualize the deconvolution process of the deconvolution CNN more clearly. Finally, the filters in the deconvolution CNN will be plotted.

The core command of applying the deconvolution CNN to the input signal is output = CNN(input, deconvolver); Therefore, you can create your own test signal and test the deconvolution CNN on the test signal to evaluate the performance.

## Performance metric for the deconvolution CNN

In this [program](/deconvolver/figures.m), the deconvolution CNN is evaluated through MSE and SNR metrics. 

First, the structure and filters of the deconvolution CNN are shown. Then the deconvolution CNN is applied to different types of input signals, and the output signals are displayed compared to the input signals and the grountruth. The deconvolution CNN is also compared to the inverse filter with a denoising CNN under different types of input signals. What's more, the MSE and SNR of the deconvolution CNN are calculated and plotted with different sparsities and noise levels to show how the sparsity and noise level influence the performance of the CNN. Finally, The MSE and SNR performance of the deconvolution CNN are compared with other CNNs and the inverse filter under multiple sparsities and noise levels to show that the proposed deconvolution CNN can achieve the best performance under most circumstances.

You can change the parameters in the program to change the test data to evaluate the deconvolution CNN under different conditions. You can also add other deconvolution methods and compare the deconvolution CNN with those methods and evaluate their SNR and MSE. Because the test data is generated randomly, there may be slight differences between the outputs of the program and the results in the thesis.
