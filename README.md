# Convolutional Neural Networks for Sparse Signal Denoising and Deconvolution -- Master Thesis

This is the repository for Master thesis in 2022 advised by [Professor Ivan Selesnick](https://eeweb.engineering.nyu.edu/iselesni/) at NYU Tandon School of Engineering.

This repository intends to save the programs to reproduce the results in the thesis.
There are two jupyter programs used to train the CNNs with and without constraints, respectively.
The trained CNNs can be saved as a .mat file which can be evaluated by the following Matlab programs.
There are four main Matlab program files and several Matlab function files as well as .mat files which are necessary to run the main programs.
Two demos for the denoising and deconvolution tasks are included in the program files.
Other two programs are to metric and show the performance of the denoising and deconvolution CNNs.

## Contents
1. [Programs for training the CNN](/training/README.md)
2. [Demo for denoising task](/denoiser/README.md)
3. [Demo for deconvolution task](/deconvolver/README.md)
4. [Performance metric for the denoising CNN](/denoiser/README.md)
5. [Performance metric for the deconvolution CNN](/deconvolver/README.md)

## Experimental settings

The training process can be implemented using PyTorch 1.4.0 and Python 3.7. I ran the program on [Google Colab](https://colab.research.google.com/).

All the software needed to run the MATLAB demo and performance programs is [MATLAB](https://www.mathworks.com/products/get-matlab.html?s_tid=gn_getml). 

In the thesis, all the experiments were implemented using MATLAB (version 9.9) on a MacBook Pro (Apple M1 Pro).

## List of Matlab program files available in this repository

Following is the list of demos available in this repository. When you run the main programs, you need to download the necessary function files as well as the .mat files into the same directory where the main programs are. The figures which show performance of the CNNs will be saved as eps file in the 'figures' directory.

| Directory Name |              File Name                                          |                                               Description                                               |
|:--------------:|:---------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------:|
|    denoiser    |           [demo_denoiser.m](/denoiser/demo_denoiser.m)          |                                    Demonstration of the denoising CNN                                   |
|    denoiser    |              [figures.m](/denoiser/figures.m)                   |   Metric and show the performance of the denoising CNN. Reproduce the figures and tables in the thesis   |
|    denoiser    |              [figures.mlx](/denoiser/figures.mlx)               |   Matlab real-time script including both codes and outputs          |
|    denoiser    |              [figures.pdf](/denoiser/figures.pdf)               |   PDF file including both codes and outputs                         |
|    denoiser    |                [CNN.m](/denoiser/CNN.m)                         |                             Function file: apply the CNN to the input signal                            |
|    denoiser    |               [ReLU.m](/denoiser/ReLU.m)                        |                                Function file: implement the ReLU function                               |
|    denoiser    |              [conv1d.m](/denoiser/conv1d.m)                     |                       Function file: implement the convolution function in a layer                      |
|    denoiser    |         [create_denoiser.m](/denoiser/create_denoiser.m)        |                            Function file: create the simplified denoising CNN                           |
|    denoiser    |               [layer.m](/denoiser/layer.m)                      |                         Function file: combine conv1d and ReLU for a whole layer                        |
|    denoiser    |         [set_plot_defaults.m](/denoiser/set_plot_defaults.m)    |                       Function file: set figure properties when saving the figures                      |
|    denoiser    |        [sparse_denoise_MAD.m](/denoiser/sparse_denoise_MAD.m)   |                         Function files: Implement the MAD-soft denoising method                         |
|    denoiser    | [sparse_denoise_MAD_calculate_SNR.m](/denoiser/sparse_denoise_MAD_calculate_SNR.m)  |                         Function files: Implement the MAD-soft denoising method                         |
|    denoiser    |           [sparse_signal.m](/denoiser/sparse_signal.m)          |                         Function files: Implement the MAD-soft denoising method                         |
|    denoiser    |                [SNR.m](/denoiser/SNR.m)                         |                         Function files: Implement the MAD-soft denoising method                         |
|   deconvolver  |         [demo_deconvolver.m](/deconvolver/demo_deconvolver.m)   |                                  Demonstration of the deconvolution CNN                                 |
|   deconvolver  |              [figures.m](/deconvolver/figures.m)                | Metric and show the performance of the deconvolution CNN. Reproduce the figures and tables in the thesis |
|   deconvolver  |              [figures.mlx](/deconvolver/figures.mlx)                | Matlab real-time script including both codes and outputs     |
|   deconvolver  |              [figures.pdf](/deconvolver/figures.pdf)                | PDF file including both codes and outputs                    |
|   deconvolver  |          [conv1d_withplot.m](/deconvolver/conv1d_withplot.m)    |  Function file: implement the convolution function in a layer while plotting the output of each filter  |
|   deconvolver  |                [CNN.m](/deconvolver/CNN.m)                      |                             The same function as that in denoiser directory                             |
|   deconvolver  |               [ReLU.m](/deconvolver/ReLU.m)                     |                             The same function as that in denoiser directory                             |
|   deconvolver  |              [conv1d.m](/deconvolver/conv1d.m)                  |                             The same function as that in denoiser directory                             |
|   deconvolver  |         [create_denoiser.m](/deconvolver/create_denoiser.m)     |                             The same function as that in denoiser directory                             |
|   deconvolver  |               [layer.m](/deconvolver/layer.m)                   |                             The same function as that in denoiser directory                             |
|   deconvolver  |         [set_plot_defaults.m](/deconvolver/set_plot_defaults.m) |                             The same function as that in denoiser directory                             |
