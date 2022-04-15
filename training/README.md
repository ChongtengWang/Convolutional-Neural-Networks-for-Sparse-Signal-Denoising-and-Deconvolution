# Train the CNN for denoising or deconvolution in PyTorch

## Regular training

In this [training program](\training\regular_training.ipynb), a 5-layer CNN is trained. 

First, a dataset is loaded from Google Drive. 
For the denoising task, the dataset can be sparse signals (with random sparsities) corrupted by Gaussian noise (with random standard deviations).
For the deconovlution task, the dataset can be sparse signals (with random sparsities) convolved with a known convolution filter and corrupted by Gaussian noise (with random standard deviations).

Second, a CNN is defined. Here there are 5 convolutional layers which are 1x2, 2x2, 2x2, 2x2 and 2x1, respectively.
Each layer except the last layer is followed by a ReLU activation.
The same structure can be used for both denoising task and deconvolution task.
You can also try different structures by adding or removing the convolutional layers, or changing the number of input or output channels.

Third, the designed CNN is initialized. A pre-set CNN is loaded from Google Drive as the initialization for the CNN.
There is no constraint for the CNN except that the first, fourth and fifth layers are fixed (non-trainable).
The trainable layers (second and third) are initialized to random values.

Finally, the CNN is trained and the training and testing loss are calculated. 
Some training examples are also displayed. The trained CNN is saved as a .mat file that can be evaluated in Matlab.

## Constrained training

In this [training program](\training\constrained_training.ipynb), a 5-layer CNN is trained. 

First, a dataset is loaded from Google Drive. 
For the denoising task, the dataset can be sparse signals (with random sparsities) corrupted by Gaussian noise (with random standard deviations).
For the deconovlution task, the dataset can be sparse signals (with random sparsities) convolved with a known convolution filter and corrupted by Gaussian noise (with random standard deviations).

Second, a CNN is defined. Here there are 5 convolutional layers which are 1x2, 2x2, 2x2, 2x2 and 2x1, respectively.
Each layer except the last layer is followed by a ReLU activation.
The same structure can be used for both denoising task and deconvolution task.
You can also try different structures by adding or removing the convolutional layers, or changing the number of input or output channels.
Compared with the regular training program, the difference is that there some constraints on the trainable layers.

For example, in the second layer, the filter (1,1) has to be the same as (2,2), and the filter (1,2) has to be the same as (2,1).
So in the training process, these filters are constrained to be the same.
One way to do this is shown in the codes: 
use a 1x2 trainable layer to process the input signal, and then use the filpped weights (non-trainable) of the layer to process the input signal.
This can make sure that the filter in (1,1) is exactly the same as that in (2,2), and the filters in (1,2) is exactly the same as that in (2,1).

Third, the designed CNN is initialized. A pre-set CNN is loaded from Google Drive as the initialization for the CNN.
The first, fourth and fifth layers are fixed (non-trainable).

Finally, the CNN is trained and the training and testing loss are calculated. 
Some training examples are also displayed. The trained CNN is saved as a .mat file that can be evaluated in Matlab.
