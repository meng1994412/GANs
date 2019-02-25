# GANs
## Objectives
Implemented DCGANs and train them on MNIST dataset to create synthetic but identical
and indistinguishable to any hand drawn digit in MNIST dataset.
* Built a generator to generate fake images and constructed a discriminator to spot the
synthetic images from the authentic ones.
* Trained the DCGANs to create synthetic images that are identical to MNIST dataset.

## Packages Used
* Python 3.6
* [OpenCV](https://docs.opencv.org/3.4.4/) 4.0.0
* [keras](https://keras.io/) 2.1.0
* [Tensorflow](https://www.tensorflow.org/install/) 1.13.0
* [cuda toolkit](https://developer.nvidia.com/cuda-toolkit) 10.0
* [cuDNN](https://developer.nvidia.com/cudnn) 7.4.2
* [scikit-learn](https://scikit-learn.org/stable/) 0.20.1
* [Imutils](https://github.com/jrosebr1/imutils)
* [NumPy](http://www.numpy.org/)

## Approaches
### Generator and discriminator in DCGANs
The `dcgan.py` defines both a generator, which accepts an input vector of randomly generated noise and produce synthetic image, and a discriminator, which attempts to determine if a given image is authentic or synthetic.

According to *Ganerative Adversarial Networks* ([reference](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)), Radford et al. provides some recommendations for building more stable GANs, which are used in the project. We can replace any pooling layers with strided convolutions (fully convolutional). We can use batch normalization in both generator and discriminator. We can use `ReLU` in generator except for the final layer which will utilize `tanh`. We can use `LeakyReLU` in discriminator.

Table 1 shows the model summary of generator.

Table 1: Model summary of generator.

| Layer (Type)                               | Output Size        | Param # |
| ------------------------------------------ |:------------------:|:-------:|
| dense_1 (Dense)                            | (None, 512)        | 51712   |
| activation_1 (Activation)                  | (None, 512)        | 0       |
| batch_normalization_1 (BatchNormalization) | (None, 512)        | 2048    |
| dense_2 (Dense)                            | (None, 3136)       | 1608768 |
| activation_2 (Activation)                  | (None, 3136)       | 0       |
| batch_normalization_2 (BatchNormalization) | (None, 3136)       | 12544   |
| reshape_1 (Reshape)                        | (None, 7, 7, 64)   | 0       |
| conv2d_transpose_1 (Conv2DTranspose)       | (None, 14, 14, 32) | 51232   |
| activation_3 (Activation)                  | (None, 14, 14, 32) | 0       |
| batch_normalization_3 (BatchNormalization) | (None, 14, 14, 32) | 128     |
| conv2d_transpose_2 (Conv2DTranspose)       | (None, 28, 28, 1)  | 801     |
| activation_4 (Activation)                  | (None, 28, 28, 1)  | 0       |

### Training process of DCGANs
At each iteration of training process, we generate random images and then train the discriminator to correctly distinguish between the authentic images and synthetic images. We then generate additional synthetic images, but this time purposely trying to fool the discriminator. And finally we update the weights of the generator based on the feedback of the discriminator, thereby allowing us to generate more authentic images.

## Results
Figure 1 and 2 show the early synthetic images in the early epochs (epoch 1 to 6). Figure 3 shows the later synthetic images (epoch 20, 35, 50).

<img src="https://github.com/meng1994412/GANs/blob/master/output/epoch_0001_output.png" height="250"> <img src="https://github.com/meng1994412/GANs/blob/master/output/epoch_0002_output.png" height="250"> <img src="https://github.com/meng1994412/GANs/blob/master/output/epoch_0003_output.png" height="250">

Figure 1: Synthetic images for epoch 1 (left), epoch 2 (middle), and epoch 3 (right).

<img src="https://github.com/meng1994412/GANs/blob/master/output/epoch_0004_output.png" height="250"> <img src="https://github.com/meng1994412/GANs/blob/master/output/epoch_0005_output.png" height="250"> <img src="https://github.com/meng1994412/GANs/blob/master/output/epoch_0006_output.png" height="250">

Figure 2: Synthetic images for epoch 4 (left), epoch 5 (middle), and epoch 6 (right).

<img src="https://github.com/meng1994412/GANs/blob/master/output/epoch_0020_output.png" height="250"> <img src="https://github.com/meng1994412/GANs/blob/master/output/epoch_0035_output.png" height="250"> <img src="https://github.com/meng1994412/GANs/blob/master/output/epoch_0050_output.png" height="250">

Figure 2: Synthetic images for epoch 20 (left), epoch 35 (middle), and epoch 50 (right).
