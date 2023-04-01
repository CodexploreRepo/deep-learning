# Computer Vision

## 1. Datasets

### 1.1. Dataset types

#### Images

- Rank-4 tensors of shape `(samples, height, width, channels)`
  - where each sample is a 2D grid of pixels, and each pixel is represented by a vector of values (“channels”)
- Color:
  - Grey Scale: 1-channel (0 to 255)
  - RGB Color: 3-channel
    - Example: A batch of 128 grayscale images of size 256 × 256 could thus be stored in a tensor of shape `(128, 256, 256, 1)`, and a batch of 128 color images could be stored in a tensor of shape `(128, 256, 256, 3)`
- There are two conventions for shapes of image tensors: `(128, 256, 256, 1)` and `(128, 256, 256, 3)`
  - **Channels-last** convention (which is standard in TensorFlow):
  - **Channels-first** convention (which is increasingly falling out of favor): `(128, 1, 256, 256)` and `(128, 3, 256, 256)`

<p align="center"><img src="https://user-images.githubusercontent.com/64508435/222752853-7b4f9bce-174c-4f54-8c65-4016062f46b0.png"><br>A rank-4 image data tensor</p>

#### Video

- A video can be understood as a sequence of frames, each frame being a color image (a rank-3 tensor `(height, width, color_ depth)`)
- Rank-5 tensors of shape `(samples, frames, height, width, channels)`
  - where each sample is a sequence (of length frames) of images
  - Example: a 60-second, 144 × 256 YouTube video clip sampled at 4 frames per second would have 240 frames.
    - A batch of four such video clips would be stored in a tensor of shape `(4, 240, 144, 256, 3)`.
    - That’s a total of 106,168,320 values! If the dtype of the tensor was float32, each value would be stored in 32 bits, so the tensor would represent 405 MB.
    - In real life, Videos you encounter are much lighter, because they aren’t stored in float32, and they’re typically compressed by a large factor (such as in the MPEG format).

### 1.2. List of Datasets

| Name | Size | Description | Contains |
| ---- | ---- | ----------- | -------- |
| **[CIFAR-10](https://www.kaggle.com/c/cifar-10)** a subset of CIFAR-100             | a subset of the 80 million tiny images dataset and consists of 60,000 32x32 color images containing one of 10 object classes, with 6000 images per class | CIFAR-10 is an established computer-vision dataset used for object recognition. It is a subset of the 80 million tiny images dataset and consists of 60,000 32x32 color images containing one of 10 object classes, with 6000 images per class. It was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. | airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and truck |
| **[IAM](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)**            |                                                                                                                                                          | contains forms of handwritten English text which can be used to train and test handwritten text recognizers and to perform writer identification and verification experiments.                                                                                                                                        |
| **[ImageNet](https://paperswithcode.com/dataset/imagenet)**            |                                                                                                                                                          | The ImageNet project is a large visual database designed for use in visual object recognition software research. More than 14 million images have been hand-annotated by the project | For object detection and image classification at large scale |                                                                                   
| **[MNIST](https://keras.io/examples/vision/mnist_convnet/)**                        | 60,000 training images, plus 10,000 test images                                                                                                          | Assembled by the National Institute of Standards and Technology (the NIST in MNIST) in the 1980s                                                                                                                                                                                                                      | grayscale images of handwritten digits (28 × 28 pixels) into their 10 categories (0 through 9) |
| **[EMNIST (Extened MNIST)](https://keras.io/examples/vision/mnist_convnet/)**       |                                                                                                                                                          | [EMNIST (an extension of MNIST to handwritten letters)](https://arxiv.org/abs/1702.05373v1) **handwritten character digits** (letters and digits) and that shares the same image structure as `MNIST`, derived from the NIST Special Database 19 and converted to a 28x28 pixel image format and dataset structure.  | `emnist/letters`, `emnist/digits`                                                              |
| **[K-MNIST (Kuzushiji-MNIST)](https://www.tensorflow.org/datasets/catalog/kmnist)** |                                                                                                                                                          | Grayscale images of Japanese handwritten. Since MNIST restricts us to 10 classes, we chose one character to represent each of the 10 rows of Hiragana when creating Kuzushiji-MNIST. digits                                                                                                                           |                                                                                                |     |
| **[Fashion-MNIST](https://www.tensorflow.org/datasets/catalog/fashion_mnist)**      |                                                                                                                                                          | a dataset of Zalando's article images                                                                                                                                                                                                                                                                                 | 28x28 grayscale image, associated with a label from 10 classes.                                |

# 2. Convolution Neural Network
## 2.1. Layer Pattern
The most common form of a ConvNet architecture
- Stacks a few CONV-RELU layers, follows them with POOL layers,
- And repeats this pattern until the image has been merged spatially to a small size.
- At some point, it is common to transition to fully-connected layers
```
INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> Final FC
– * indicates repetition
– POOL? indicates an optional pooling layer
– N >= 0 and usually N <= 3
– M >= 0,
– K >= 0 and usually K < 3

Example: INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]*3 -> [FC -> RELU]*2 -> FC
```
## 2.2. The Evolution of Architecture
Keyword: batch normalization, 1x1 convolution, average pooling
### 2.2.1. LeNet
- Yann Lecun's LeNet-5 (“Gradient-based learning applied to document recognition”) model was developed in 1998 to identify handwritten digits for zip code recognition in the postal service.
- This pioneering model largely introduced the convolutional neural network as we know it today
- The subsampling layers use a form of `average pooling`.
<p align="center"><img width="850" alt="Screenshot 2023-04-01 at 17 14 46" src="https://user-images.githubusercontent.com/64508435/229277321-23bf5c05-17ab-4e96-a16c-00521e2cd058.png"></p>

### 2.2.2. AlexNet
- The first work that popularized CNN in Computer Vision was the AlexNet, developed by Alex Krizhevsky, Ilya Sutskever and Geoffrey Hinton in 2012
- The Network had a very similar architecture to LeNet, but was deeper, bigger
- Featured Convolutional Layers stacked on top of each other (previously it was **common to only have a single CONV layer always immediately followed by a POOL layer**)
<p align="center">
<img width="800" alt="Screenshot 2023-04-01 at 17 21 18" src="https://user-images.githubusercontent.com/64508435/229277635-9577359f-f1e1-45de-91c5-05e995dd9ae4.png"></p>

### 2.2.3. VGGNet
- Main contribution was in showing that the depth of the network is a critical component for good performance, and is developed in 2014 by Karen Simonyan and Andrew Zisserman
- At the time of its introduction, this model was considered to be very deep
<p align="center"><img width="600" alt="Screenshot 2023-04-01 at 17 23 00" src="https://user-images.githubusercontent.com/64508435/229277725-181d89d0-b154-419a-9bbe-df7f89351cca.png"></p>

### 2.2.4. GoogleNet
- Introduced in 2014 by Google, the main contribution was the development of an **Inception Module** that dramatically reduced the number of parameters in the network (4M, compared to AlexNet with 60M)
  - A set of 1x1, 3x3, and 5x5 filters which can learn to extract features at different scales from the input
  - *1x1 convolutions are used to reduce the input channel depth*
- Additionally, this paper uses Average Pooling instead of Fully Connected layers at the top of the ConvNet, eliminating a large amount of parameters that do not seem to matter much
<p align="center">
<img width="900" alt="Screenshot 2023-04-01 at 17 24 53" src="https://user-images.githubusercontent.com/64508435/229277822-fd8872ba-2558-420f-adce-7f56d3f1585a.png"><br>Inception Module in GoogleNet<br>
<img width="550" alt="Screenshot 2023-04-01 at 17 26 59" src="https://user-images.githubusercontent.com/64508435/229277924-bdcc88a0-08b7-436e-9477-295ab9323121.png"><br>1x1 convolutions are used to reduce the input channel depth<br>
<img width="750" alt="Screenshot 2023-04-01 at 17 31 01" src="https://user-images.githubusercontent.com/64508435/229278105-cda9483d-7542-4dab-8a23-466ed94be877.png"><br>Number of Parameters
</p>

### 2.2.5. MobileNet
- The MobileNet model is based on depth-wise separable convolutions which is a form of factorized convolutions which factorize a standard convolution into
  - a depthwise convolution and
  - a 1x1 convolution called a **pointwise convolution**.
<p align="center"><img width="500" alt="Screenshot 2023-04-01 at 17 41 02" src="https://user-images.githubusercontent.com/64508435/229278539-89d1bdb8-0e1d-4995-bb91-c4fcec5c0f42.png"></p>

### 2.2.6. EfficientNet
- Empirical study shows that it is critical to balance all dimensions of network width/depth/resolution, and surprisingly such balance can be achieved by simply scaling each of them with constant ratio.
<p align="center"><img width="1064" alt="Screenshot 2023-04-01 at 17 41 54" src="https://user-images.githubusercontent.com/64508435/229278581-0ddab32b-b98d-4d52-997d-39119342086a.png"></p>

### 2.2.7. Resnet
- Residual Network developed by Kaiming He et al. was the winner of ILSVRC 2015. It features special **skip connections** and a heavy use of **batch normalization**.
- Intermediate layers of a block learn a residual function with reference to the block input.
- ResNets are currently by far state of the art Convolutional Neural Network models and are the default choice for using ConvNets in practice.
<p align="center">
<img width="850" alt="Screenshot 2023-04-01 at 17 35 34" src="https://user-images.githubusercontent.com/64508435/229278292-2d0b9399-d6de-4ded-849a-59b77ef0e417.png"></p>

### 2.2.8. Vision Transformer


