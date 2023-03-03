# Computer Vision
## 1. Datasets
### 1.1. Dataset types
- **Images** - Rank-4 tensors of shape `(samples, height, width, channels)`
  - where each sample is a 2D grid of pixels, and each pixel is represented by a vector of values (“channels”)
- **Video** - Rank-5 tensors of shape `(samples, frames, height, width, channels)`
  - where each sample is a sequence (of length frames) of images
#### 1.1.1. Images
- Grey Scale: 1-channel (0 to 255)
- RGB: 3-channel
### 1.2. List of Datasets
- **[MNIST](https://keras.io/examples/vision/mnist_convnet/)**: assembled by the National Institute of Standards and Technology (the NIST in MNIST) in the 1980s
  - *Contains*: grayscale images of handwritten digits (28 × 28 pixels) into their 10 categories (0 through 9)
  - A set of 60,000 training images, plus 10,000 test images

