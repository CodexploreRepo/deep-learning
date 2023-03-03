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
    - 
### 1.2. List of Datasets
- **[MNIST](https://keras.io/examples/vision/mnist_convnet/)**: assembled by the National Institute of Standards and Technology (the NIST in MNIST) in the 1980s
  - *Contains*: grayscale images of handwritten digits (28 × 28 pixels) into their 10 categories (0 through 9)
  - A set of 60,000 training images, plus 10,000 test images

