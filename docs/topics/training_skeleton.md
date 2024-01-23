# Deep Learning Training Skeleton

## Data Pre-processing

### Input Data Type Conversion

- Transform input (x, y) data type to a `float32`
- For image: convert pixel from range `[0, 255]` to `[0,1]`
  - Tensorflow: divide the tensor / 255
  - Pytorch: `ToTensor()` function will help convert to

### Input Data Normalization

- When inputting data to a deep learning model, it is standard practice to normalize the data to zero mean and unit variance.
- For example, the input data consists of several features x1, x2,…xn. Each feature might have a different range of values. For instance, values for feature x1 might range from 1 through 5, while values for feature x2 might range from 1000 to 99999.
- Hence, for each feature column separately, we take the values of all samples in the dataset and compute the mean and the variance. And then normalize the values using the formula below.

$$ X_i = (X_i - Mean_i) \over (StdDev_i) $$

<p align="center"><img width="250" src="https://user-images.githubusercontent.com/64508435/226076715-f5974ac9-a1dd-4736-b234-c44bdcc4be3a.png"></p>

- In the picture below, The original values (in blue) are now centered around zero (in red). This ensures that all the feature values are now on the same scale.
<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/64508435/226076957-cce2cb01-a566-4c30-baf2-193b411873c9.png">
  <br>The effect of normalizing data
</p>

#### Why needs Input Data Normalization

- To understand what happens without normalization, let’s look at an example with just two features that are on drastically different scales.
- Since the network output is a linear combination of each feature vector, this means that the network learns weights for each feature that are also on different scales.
- Therefore, during gradient descent, in order to “move the needle” for the Loss, the network would have to make a large update to one weight compared to the other weight. This can cause the gradient descent trajectory to oscillate back and forth along one dimension, thus taking more steps to reach the minimum.
- Instead, if the features are on the same scale, the loss landscape is more uniform like a bowl. Gradient descent can then proceed smoothly down to the minimum.

#### Compute Mean & Std for Image

- Step 1:
  - **sum**: used to compute means
  - **squared sum of pixel values**: needed for standard deviation calculations.
  - The first two steps are done in the snippet below. Note that we set axis = `[0, 2, 3]` to compute mean values with respect to axis 1 `(Batch_size, C, H, W)`. The dimensions of inputs is `[batch_size x 3 x image_size x image_size]`, so we need to make sure we aggregate values per each RGB channel separately
- Step 2: Loop through the batches and add up channel-specific sum and squared sum values.
- Step 3: Perform final calculations to obtain data-level mean and standard deviation.
  - **Mean**: simply divide the sum of pixel values by the total count - number of pixels in the dataset computed as `len(df) * image_size * image_size`
  - **Standard deviation**: use the following equation: `total_std = sqrt(psum_sq / count - total_mean ** 2)` - Note: when mean is not available, we can use [Sum of Squares Formula Shortcut](https://www.thoughtco.com/sum-of-squares-formula-shortcut-3126266) to compute the variance
    $$Variance = \sum_{i=1}^n \left( x_i - \bar{x} \right)^2 = \left[ \sum_{i=1}^n x_i^2 \right] - \frac{1}{n} \left[ \sum_{k=1}^n x_i \right]^2$$

```Python
###### COMPUTE MEAN / STD

# placeholders
psum    = torch.tensor([0.0, 0.0, 0.0])
psum_sq = torch.tensor([0.0, 0.0, 0.0])

# loop through images
for inputs, labels in tqdm(train_dataloader):
   psum    += inputs.sum(axis        = [0, 2, 3])
   psum_sq += (inputs ** 2).sum(axis = [0, 2, 3])

image_size = 256

####### FINAL CALCULATIONS
# pixel count
count = len(train_df) * image_size * image_size

# mean and std
total_mean = psum / count
total_var  = (psum_sq / count) - (total_mean ** 2)
total_std  = torch.sqrt(total_var)
```
