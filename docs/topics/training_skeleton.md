# Deep Learning Training Skeleton
## Data Pre-processing
### Normalizing Input Data
- When inputting data to a deep learning model, it is standard practice to normalize the data to zero mean and unit variance. 
- For example, the input data consists of several features x1, x2,…xn. Each feature might have a different range of values. For instance, values for feature x1 might range from 1 through 5, while values for feature x2 might range from 1000 to 99999.
- Hence, for each feature column separately, we take the values of all samples in the dataset and compute the mean and the variance. And then normalize the values using the formula below.

$$ X_i = (X_i - Mean_i) \over (StdDev_i) $$

<p align="center"><img width="250" src="https://user-images.githubusercontent.com/64508435/226076715-f5974ac9-a1dd-4736-b234-c44bdcc4be3a.png"></p>

#### Why needs Input Data Normalization
- To understand what happens without normalization, let’s look at an example with just two features that are on drastically different scales. 
- Since the network output is a linear combination of each feature vector, this means that the network learns weights for each feature that are also on different scales.
- Therefore, during gradient descent, in order to “move the needle” for the Loss, the network would have to make a large update to one weight compared to the other weight. This can cause the gradient descent trajectory to oscillate back and forth along one dimension, thus taking more steps to reach the minimum.
- Instead, if the features are on the same scale, the loss landscape is more uniform like a bowl. Gradient descent can then proceed smoothly down to the minimum.

