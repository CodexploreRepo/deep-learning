# Deep Learning Training Skeleton
## Data Pre-processing
### Normalizing Input Data
- When inputting data to a deep learning model, it is standard practice to normalize the data to zero mean and unit variance. 
- For example, the input data consists of several features x1, x2,…xn. Each feature might have a different range of values. For instance, values for feature x1 might range from 1 through 5, while values for feature x2 might range from 1000 to 99999.
- Hence, for each feature column separately, we take the values of all samples in the dataset and compute the mean and the variance. And then normalize the values using the formula below.

$$ Xi = (X_i - Mean_i) \over (StdDev_i) $$
