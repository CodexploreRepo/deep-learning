# Time Series

## 1. Datasets
### 1.1. Dataset types
- **Time Series** or **Sequence** data — Rank-3 tensors of shape `(samples, timesteps, features)`
  - The time axis is always the second axis (axis of index 1) by convention. 
  - where each sample is a sequence (of length timesteps) of feature vectors
<p align="center"><img height="120" src="https://user-images.githubusercontent.com/64508435/222749955-b55851ab-326c-465c-bce3-b515f365e879.png"></p>

- Example 1 - dataset of stock prices:
  - Every minute, we store 3 information: the current price of the stock, the highest price in the past minute, and the lowest price in the past minute.    
  - Entire day of trading has 390 minutes of trading, which is encoded as a matrix of shape `(390, 3)`
  - Entire year has 250 days’ worth of data can be stored in a rank-3 tensor of shape `(250, 390, 3)`
- Example 2 - dataset of tweets
  -  Each tweet as a sequence of 280 characters out of an alphabet of 128 unique characters. 
  -  In this setting, each character can be encoded as a binary vector of size 128 (an all-zeros vector except for a 1 entry at the index corresponding to the character). 
  -  Then each tweet can be encoded as a rank-2 tensor of shape `(280, 128)`
  -  A dataset of 1 million tweets can be stored in a tensor of shape `(1000000, 280, 128)`. 

# Resources
- [Time series prediction with LSTM in Tensorflow](https://towardsdatascience.com/time-series-prediction-with-lstm-in-tensorflow-42104db39340)
