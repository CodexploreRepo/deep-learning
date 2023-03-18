# Batch Normalization (Batch Norm)
- Batch Norm ([Paper](https://arxiv.org/pdf/1502.03167.pdf)) is a neural network layer that is now commonly used in many architectures. 
- It often gets added as part of a Linear or Convolutional block and helps to stabilize the network during training.
- Batch Normalization was recognized as being transformational in creating deeper neural networks that could be trained faster.

## The need for Batch Norm
- The activations from the previous layer are simply the inputs to this layer. 
  - For instance, from the perspective of Layer 2 in the picture below, if we “blank out” all the previous layers, the activations coming from Layer 1 are no different from the original inputs, which also must be normalized

<p align="center"><img src="https://user-images.githubusercontent.com/64508435/226077388-dda61fb2-30c9-4c3b-ad9c-ef5128b0b985.png"/><br>The inputs of each hidden layer are the activations from the previous layer, and must also be normalized</p>

## How does Batch Norm work ?
- Batch Norm is just another network layer that gets inserted between a hidden layer and the next hidden layer. Its job is to take the outputs from the first hidden layer and normalize them before passing them on as the input of the next hidden layer.
- Batch Norm layer also has parameters of its own:
  - **Two learnable parameters**: `beta` and `gamma`.
  - **Two non-learnable parameters**: *Mean Moving Average* and *Variance Moving Average* are saved as part of the ‘state’ of the Batch Norm layer.

<p align="center"><img width=500 src="https://user-images.githubusercontent.com/64508435/226077539-e33446e3-1546-4420-9fea-6dc16bedf13c.png"/></p>

- So if we have, say, three hidden layers and three Batch Norm layers in the network, we would have three learnable beta and gamma parameters for the three layers. Similarly for the Moving Average parameters.
![image](https://user-images.githubusercontent.com/64508435/226077638-5c20103d-46ca-47a1-86c0-d05fd5659588.png)

### During Training
- During training, we feed the network one mini-batch of data at a time. During the forward pass, each layer of the network processes that mini-batch of data. The Batch Norm layer processes its data as follows:
<p align="center"><img width=600 src="https://user-images.githubusercontent.com/64508435/226078414-48b3d9cf-22e0-43fe-994d-313cd42f8877.png"/></p>

Step 1: Activations
- The activations from the previous layer are passed as input to the Batch Norm. There is one activation vector for each feature in the data.

Step 2: Calculate Mean and Variance
- For each activation vector separately, calculate the mean and variance of all the values in the mini-batch.

Step 3: Normalize
- Calculate the normalized values for each activation feature vector using the corresponding mean and variance. 
- These normalized values now have zero mean and unit variance.

Step 4: Scale and Shift
- This step is the huge innovation introduced by Batch Norm that gives it its power. 
- Unlike the input layer, which requires all normalized values to have zero mean and unit variance, Batch Norm allows its values to be shifted (to a different mean) and scaled (to a different variance). 
- It does this by multiplying the normalized values by a factor, `gamma`, and adding to it a factor, `beta`. 
  - Note that this is an element-wise multiply, not a matrix multiply.
- In other words, each Batch Norm layer is able to optimally find the best factors for itself, and can thus shift and scale the normalized values to get the best predictions.

Step 5: Moving Average
- In addition, Batch Norm also keeps a running count of the Exponential Moving Average (EMA) of the `mean` and `variance`. 
- During training, it simply calculates this EMA but does not do anything with it. 
- At the end of training, it simply saves this value as part of the layer’s state, for use during the Inference phase.


## Reference:
- [Batch Norm Explained Visually — How it works, and why neural networks need it](https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739)
