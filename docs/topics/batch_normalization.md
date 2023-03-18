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

![image](https://user-images.githubusercontent.com/64508435/226077638-5c20103d-46ca-47a1-86c0-d05fd5659588.png)

## Reference:
- [Batch Norm Explained Visually — How it works, and why neural networks need it](https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739)
