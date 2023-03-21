# Deep Learning Experience

### Computer Vision
#### Image
##### Learning Rate & Batch Size Selection
- Learning rate:
  - `lr=0.1` the loss significantly increase from 0.6 to 56
  - `lr=0.0001` the loss is reduced, so can start with slow lr and increase
- Batch size:
  - Choose `batch_size = 4`: model is learning quite slow, and loss not reduced much
  - Choose `batch_size = 128`: model is learning quite fast, and the loss reduced faster
##### Augmentation
##### Convolution Layer
- Filter Size: the smaller the better (3x3 filter size yield a better accuracy then 7x7 one)
