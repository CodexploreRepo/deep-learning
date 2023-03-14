# Deep Learning Daily Knowledge
## Day 1
### Learning Rate & Batch Size Selection
#### Computer Vision: Image
- Learning rate:
  - `lr=0.1` the loss significantly increase from 0.6 to 56
  - `lr=0.0001` the loss is reduced, so can start with slow lr and increase
- Batch size:
  - Choose `batch_size = 4`: model is learning quite slow, and loss not reduced much
  - Choose `batch_size = 128`: model is learning quite fast, and the loss reduced faster
