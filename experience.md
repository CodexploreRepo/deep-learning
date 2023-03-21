# Deep Learning Experience
## Training Parameters
### Batch Size 
- Batch size: depends on the dataset
  - Choose `batch_size = 4`: model is learning quite slow, and loss not reduced much
  - Choose `batch_size = 64, 128`: model is learning quite fast, and the loss reduced faster
### Learning Rate
- Learning rate:
  - `lr=0.1` the loss significantly increase from 0.6 to 56
  - `lr=0.001` the loss is reduced, so can start with slow lr and increase
### Optimizer

## Model Architecture
- **Batch Norm** layer helps improve model performance
## Computer Vision
### Image
#### Data Augmentation
- Tip 1: training process with the augmentations has a better result than withoutaugmentations
- Tip 2: need to find the best combination of augmentations, stacking a lot of augmentations do not guarantee yielding a better result.

```Python
# Define the normal transformations to be applied to the test data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Define the transforms for data augmentation to be applied to the train data
augmented_transform = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      # transforms.ColorJitter(), # not good as the above two RandomCrop & RandomHorizontalFlip alone
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
```
#### Convolution Layer
- Filter Size: the smaller the better (3x3 filter size yield a better accuracy then 7x7 one)

