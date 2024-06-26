# Deep Learning Daily Knowledge

## Day 3

### Model Architecture

- Weight initialization: we can initialize the weights for layers in the networks

```Python
# initialize the weights to 0, which makes the training procedure slightly faster.
Sequential([
    Dense(1, kernel_initializer=tf.initializers.zeros)
])
```

## Day 2

### Training Debug

- Binary Classification: Accuracy & Loss of Validation during traininig is approach 90% and 0.001 respectively, but the trained model when doing inferencing only yield `"negative"` class
  - Root Cause: using `load_dataset(split='"train[:10%]"`, it only contains the `"negative"` class in both the training and validation set.
  - Lesson Learnt: always check the distribution of positive & negative classes in the training and test sets before training.
  ```Python
  # display % of training data with label=1
  np.array(dataset['train']['label']).sum()/len(dataset['train']['label'])
  ```

### Experiment Tracking

- [Experiment Tracking with Weights and Biases](https://www.kaggle.com/code/ayuraj/experiment-tracking-with-weights-and-biases/notebook)
- HuggingFace's `Trainer` class already include W&B tracking

### GPU on Mac

- In the **Activity Monitor** app on your Mac, choose `Window > GPU History`.
- Device: `mps`

### Model Architecture

#### Layers

- **Batch Norm** layer helps improve model performance
- **Drop Out** to reduce the overfitting in the training data

#### Weights

- You can initialize the weights of the model

## Day 1

### Training Parameters

#### Sampling for Faster Parameters & Hyper-paramters Tuning

```Python
# Split the dataset into training and validation sets
import random
trainset_size = 4000
testset_size = 1000

trainset_indices = random.sample(range(len(trainset)), trainset_size)
testset_indices = random.sample(range(len(testset)), testset_size)

# obtaining subset of data
trainset_subset = torch.utils.data.Subset(trainset, trainset_indices)
augmented_trainset_subset = torch.utils.data.Subset(augmented_trainset, trainset_indices)
testset_subset = torch.utils.data.Subset(testset, testset_indices)
```

#### Batch Size

- Batch size: depends on the dataset
  - From small to bigger:
    - Choose `batch_size = 4`: model is learning quite slow, and loss not reduced much
    - Choose `batch_size = 64, 128`: model is learning quite fast, and the loss reduced faster
  - From bigger to smaller:
    - Choose `batch_size = 256`: val_accuracy & val_loss is fluctuating
    - Choose `batch_size = 128 -> 64`: val_accuracy start improving

#### Optimizer

- Experience: need to determine the optimizer that is suitable for the dataset first before choosing the learning rate, as different optimizer will work well with different learning rate.
  - **SGD** default `learning_rate=0.01`
  - **Adam** default `learning_rate=0.001` a fast and efficient optimizer
    - Note: Have to set a very small learning rate `lr=1e-4` so the loss won't begin to diverge after decrease to a point.
    - It's very useful to Adam with learning rate decay. However, be careful when using weight decay with the vanilla Adam optimizer, as it appears that the vanilla Adam formula is wrong when using weight decay
  - **AdamW**: use the AdamW variant when you want to use Adam with weight decay.

##### SGD Optimiser

- The SGD or Stochastic Gradient Optimizer is an optimizer in which the weights are updated for each training sample or a small subset of data.
- Pytorch Syntax
  - **params** (iterable) — These are the parameters that help in the optimization.
    lr (float) — This parameter is the learning rate
  - **momentum** (float, optional) — Here we pass the momentum factor
  - **weight_decay** (float, optional) — This argument is containing the weight decay
  - **dampening** (float, optional) — To dampen the momentum, we use this parameter

```Python
torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
```

##### Adam Optimiser

- Adam Optimizer uses both momentum and adaptive learning rate for better convergence. This is one of the most widely used optimizer for practical purposes for training neural networks.
- Pytorch Syntax
  - **params** (`Union[Iterable[Tensor], Iterable[Dict[str, Any]]]`) – These are the iterable parameters that help in optimization
  - **lr** (float) – Learning rate helping optimization (default: 1e-3)
  - **betas** (`Tuple[float, float]`) – This parameter is used for calculating and running the averages for gradient (default: (0.9, 0.999))
  - **beta3** (float) – Smoothing coefficient (default: 0.9999)
  - **eps** (float) – For improving the numerical stability (default: 1e-8)
  - **weight_decay** (float) – For adding the weight decay (L2 penalty) (default: 0)

```Python
torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
```

##### Adagrad Optimiser

- Adaptive Gradient Algorithm (Adagrad) is an algorithm for gradient-based optimization where each parameter has its own learning rate that improves performance on problems with sparse gradients.
- Pytorch Syntax
  - **params** (`Union[Iterable[Tensor], Iterable[Dict[str, Any]]]`) – These are the iterable parameters that help in optimization
  - **lr** (float) – Learning rate helping optimization (default: 1e-3)
  - **lr_decay** (float, optional) – learning rate decay (default: 0)
  - **eps** (float) – For improving the numerical stability (default: 1e-8)
  - **weight_decay** (float) – For adding the weight decay (L2 penalty) (default: 0)

```Python
torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
```

#### Learning Rate

- `lr=0.1` the loss significantly increase from 0.6 to 56
- `lr=0.001` the loss is reduced, so can start with slow lr and increase
- **Learning Rate Scheduler**:to reduce the learning rate when the loss starts flattening to help model to learn if it stucks local minimum
  - Observation: longer schedule improves the final performance of the network
  - Methodology: use a learning rate schedule in which we ramp-up, hold, then exponentially decay the learning rate until it reaches 1/100 of its maximum value
  - Conclusion: The learning rate schedule turns out to be an _important factor_ in determining the performance of ASR networks, especially so when _augmentation is present_.

```Python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
```

### Computer Vision

#### Image

##### Data Augmentation

- Tip 1: training process with the augmentations has a better result than withoutaugmentations
- Tip 2: need to find the best combination of augmentations, stacking a lot of augmentations do not guarantee yielding a better result.

```Python
# Define the normal transformations to be applied to the test data
transform = transforms.Compose([
    transforms.ToTensor(), # convert from [0,255] to [0,1] & to Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # data normalization
])


# Define the transforms for data augmentation to be applied to the train data
augmented_transform = transforms.Compose([
      transforms.ToPILImage(), # added in if the image is in Numpy format, so need to convert to PIL for operations like crop, flip to work
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      # transforms.ColorJitter(), # not good as the above two RandomCrop & RandomHorizontalFlip alone
      transforms.ToTensor(), # convert from [0,255] to [0,1] & to Tensor
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load the CIFAR-10 dataset without transformations
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# Load the CIFAR-10 dataset and apply transformations
augmented_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=augmented_transform)
# Combine the original CIFAR-10 dataset with the augmented dataset to create a larger dataset
combined_augmented_trainset = torch.utils.data.ConcatDataset([trainset, augmented_trainset])
```

#### Convolution Layer

- Filter Size: the smaller the better (3x3 filter size yield a better accuracy then 7x7 one)
