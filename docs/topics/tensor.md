# Tensor

## Tensor Basics

- `dtype`, `ndim` (rank), and `shape`

## Tensor Operations

#### Broadcasting

- Broadcast axes are added to the smaller tensor to match the ndim of the larger tensor for addition those 2 tensors.
  - The smaller tensor is repeated alongside these new axes to match the full shape of the larger tensor

```Python
x = np.random.random((2,5))
y = np.random.random((5,))

# X-shape: (2, 5)
# [[0.70258491 0.95322527 0.93125586 0.58870387 0.37808884]
#  [0.58336897 0.59729941 0.27867715 0.63416763 0.6219389 ]]
# y-shape: (5,)
#  [0.72331562 0.86843338 0.6167525  0.70972164 0.17234472]

"""Broacasting"""
# add an empty first axis to y, whose shape becomes (1, 5)
y = np.expand_dims(y, axis=0)
# y-shape: (1, 5)
# [[0.72331562 0.86843338 0.6167525  0.70972164 0.17234472]]

# repeate y 2 times, i.e [y, y] along this new axis (the first axis, or axis=0), so that y.shape end up (2,5)
y = np.concatenate([y]*2, axis=0)

# y-shape: (2, 5)
# [[0.72331562 0.86843338 0.6167525  0.70972164 0.17234472]
#  [0.72331562 0.86843338 0.6167525  0.70972164 0.17234472]]
```

#### Reshaping

- Reshaping a tensor means rearranging its rows and columns to match a target shape.

```Python
train_images = train_images.reshape((60000, 28 * 28))
```

##### Special Reshaping: Transpose

- Transposing a matrix means exchanging its rows and its columns

```Python
x = np.transpose(x)
```
