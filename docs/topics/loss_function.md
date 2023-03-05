# Loss Function
## Classification
- **Categorical cross-entropy**: is used when true labels are one-hot encoded, for example, we have the following true values for 3-class classification problem `[1,0,0]`, `[0,1,0]` and `[0,0,1]`.
```Python
# need to convert class vectors to binary class matrices: 1 => [0 1 0 0 ...] 
# 1              => sparse_categorical_crossentropy
# [0 1 0 0 ...]  => categorical_crossentropy
y_train_label = keras.utils.to_categorical(y_train, class_nums)
y_test_label = keras.utils.to_categorical(y_test, class_nums)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
- **Sparse categorical cross-entropy**: truth labels are integer encoded, for example, `[1]`, `[2]` and `[3]` for 3-class problem.
```Python
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```
