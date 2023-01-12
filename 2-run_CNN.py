import numpy as np
import training.training as tt

x_train = np.load("data/x_train.npy")
y_train = np.load("data/y_train.npy")
x_test = np.load("data/x_test.npy")
y_test = np.load("data/y_test.npy")
x_validation = np.load("data/x_validation.npy")
y_validation = np.load("data/y_validation.npy")

model = tt.define_model_structure(
    (256, 256, 1),
    filters=[32, 32, 32, 32, 32],
    kernel_sizes=[(5, 5), (5, 5), (5, 5), (5, 5), (5, 5)],
    strides_conv=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
    padding="same",
    pool_sizes=[(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)],
    strides_maxpool=[(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)],
    dim_dense_layers=[128, 64, 3],
    activations=["relu", "relu", "relu", "relu", "relu", "relu", "relu", "softmax"],
)

optimizer = "adam"
loss = "categorical_crossentropy"
model.compile(optimizer=optimizer, loss=loss)

model.fit(
    x_train,
    y_train,
    epochs=2,
    batch_size=218,
    shuffle=True,
    validation_data=(x_validation, y_validation)
)
