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


history = model.fit(
    x_train,
    y_train,
    epochs=2,
    batch_size=218,
    shuffle=True,
    validation_data=(x_validation, y_validation),
)


nb_params = []
accuracies_train = []
accuracies_test = []
nb_couches = []
optimizer = "adam"
loss = "categorical_crossentropy"

for nb_couche in range(2, 7):

    model = tt.define_model_structure(
        (256, 256, 1),
        filters=[32] * nb_couche,
        kernel_sizes=[(5, 5)] * nb_couche,
        strides_conv=[(1, 1)] * nb_couche,
        padding="same",
        pool_sizes=[(2, 2)] * nb_couche,
        strides_maxpool=[(2, 2)] * nb_couche,
        dim_dense_layers=[128, 64, 3],
        activations=["relu"] * nb_couche + ["relu", "relu", "softmax"],
    )
    model.compile(optimizer=optimizer, loss=loss)

    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=218,
        shuffle=True,
        validation_data=(x_validation, y_validation),
    )

    accuracy_test = get_accuracy(model, x_test, y_test)
    accuracy_train = get_accuracy(model, x_train, y_train)

    nb_params.append(model.count_params())
    accuracies_test.append(accuracy_test)
    accuracies_train.append(accuracy_train)
    nb_couches.append(f"{nb_couche}")
