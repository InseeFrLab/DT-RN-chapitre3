import numpy as np
import training.training as tt
import plotting.plotting as plotting
import matplotlib.pyplot as plt
from matplotlib.image import imsave
from PIL import Image

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


## Choix nombre de couche de convolution
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

    accuracy_test = tt.get_accuracy(model, x_test, y_test)
    accuracy_train = tt.get_accuracy(model, x_train, y_train)

    nb_params.append(model.count_params())
    accuracies_test.append(accuracy_test)
    accuracies_train.append(accuracy_train)
    nb_couches.append(f"{nb_couche}")


plt.figure(figsize=(15, 15))

acc_train = [np.float(a) * 100 for a in accuracies_train]
acc_test = [np.float(a) * 100 for a in accuracies_test]
lnb_params = [np.log(p) for p in nb_params]


plt.plot(lnb_params, acc_train, label="Echantillon d'entrainement")
plt.plot(lnb_params, acc_test, label="Echantillon test")
plt.legend()

labx = [f"{n} ({c} couches)" for (n, c) in zip(nb_params, nb_couches)]

plt.xticks(lnb_params, labx, rotation=90)
plt.xlabel("Nombre de parametres")
plt.ylabel("Precision")

## Choix taille des filtres
nb_params = []
accuracies_train = []
accuracies_test = []
filter_size = []
optimizer = "adam"
loss = "categorical_crossentropy"
nb_couches = 5

for size in range(3, 10, 2):

    model = tt.define_model_structure(
        (256, 256, 1),
        filters=[32] * nb_couche,
        kernel_sizes=[(size, size)] * nb_couche,
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

    accuracy_test = tt.get_accuracy(model, x_test, y_test)
    accuracy_train = tt.get_accuracy(model, x_train, y_train)

    nb_params.append(model.count_params())
    accuracies_test.append(accuracy_test)
    accuracies_train.append(accuracy_train)
    filter_size.append(f"{size}")


plt.figure(figsize=(15, 15))

acc_train = [np.float(a) * 100 for a in accuracies_train]
acc_test = [np.float(a) * 100 for a in accuracies_test]
lnb_params = [np.log(p) for p in nb_params]


plt.title("Taille de la fenêtre\n (avec 5 couches de convolution)")

plt.plot(lnb_params, acc_train, label="Echantillon d'entrainement")
plt.plot(lnb_params, acc_test, label="Echantillon test")
plt.legend()

labx = [f"{n} ({s})" for (n, s) in zip(nb_params, filter_size)]

plt.xticks(lnb_params, labx, rotation=90)
plt.xlabel("Nombre de parametres")
plt.ylabel("Precision")

# see https://github.com/maxpumperla/hyperas

# Final model
nb_couche = 6
optimizer = "adam"
loss = "categorical_crossentropy"

model = tt.define_model_structure(
    (256, 256, 1),
    filters=[16, 16, 32, 32, 48, 48],
    kernel_sizes=[(7, 7)] * nb_couche,
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
    epochs=1,
    batch_size=218,
    shuffle=True,
    validation_data=(x_validation, y_validation),
)

accuracy_test = tt.get_accuracy(model, x_test, y_test)
accuracy_train = tt.get_accuracy(model, x_train, y_train)

# Mise en oeuvre de l'algorithme Grad-Cam
predictions = model.predict(
    x_test[
        :15,
    ]
)
Y_predicted = np.argmax(predictions, axis=1)
Y_predicted_label = [
    "ville" if c == 2 else "Manche" if c == 0 else "Marne" for c in Y_predicted
]
Y_true_label = [
    "ville" if c == 2 else "Manche" if c == 0 else "Marne"
    for c in np.argmax(y_test, axis=1)
]

names = [l.name for l in model.layers]
last_conv_layer_name = [n for n in names if n[:4] == "conv"][-1]

no = [k for k in range(len(names)) if names[k] == last_conv_layer_name][0]
classifier_layer_names = names[(no + 1) : -1]


figure, axis = plt.subplots(3, 5, figsize=(20, 10))
for i in range(15):
    img_path = x_test[
        i,
    ].reshape(256, 256)

    imsave("outfile.jpg", img_path, cmap="binary")
    heatmap = plotting.make_gradcam_heatmap(
        x_test[
            i : i + 1,
        ],
        model,
        last_conv_layer_name,
        classifier_layer_names,
    )
    superimposed_img = plotting.get_images(heatmap)
    # Save the superimposed image
    superimposed_img.save("parcelles_cam.jpg")
    # load the image
    image = Image.open("parcelles_cam.jpg")
    n = int(np.floor(i / 5))
    axis[n, i - 5 * n].imshow(image)
    if Y_true_label[i] == Y_predicted_label[i]:
        axis[n, i - 5 * n].set_title(
            f"vrai :{Y_true_label[i]}, predit: {Y_predicted_label[i]}", color="green"
        )
    else:
        axis[n, i - 5 * n].set_title(
            f"vrai :{Y_true_label[i]}, predit: {Y_predicted_label[i]}", color="red"
        )
plt.show()
plt.close()
