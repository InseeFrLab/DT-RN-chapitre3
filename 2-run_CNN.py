import numpy as np
import training.training as tt
import plotting.plotting as plotting
import matplotlib.pyplot as plt
from matplotlib.image import imsave
from PIL import Image

# On cherche à batir un réseau de neurones afin de distinguer si les images
# du cadastre appartiennent plutôt à la Manche (50) ou à la Champagne (51).
# Visuellement, les parcelles ne semblent pas avoir la même forme. Il faudrait
# que le réseau de neurones soit capable de détecter la structure des parcelles
# et pas seulement la densité de celles-ci (auquel cas, un simple dénombrement
# des pixels allumés suffirait à faire la distinction entre les deux types
# d'occupation des sols).

# On importe les données précédemment créées
x_train = np.load("data/x_train.npy")
y_train = np.load("data/y_train.npy")
x_test = np.load("data/x_test.npy")
y_test = np.load("data/y_test.npy")
x_validation = np.load("data/x_validation.npy")
y_validation = np.load("data/y_validation.npy")

# Le modèle est défini de façon séquentielle à l'aide de la librairie Keras.
# Un premier modèle est défini. Il comporte 5 couches de convolutions.
# La taille des filtres est de 5 pixels (kernel_sizes=(5,5)). Les filtres se
# déplacent de 1 pixel à chaque fois (strides = (1,1)). Parfois, l'image est
# prolongée sur les côtés afin de pouvoir calculer une valeur sur les
# bords (padding='same'). La taille des filtres (filters) est fixée arbitrairement
# à 32. Un perceptron multicouche est utilisé afin de trier les images.
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

# Le réseau est compilé puis estimé (il voit 10 fois l'échantillon initial
#  (epochs=10) sur des ensembles de 218 images (batch_size).
model.compile(optimizer="adam", loss="categorical_crossentropy")
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=218,
    shuffle=True,
    validation_data=(x_validation, y_validation),
)


#### Choix des hyper-paramètres ####

## 1- Choix du nombre de couche de convolution (nb_couche)
nb_params = []
accuracies_train = []
accuracies_validation = []
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

    accuracy_validation = tt.get_accuracy(model, x_validation, y_validation)
    accuracy_train = tt.get_accuracy(model, x_train, y_train)

    nb_params.append(model.count_params())
    accuracies_validation.append(accuracy_validation)
    accuracies_train.append(accuracy_train)
    nb_couches.append(f"{nb_couche}")

# On peut visualiser la performance du réseau en fontion du nombre de couche
plt.figure(figsize=(15, 15))

acc_train = [np.float(a) * 100 for a in accuracies_train]
acc_validation = [np.float(a) * 100 for a in accuracies_validation]
lnb_params = [np.log(p) for p in nb_params]


plt.plot(lnb_params, acc_train, label="Echantillon d'entrainement")
plt.plot(lnb_params, acc_validation, label="Echantillon de validation")
plt.legend()

labx = [f"{n} ({c} couches)" for (n, c) in zip(nb_params, nb_couches)]

plt.xticks(lnb_params, labx, rotation=90)
plt.xlabel("Nombre de parametres")
plt.ylabel("Precision")

## 2- Choix taille des filtres
nb_params = []
accuracies_train = []
accuracies_validation = []
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

    accuracy_validation = tt.get_accuracy(model, x_validation, y_validation)
    accuracy_train = tt.get_accuracy(model, x_train, y_train)

    nb_params.append(model.count_params())
    accuracies_validation.append(accuracy_validation)
    accuracies_train.append(accuracy_train)
    filter_size.append(f"{size}")

# On peut visualiser la performance du réseau en fontion de la taille des filtres
plt.figure(figsize=(15, 15))

acc_train = [np.float(a) * 100 for a in accuracies_train]
acc_validation = [np.float(a) * 100 for a in accuracies_validation]
lnb_params = [np.log(p) for p in nb_params]


plt.title("Taille de la fenêtre\n (avec 5 couches de convolution)")

plt.plot(lnb_params, acc_train, label="Echantillon d'entrainement")
plt.plot(lnb_params, acc_validation, label="Echantillon de validation")
plt.legend()

labx = [f"{n} ({s})" for (n, s) in zip(nb_params, filter_size)]

plt.xticks(lnb_params, labx, rotation=90)
plt.xlabel("Nombre de parametres")
plt.ylabel("Precision")

# Pour en savoir plus, voir https://github.com/maxpumperla/hyperas. C'est une
# libraire qui permet des réaliser un grid search de manière optimisée des
# hyper-paramètres d'un modèle compilé avec Keras.

# Définition du modèle final choisi après investigations.
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

#### Mise en oeuvre de l'algorithme Grad-Cam ####

# Les réseaux de neurones ont un aspect "boîte noire". Il n'est pas possible
# d'identifier immédiatement les variables qui impacte le plus la prédiction.
# Dans le cas des convnets, plusieurs algorithmes ont été développés afin
# d'identifier les structures perçues par le réseau.
# L'une de ces méthodes est l'algorithme Grad-Cam présenté ici.

# On récupère les prédictions sur les 15 première images
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

# On affiche le nom des couches du modèle (final) précédent
names = [l.name for l in model.layers]

# On récupère le nom de la dernière couche de convolution
last_conv_layer_name = [n for n in names if n[:4] == "conv"][-1]

# On récupère l'ensemble des couches entre la dernière couche de convolution
#  (inclus) et le softmax final (exclus).
no = [k for k in range(len(names)) if names[k] == last_conv_layer_name][0]
classifier_layer_names = names[(no + 1) : -1]

# On visualise le résultat de l'algorithme
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
