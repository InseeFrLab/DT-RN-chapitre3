import preprocessing.preprocessing as prep
import numpy as np
from rasterio import features
from sklearn.feature_extraction import image
from sklearn.model_selection import train_test_split
import keras

# Les données sont disponibles par départements sur le site data.gouv.fr.
# Les fichiers mis à disposition sont des fichiers shapefile (polygones).
# Il faut les traduire au format numérique (raster).
# Le réseau de neurones nécessite en entrée un échantillon d'images qu'il doit
# apprendre à classer :
#     - des images raster, qui sont des nombres (1 ou 0) varaint en fonction de
#       l'intensité des pixels. Sur ces images, les lignes délimitant les parcelles
#       seront des 1 tandis que les vides seront des 0.
#     - ces images doivent être représentatives des images françaises c'est-à-dire
#       qu'elles doivent couvrir l'ensemble du territoire (échantillon d'images de
#       plusieurs départements différents en termes de structure) et l'ensemble des
#       patterns ou structure de parcelles cadastrales.


# Les fichiers sont disponibles par départements.
# Seuls les départements 50 (Manche) et 51 (Marne) sont retenus pour la constitution
# de l'échantillon d'apprentissage. Ces deux départements nous ont paru suffisamment
# représentatives des structures champs ouverts et champs clôturés afin d'entraîner
# le réseau.

dep_images_labels = {}

for dep in [50, 51]:
    print(f"\nBuilding dataset for department {dep}...\n")
    parcelles = prep.get_parcel_data(dep)
    parcelles.to_file(f"data/villages_{dep}.geojson", driver="GeoJSON")
    print("\n*** Parcels data retrieved!\n")

    limites = prep.get_parcel_limits(parcelles)

    geo_parcelles = prep.transform_data(parcelles, limites)

    raster_areas = features.rasterize(
        geo_parcelles, prep.get_dim_image(limites), fill=0, all_touched=True
    )
    np.save(f"data/raster_areas_{dep}.npy", raster_areas)

    raster_bounds = features.rasterize(
        geo_parcelles.boundary, prep.get_dim_image(limites), fill=0, all_touched=True
    )
    np.save(f"data/raster_bounds_{dep}.npy", raster_bounds)

    villages = prep.get_urbanisation_data(dep, 2154)
    villages.to_file(f"data/villages_{dep}.geojson", driver="GeoJSON")
    print("\n*** Cities data retrieved!\n")

    filter = [
        (limites["max_x"] >= geo.bounds[0])
        and (limites["max_y"] >= geo.bounds[1])
        and (limites["min_x"] <= geo.bounds[2])
        and (limites["min_y"] <= geo.bounds[3])
        for geo in villages.geometry
    ]

    geo_villages = prep.transform_data(villages[filter], limites)

    raster_villes = features.rasterize(
        geo_villages, prep.get_dim_image(limites), fill=0, all_touched=True
    )
    np.save(f"data/raster_villes_{dep}.npy", raster_villes)
    print("\n*** Rasters built!\n")

    data_3d = prep.create_compact_object(raster_bounds, raster_areas, raster_villes)

    # La fonction extract_patches_2d du package scikit-learn permet d'échantillonner
    # une image (ici on choisit des images de 256 x 256 pixels).
    # 25 000 images sont sélectionnées dans chaque département.
    patch = image.extract_patches_2d(data_3d, (256, 256), max_patches=25000)

    # On sélectionne les images dont plus de 80 % de la surface sont dans une
    # parcelle cadastrale (on évite les "trous" du cadastre)
    patch = prep.get_significant_images(patch, 0.8)
    # Identification des carreaux urbains = plus de 50 % de l'image en ville
    y = prep.get_labels(patch, dep, 0.5)
    print("\n*** Data labeled!\n")

    # Le nombre d'image de parcelles "urbaines" est sous représenté dans
    # l'échantillon. On génére artificiellement de nouvelles images afin
    # d'accroître leur nombre (data augmentation). Pour cela, on permute
    # (flip) les pixels des images (verticalement, horizontalement ou en diagonale).
    artificial_images = prep.generate_artificial_images(patch[y == 2])
    patch = np.vstack((patch, artificial_images))
    # On ne conserve que la couche des contours des parcelles
    patch = patch[:, :, :, 0]
    y = np.hstack((y, 2 * np.ones(artificial_images.shape[0])))
    print("\n*** Artificial data added!\n")

    dep_images_labels[dep] = {"X": patch, "Y": y}

# On rassemble les données des deux départements
full_patch = np.vstack((dep_images_labels[50]["X"], dep_images_labels[51]["X"]))
y = np.hstack((dep_images_labels[50]["Y"], dep_images_labels[51]["Y"]))

# On garde que les images qui contiennent suffisament de pixels
y, full_patch = prep.keep_only_enough_pixels(y, full_patch, 1000)

# On passe les labels au format catégoriel
y = keras.utils.to_categorical(y, 3)

# On créer les fichiers de test
x_train, x_test, y_train, y_test = train_test_split(
    full_patch, y, test_size=0.2, random_state=42, shuffle=True
)

# On créer les fichiers d'entrainement et de validation
x_train, x_validation, y_train, y_validation = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42, shuffle=True
)

# On rajoute un nouvel axe pour les besoins de l'apprentissage (les images ont
# souvent un axe supplémentaire pour les couleurs : il y a par exemple trois
# canaux pour rouge, vert et bleu)).
x_train = x_train[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]
x_validation = x_validation[:, :, :, np.newaxis]

# On sauvegarde les données
np.save("data/x_train.npy", x_train)
np.save("data/y_train.npy", y_train)
np.save("data/x_test.npy", x_test)
np.save("data/y_test.npy", y_test)
np.save("data/x_validation.npy", x_validation)
np.save("data/y_validation.npy", y_validation)
