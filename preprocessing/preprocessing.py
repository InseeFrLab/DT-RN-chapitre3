import geopandas as gpd
from sklearn.cluster import DBSCAN
import math
import numpy as np
import tensorflow as tf


def get_parcel_data(department):
    url_cadastre = "https://cadastre.data.gouv.fr/data/etalab-cadastre/2020-07-01/shp/departements/"
    parcel = gpd.read_file(
        url_cadastre + f"{department}/cadastre-{department}-parcelles-shp.zip"
    )
    return parcel


def get_urbanisation_data(department):

    # Ce notebook donne un exemple de calcul d'emprise de villes. Il ne s'agit pas de "tâche urbaine" au sens usuel du terme mais uniquement de calcul de zone convexe concentrant des bâtiments.

    # Par soucis de simplification :

    #     les regroupements de bâtiments sont calculés à partir des centroïdes des objets géométriques
    #     le calcul est réalisé sur un seul département. Il faudrait prendre en compte les départements limitrophes pour tenir compte des effets de bords.
    #     les regroupements sont obtenus à l'aide de l'algorithme DBSCAN

    url_cadastre = "https://cadastre.data.gouv.fr/data/etalab-cadastre/2020-07-01/shp/departements/"
    bats = gpd.read_file(
        url_cadastre + f"{department}/cadastre-{department}-batiments-shp.zip",
    )

    # Certains polygones peuvent être invalides (vecteurs superposés). le buffer(0) peut régler le problème
    bats["geometry"] = bats.geometry.buffer(0)

    # ## Calcul des regroupements de bâtiments ##
    # on recupere les centroïdes
    centres = [[b.centroid.x, b.centroid.y] for b in bats.geometry]
    # on recupere les aires (conservation des regroupements de batiments de plus de 5 000 m2)
    areas = [b.area for b in bats.geometry]
    clustering = DBSCAN(eps=200, min_samples=5000, n_jobs=5).fit(
        centres, sample_weight=areas
    )

    bats_cluster = bats.copy()
    bats_cluster["cluster"] = list(clustering.labels_)
    bats_cluster = bats_cluster.loc[bats_cluster.cluster != -1]
    # regroupement des batiments d'un même cluster dans un meme objet
    villages = bats_cluster.dissolve(by="cluster", aggfunc="sum")
    # calcul de l'enveloppe convexe des regroupements
    villages.geometry = [c.convex_hull for c in villages.geometry]

    # ## On sauvegarde les résultats :##
    # calcul des coordonées dans le systeme WSG84
    villages = villages.to_crs(epsg=2154)

    return villages


def get_parcel_limits(parcel):
    min_x = math.floor(np.min(parcel.bounds.minx) / 1024) * 1024
    min_y = math.floor(np.min(parcel.bounds.miny) / 1024) * 1024
    max_x = math.ceil(np.max(parcel.bounds.maxx) / 1024) * 1024
    max_y = math.ceil(np.max(parcel.bounds.maxy) / 1024) * 1024
    return {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y}


def transform_data(data, limits):
    transformed_geo = data.geometry.translate(
        xoff=-limits["min_x"], yoff=-limits["min_y"], zoff=0.0
    )
    transformed_geo = transformed_geo.scale(xfact=1 / 4, yfact=1 / 4, origin=(0, 0))
    return transformed_geo


def get_dim_image(limits):
    height = int((limits["max_y"] - limits["min_y"]) / 4)
    width = int((limits["max_x"] - limits["min_x"]) / 4)
    return (height, width)


def create_compact_object(borders, areas, cities):
    data = np.zeros((borders.shape[0], borders.shape[1], 3), dtype="uint8")
    data[:, :, 0] = borders
    data[:, :, 1] = cities
    data[:, :, 2] = areas
    return data


def get_significant_images(patch, parcel_area_threshold):
    # On s'assure qu'un certain pourcentage de la surface est dans une parcelle cadastrale
    pixel_per_image = np.sum(patch, axis=(1, 2))
    idx = pixel_per_image[:, 2] >= 256 * 256 * parcel_area_threshold
    return patch[idx]


def get_labels(patch, department, city_area_threshold):
    if department == 50:
        label = 0
    else:
        label = 1

    pixel_per_image = np.sum(patch, axis=(1, 2))
    idx = 256 * 256 * city_area_threshold < pixel_per_image[:, 1]
    return np.array([2 if image else label for image in idx])


def generate_artificial_images(images):
    artificial_data = (
        tf.image.flip_up_down(images),
        tf.image.flip_left_right(images),
        tf.image.random_flip_up_down(images),
        tf.image.random_flip_left_right(images),
    )
    return np.vstack(artificial_data)


def keep_only_enough_pixels(Y, X, min_pixel):
    non_zero_pixels = np.sum(X, axis=(1, 2)).reshape(X.shape[0])
    Y = Y[(non_zero_pixels >= min_pixel)]
    X = X[(non_zero_pixels >= min_pixel)]
    return (Y, X)
