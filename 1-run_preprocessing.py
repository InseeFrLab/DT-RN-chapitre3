import preprocessing.preprocessing as prep
import geopandas as gpd
import numpy as np
from rasterio import features
from sklearn.feature_extraction import image
from sklearn.model_selection import train_test_split

dep_images_labels = {}

for dep in [50, 51]:
    # parcelles = prep.get_parcel_data(dep)
    parcelles = gpd.read_file(f"data/parcelles_{dep}.geojson")

    limites = prep.get_parcel_limits(parcelles)

    geo_parcelles = prep.transform_data(parcelles, limites)

    raster_areas = features.rasterize(
        geo_parcelles, prep.get_dim_image(limites), fill=0, all_touched=True
    )

    raster_bounds = features.rasterize(
        geo_parcelles.boundary, prep.get_dim_image(limites), fill=0, all_touched=True
    )

    villages = gpd.read_file(f"data/villages_{dep}.geojson")

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

    data_3d = prep.create_compact_object(raster_bounds, raster_areas, raster_villes)
    patch = image.extract_patches_2d(data_3d, (256, 256), max_patches=25000)
    patch = prep.get_significant_images(patch, 0.8)
    y = prep.get_labels(patch, dep, 0.5)
    artificial_images = prep.generate_artificial_images(patch[y == 2])
    patch = np.vstack((patch, artificial_images))
    # On ne conserve que la couche des contours des parcelles
    patch = patch[:, :, :, 0]
    y = np.hstack((y, 2 * np.ones(artificial_images.shape[0])))

    dep_images_labels[dep] = {"X": patch, "Y": y}

full_patch = np.vstack((dep_images_labels[50]["X"], dep_images_labels[51]["X"]))
y = np.hstack((dep_images_labels[50]["Y"], dep_images_labels[51]["Y"]))


x_train, x_test, y_train, y_test = train_test_split(
    full_patch, y, test_size=0.2, random_state=42, shuffle=True
)


x_train = x_train[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]
