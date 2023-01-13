import preprocessing.preprocessing as prep
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

dep_data = []

for dep in [50, 51]:

    villages = prep.get_urbanisation_data(dep, 2154)
    villages.reset_index(inplace=True)
    parcelles = prep.get_parcel_data(dep)
    # On effectue une jointure (intersection des parcelles et des zones convexes des villages) :
    union = gpd.sjoin(parcelles, villages[["cluster", "geometry"]], how="left")
    # La catégorie est 3 si la parcelle intersecte un village, elle est 0 sinon (parcelles de la Manche hors villages) :
    union["categorie"] = [3 if c == c else 0 for c in union.cluster]
    parcelles = union[["id", "geometry", "categorie"]].drop_duplicates()

    dep_data.append(parcelles)

parcelles = pd.concat(dep_data)


def get_features(data):
    area = data.geometry.area
    perimeter = data.geometry.length
    categ = data.categorie
    gravelius = perimeter / np.sqrt(area)
    cx = data.geometry.centroid.x.apply(lambda x: int(x / 1024))
    cy = data.geometry.centroid.y.apply(lambda x: int(x / 1024))

    q1_gravelius = gravelius.quantile(0.33)
    q3_gravelius = gravelius.quantile(0.66)

    q1_area = area.quantile(0.33)
    q3_area = area.quantile(0.66)

    gravelius_code = pd.Series(
        [
            "allongee"
            if a < q1_gravelius
            else "compacte"
            if a >= q3_gravelius
            else "intermediaire"
            for a in gravelius.values
        ]
    )
    area_code = pd.Series(
        [
            "petite" if a < q1_area else "grande" if a >= q3_area else "moyenne"
            for a in area.values
        ]
    )

    gravelius = pd.get_dummies(gravelius_code)
    area = pd.get_dummies(area_code)
    categ = pd.get_dummies(categ)
    categ.columns = ["manche", "marne", "ville"]

    features = pd.concat([cx, cy, gravelius, area, categ], axis=1)

    features = pd.DataFrame(features.groupby(["cx", "cy"]).sum())

    somme = features.sum(axis=1).values.copy()
    for c in features.columns:
        features.loc[:, c] = 3 * features.loc[:, c] / somme

    for i in features.index:
        if features.loc[i, "ville"] > 0.8:
            features.loc[i, "ville"] = 1
            features.loc[i, "manche"] = 0
            features.loc[i, "marne"] = 0
        else:
            if features.loc[i, "manche"] > 0.5:
                features.loc[i, "ville"] = 0
                features.loc[i, "manche"] = 1
                features.loc[i, "marne"] = 0
            else:
                features.loc[i, "ville"] = 0
                features.loc[i, "manche"] = 0
                features.loc[i, "marne"] = 1

    return features


# Model training
features = prep.get_features(parcelles)

X_train, X_test, y_train, y_test = train_test_split(
    features[["allongee", "compacte", "intermediaire", "grande", "moyenne", "petite"]],
    features[["manche", "marne", "ville"]],
    test_size=0.20,
    random_state=42,
)
clf = RandomForestClassifier(max_depth=5, random_state=0)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

y_pred = clf.predict(X_test)
confusion_matrix(y_test, y_pred)
# gravelius.apply(lambda x: "allongee" if x < gravelius.quantile(0.33) else "compacte")
