import preprocessing.preprocessing as prep
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

dep_data = []

for dep in [50, 51]:
    if dep == 50:
        label = 0
    else:
        label = 1
    villages = prep.get_urbanisation_data(dep, 2154)
    villages.reset_index(inplace=True)
    parcelles = prep.get_parcel_data(dep)
    # On effectue une jointure (intersection des parcelles et des zones convexes des villages) :
    union = gpd.sjoin(parcelles, villages[["cluster", "geometry"]], how="left")
    # La catégorie est 3 si la parcelle intersecte un village, elle est 0 sinon (parcelles de la Manche hors villages) :
    union["categorie"] = [3 if c == c else label for c in union.cluster]
    parcelles = union[["id", "geometry", "categorie"]].drop_duplicates()

    dep_data.append(parcelles)

parcelles = pd.concat(dep_data)

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
confusion_matrix(np.argmax(np.array(y_test), axis=1), np.argmax(y_pred, axis=1))
