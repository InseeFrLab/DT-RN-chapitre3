# Chapitre 3 : Réseaux convolutifs et analyse d'images

[![Onyxia](https://img.shields.io/badge/Launch-Datalab-orange?logo=R)](https://datalab.sspcloud.fr/launcher/ide/rstudio?autoLaunch=true&onyxia.friendlyName=«dt-chap3»&security.allowlist.enabled=false&service.image.custom.enabled=true&service.image.pullPolicy=«Always»&service.image.custom.version=«thomasfaria%2Fdt-rn-chapitre3»)
[![Build](https://img.shields.io/github/actions/workflow/status/ThomasFaria/DT-RN-chapitre3/build-image.yaml?label=Build
)](https://hub.docker.com/repository/docker/thomasfaria/dt-rn-chapitre3)

## Prise en main
L'ensemble du codes sources utilisés dans ce chapitre est accompagné de son [image docker](https://hub.docker.com/repository/docker/thomasfaria/dt-rn-chapitre3) pour assurer une totale reproductibilité des résultats.

Celle-ci peut être utilisée pour vous éviter de télécharger les dépendances nécessaires à ce chapitre. Vous pouvez la récupérer avec la commande suivante :

```
docker pull thomasfaria/dt-rn-chapitre3
```

Il vous est également possible de télécharger les dépendances localement en utilisant le fichier *requirements.txt*, à l'aide de la commande ```pip install -r requirements.txt```.

Cependant nous vous recommendons fortement l'utilisation d'[Onyxia](https://github.com/InseeFrLab/onyxia-web), la plateforme *datascience* développée par l'[Insee](https://www.insee.fr/fr/accueil)). Pour ce faire vous pouvez suivre ces étapes :

- Etape 0: Allez sur [https://datalab.sspcloud.fr/home](https://datalab.sspcloud.fr/home). Cliquer sur **Sign In** et ensuite **create an account** avec votre adresse email institutionnelle ou académique.
- Etape 1: Cliquez [ICI](https://datalab.sspcloud.fr/launcher/ide/rstudio?autoLaunch=true&onyxia.friendlyName=«dt-chap4»&security.allowlist.enabled=false&service.image.custom.enabled=true&service.image.pullPolicy=«Always»&service.image.custom.version=«thomasfaria%2Fdt-rn-chapitre3») ou sur le badge orange en haut de la page pour lancer un service.
- Etape 2: **Ouvrez** le service et suivez les instructions affichées concernant l'**identifiant** et le **mot de passe**.
- Etape 3: **Clonez** le projet grâce à la commande suivant : ```git clone https://github.com/ThomasFaria/DT-RN-chapitre3.git```.

Tous les packages ont déjà été installés, vous devez en mesure de relancer tous les codes présents dans le projet.

## Organisation

Les programmes de cette partie sont écrit en **python**. Le réseau (CNN) est réalisé avec la librairie Keras tandis que la forêt aléatoire utilise la librairie sklearn.

Le code se divise en 4 scripts distincts :

    1-run_preprocessiong.py : Ce code consitue les échantillons de tests et d'entrainements de notre réseau de neurones. Pour cela :    
        - Il importe les données du cadastre (département de la Manche et de la Marne), ainsi que les empreintes des villages ;
        - Il transforme les données géométrique en images (rasters) ;
        - Il tire des échantillons d'images et crée les labels (suppression des images peu ou pas du tout remplies, identification des villes) ;
        - Il réalise le sur_échantillonage des villes, sous représentées.

    2-run_CNN.py : Ce code permets de définir la structure du réseau de neurones que l'on souhaite estimer à l'aide de la librairie Keras. Il montre comment la sélection des hyperparamètres (nombre de couches de convolutions, profondeur et taille des filtres) peut être réalisées. Enfin il met illustre également l'algorithme de Grad-CAM.
    
    3-run_random_forest.py : Ce script présente un modèle alternatif pour réaliser la même tâche de classification. Il estime une forêt aléatoire à l'aide de features préalablement construites.
