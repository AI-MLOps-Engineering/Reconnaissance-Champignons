'''
Script permettant de classer les images dans les sous-répertoires relatifs à leur 'species', pour les champignons du Top 30 en France.

Cela permettra d'utiliser l'utilitaire 'image_dataset_from_directory' pour appeler les images dans les futurs modèles et gagner en
rapidité d'exécution.

+ AJout d'un filtre à partir de ResNet50 (ImageNet) pour retirer les images parasites

'''


# import des librairies nécessaires
import os
import shutil
import pandas as pd
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from sklearn.utils import resample # pour les méthodes de réchantillonnage (over-sampling et under-sampling)


# Définition des chemins d'accès
rep_main_csv = 'C:\\Users\\Pregassame\\Documents\\mushroom_dataset\\' # Chemin vers le fichier csv principal (à part car trop gros pour mettre sur git)
rep_data = "C:\\Users\\Pregassame\\Documents\\mushroom_dataset\\" # Chemin vers le fichier CSV 'champignons_france_top30.csv et le dataset "final dataset_30_species.csv"
rep_img_src = 'C:\\Users\\Pregassame\\Documents\\DataScience_images\\Fungi\\' # Chemin vers l'ensemble des images de champignons 
rep_img = 'C:\\Users\\Pregassame\\Documents\\DataScience_images_Especes3\\' # Chemin vers l'ensemble des images de champignons CLASSEES

# Lecture du dataset 'observations_mushroom.csv'
mushroom = pd.read_csv(rep_main_csv+'observations_mushroom.csv', sep =  ",")

# Lecture du dataset 'top_30_France.csv'
top_30_France = pd.read_csv(rep_data+'champignons_france_top30.csv', sep = ';', encoding='latin-1')

# Fusion des 2 Datasets
mushroom = mushroom.merge(right = top_30_France, left_on='gbif_info/species', right_on='Nom scientifique', how = 'inner')

# Filtre pour un indice de confiance >= 92
mushroom = mushroom.loc[mushroom['gbif_info/confidence'] >= 92]
mushroom.to_csv(rep_data+"dataset_30_species.csv", index=False, encoding="utf-8")

# Simplification du fichier
mushroom = mushroom[['image_lien', 'gbif_info/species', 'Nom commun', 'Statut', 'Habitat typique']]
mushroom = mushroom.dropna()
mushroom = mushroom.rename({'image_lien' : 'image', 'gbif_info/species' : 'target'}, axis = 1)

# Définition des catégories qui nous intéressent
categories = mushroom['target'].unique()

# Nombre d'images par espèces avant over/under-sampling :
print("Nombre d'images par espèce avant over/under-sampling :\n")
for category in categories:
    order_list = mushroom.loc[mushroom['target']==category]
    count = len(order_list)
    print(f"L'espèce {category} compte {count} enregistrements")

# Charger le modèle ResNet50 pré-entraîné sur ImageNet
model = ResNet50(weights="imagenet")

# Liste des classes "champignons" dans ImageNet
fungi_classes = {
    "bolete", "agaric", "gyromitra", "stinkhorn",
    "earthstar", "hen-of-the-woods", "coral fungus", "mushroom"
}

def champi(img_path, top=3): # on check si il y a champignon dans le top 3 des recos
    """
    Vérifie si une image contient un champignon
    Args:
        img_path: chemin vers l'image
        top: nombre de prédictions à analyser
    Returns:
        True si l'image est identifiée comme champignon
    """
    # Charger et prétraiter l'image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)  # permet d'avoir x.shape = (1, 224, 224, 3) compatible avec le modèle
    x = preprocess_input(x)

    # Prédictions
    preds = model.predict(x)
    decoded = decode_predictions(preds, top=top)[0]

    # Vérifie si une des prédictions est un champignon
    return any(label in fungi_classes for _, label, _ in decoded)

for i, row in mushroom.iterrows():
    image_path = os.path.join(rep_img_src, row['image'])

    if champi(image_path):
        mushroom.loc[i, ["Champi"]] = True
    else :
        mushroom.loc[i, ["Champi"]] = False

# Filtre : on retire les images parasites
mushroom = mushroom.loc[mushroom['Champi'] == True]

# Paramétrage de l'under-sampling / under-sampling
# Objectif : réduire à 900 images les catégories sur-représentées et augmenter à 850 images les catégories sous-représentées

# random_state :
rs = 42

# Liste pour stocker les DataFrames équilibrés
list_temp = []

for category in categories:
    df_category = mushroom[mushroom['target']==category].copy()
    size = len(df_category)
    
    # Under-sampling
    if size > 900:
        df_temp = resample(
            df_category,
            replace = False,
            n_samples = 900,
            random_state = rs
        )
    
    # Over-sampling
    elif size < 850:
        df_temp = resample(
            df_category,
            replace = True,
            n_samples = 850,
            random_state = rs
        ).reset_index(drop=True)


    # RAS si 850 <= size <= 900
    else :
        df_temp = df_category
       

    list_temp.append(df_temp)


# Fusion de tous les DataFrames
mushroom_2 = pd.concat(list_temp, ignore_index=True)


# Création de la colonne image sans-doublon "image_unique" avec incrémentation, qui permettra de dupliquer les photos
dico = {}
new = []
for val in mushroom_2['image']:
    if val not in dico:
        dico[val] = 0
        new.append(val)
    else:
        dico[val] += 1
        new.append(f"{val.replace('.jpg', '')}_{dico[val]}.jpg")

mushroom_2['image_unique'] = new

# Création des sous-dossiers pour chaque catégorie
for category in categories:
    category_path = os.path.join(rep_img, category)
    os.makedirs(category_path, exist_ok=True)

# Déplacement des fichiers dans leur dossier de catégorie

## import de bibliothèque pour transormation
import tensorflow as tf
from tensorflow.keras.layers import RandomFlip
from tensorflow.keras.layers import RandomZoom
from tensorflow.keras.layers import RandomRotation
from tensorflow.keras.layers import RandomBrightness
from tensorflow.keras.layers import RandomContrast
from tensorflow.keras.layers import RandomTranslation 

# Définir les couches de transformation
random_translation = RandomTranslation(0.2, 0.2)   # Étirement
random_zoom = RandomZoom(0.2)                      # Agrandissement
random_flip = RandomFlip("horizontal")             # Retournement horizontal
random_rotation = RandomRotation(0.2)              # Rotation
random_contrast = RandomContrast(0.2)              # Contraste

for _, row in mushroom_2.iterrows():
    src_path = os.path.join(rep_img_src, row['image'])
    dst_path = os.path.join(rep_img, row['target'], row['image_unique'] )
    mushroom_2 = mushroom_2.reset_index(drop=True)
    # Vérification que le fichier existe avant de déplacer
    if os.path.exists(src_path): 
        
        # si l'image est un doublon, on le transforme aléatoirement
        if row['image_unique'] == row['image']:
            shutil.copy(src_path, dst_path)
            print(f"Image originale sauvegardée : {dst_path}")

        else:

            # Chargement de l'image
            img = image.load_img(src_path)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  

            # Appliquer les transformations
            x = random_translation(img_array)  
            x = random_zoom(x)                   
            x = random_flip(x) 
            x = random_rotation(x)
            x = random_contrast(x)

            # Conversion en image 
            img_tf = image.array_to_img(x[0])

            # Sauvegarde à l'adresse de destination
            img_tf.save(dst_path)
            print(f"Image transformée et sauvegardée : {dst_path}")
    else:
        print(f"Fichier introuvable : {src_path}")
    

    