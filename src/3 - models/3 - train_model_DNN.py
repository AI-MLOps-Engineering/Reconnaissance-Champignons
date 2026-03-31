'''

Modèle DNN

'''

# Import des librairies principales
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

# Imports nécessaires pour créer, sauvegarder et évaluer le modèle
from sklearn.neighbors import KNeighborsClassifier
import joblib

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

# Importation de l'utilitaire image_dataset_from_directory de Keras
from keras.utils import image_dataset_from_directory

# Définition des chemins d'accès
rep_img = 'C:\\Users\\Utilisateur\\Documents\\DataScience_images_Especes\\' # Chemin vers l'ensemble des images de champignons
rep_model = "C:\\Users\\Utilisateur\\Documents\\DataScience\\avr25cds_reconnaissance_champignons\\models\\" # Chemin vers les sauvegardes de modèles

# Importation de l'utilitaire image_dataset_from_directory de Keras
train_ds = image_dataset_from_directory(
    rep_img,
    validation_split=0.2,       # Fraction des données utilisée pour la validation
    subset="training",          # Charger les données d'entraînement
    seed=42,                    # Graine pour le découpage des données
    batch_size=64,  #128        # Taille des lots
    image_size=(224, 224)       # Redimensionnement des images
)

val_ds = image_dataset_from_directory(
    rep_img,
    validation_split=0.2,       # Fraction des données utilisée pour la validation
    subset="validation",              # Charger les données de validation
    seed=42,
    batch_size=64,
    image_size=(224, 224)  
)

# Découpage du dataset temp en validation + test
temp_size = tf.data.experimental.cardinality(temp_ds).numpy()
val_size = temp_size // 2
test_size = temp_size - val_size
val_ds = temp_ds.take(val_size)
test_ds = temp_ds.skip(val_size)

# Optimisation du pipeline
AUTOTUNE = tf.data.AUTOTUNE   # Optimisation automatique du nombre de fichiers et threads à charger
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE) # Pour garder en mémoire les images et gagner en vitesse // shuffle pour mélanger les images à chaque epoch pour un meilleur apprentissage
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)


# Paramétrage du réseau Dense #

# Définition de l'entrée
inputs_dense = Input(shape=(224, 224, 3))

# Couche de rescaling
normalization_layer  = Rescaling(1./255)(inputs_dense)

# Couche de Flatten
x = Flatten()(normalization_layer)

# Première couche dense
x = Dense(
    units=512, 
    activation="relu", 
    kernel_initializer='he_normal' # apprentissage plus stable et plus rapide que "normal"
    )(x)
x = BatchNormalization()(x) # plus robuste

# Deuxième couche dense
x = Dense(
    units=128,              
    activation="relu",
    kernel_initializer='he_normal'
    )(x)
x = BatchNormalization()(x)

# Couche de sortie
outputs_dense = Dense(
    units=30,                # égal nombre de target
    activation="softmax", 
    kernel_initializer='glorot_uniform' # meilleur équilibre des gradients que "normal"
    )(x)

# Création du modèle
model_dense = Model(inputs=inputs_dense, outputs=outputs_dense)

# Compilation du modèle
model_dense.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # sparse_categorical car variables catégorielles

# Callback pour sauvegarder le modèle
save = ModelCheckpoint(
    rep_model+"DNN_model.h5",
    save_best_only=True, 
    monitor='val_accuracy',
    mode='max'
)

# Callback pour réduire le taux d'apprentissage
reduceLR = ReduceLROnPlateau(
                                    monitor="val_loss",
                                    patience=3,
                                    min_delta=0.01,
                                    factor=0.1, 
                                    cooldown=4)

# Callback pour stopper l'apprentissage quand il ne progresse plus
early_stop = EarlyStopping(
                                patience=5,     # Nombre d'époques sans amélioration avant arrêt
                                min_delta=0.01, 
                                mode='min',
                                monitor='val_loss') # On surveille la perte sur l'ensemble de validation

# Entrainement du modèle
history_model = model_dense.fit(train_ds, 
                          epochs=50,                   
                          validation_data=val_ds,
                          callbacks = [reduceLR, save, early_stop])

np.save(rep_model+"DNN_model_history.npy", history_model.history)