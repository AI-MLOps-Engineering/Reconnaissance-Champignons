'''

Modèle CNN

'''

# Import des librairies principales
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

# Importation pour la transformation des images
from tensorflow.keras.layers import RandomTranslation
from tensorflow.keras.layers import RandomZoom
from tensorflow.keras.layers import RandomFlip
from tensorflow.keras.layers import RandomRotation
from tensorflow.keras.layers import RandomContrast

# Imports nécessaires pour construire / sauvegarder / évaluer le modèle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.layers import Resizing
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
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


# Paramétrage du CNN

# Entrée du modèle
inputs = Input(shape=(224, 224, 3))

# Première couche de convolution
conv2d_1 = Conv2D(
    filters=32,
    kernel_size=(3, 3),  #on passe de (4, 4) à (3, 3)
    padding='same',   # same mieux que valid car ne réduit pas l'image
    activation='relu',
)
x = BatchNormalization()(inputs) # stabilise et accélère l’apprentissage
x = conv2d_1(x)   

# Première couche de pooling
max_pooling2d_1 = MaxPooling2D(pool_size=(2, 2),)  
x = max_pooling2d_1(x)

# Deuxième couche de convolution
conv2d_2 = Conv2D(
    filters=64,                    
    kernel_size=(3, 3),            
    padding='same',             
    activation='relu',
)
x = conv2d_2(x)

# Deuxième couche de pooling
max_pooling2d_2 = MaxPooling2D(pool_size=(2, 2),)  
x = max_pooling2d_2(x)

# Troisième couche de convolution
conv2d_3 = Conv2D(
    filters=128,                    
    kernel_size=(3, 3),            
    padding='same',             
    activation='relu',
)
x = conv2d_3(x)

# Troisème couche de pooling
max_pooling2d_3 = MaxPooling2D(pool_size=(2, 2),)  
x = max_pooling2d_3(x)

# Couche de Flatten
flatten = Flatten()
x = flatten(x)

# Première couche dense
dense_1 = Dense(
    units=256,
    activation='relu',
)
x = dense_1(x)

# Couche de sortie
dense_2 = Dense(
    units=30,
    activation='softmax',
)
outputs = dense_2(x)

model_cnn = Model(inputs=inputs, outputs=outputs)

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# Callback pour sauvegarder le modèle
save = ModelCheckpoint(
    rep_model+'CNN_model.h5',
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

# Compilation du modèle
model_cnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrainement du modèle
history_model = model_cnn.fit(train_ds,
                    validation_data=val_ds,
                    epochs=50,
                    callbacks = [reduceLR, save, early_stop])

np.save(rep_model+"CNN_model_history.npy", history_model.history)