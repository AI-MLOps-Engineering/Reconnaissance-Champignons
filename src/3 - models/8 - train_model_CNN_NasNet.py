'''

Modèle CNN avec Transfer Learning NasNet

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
#rom tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.layers import Resizing
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras import regularizers

from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

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


# Base model

NUM_CLASSES = 30
inputs = Input(shape=(224, 224, 3))

# Préprocess NasNet (à la place du Rescaling)
x = preprocess_input(x)

# Backbone NasNet
base_model = NASNetMobile(
    include_top=False,          # on enlève la couche de classification ImageNet
    weights="imagenet",
    input_tensor=x
)

# Tête de classification
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation="relu")(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model_nasnet = Model(inputs, outputs)


# Dégèle d'une partie pour fine tune ensuite
base_model.trainable = True

fine_tune_at = 140  # EfficientNetB0 a environ 175 couches
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# compilation du modèle
model_nasnet.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

# Callback pour sauvegarder le modèle
save = tf.keras.callbacks.ModelCheckpoint(
    rep_model + "CNN_TL_model_nasnet.keras",
    monitor="val_loss",
    mode="min",
    save_best_only=True
)



# Callback pour réduire le taux d'apprentissage
reduceLR = ReduceLROnPlateau(
                                monitor="val_loss",
                                mode="min",
                                patience=3,
                                factor=0.5,
                                min_delta=0.001,
                                cooldown=1,
                                min_lr=1e-6,
                                verbose=1
                                    )

# Callback pour stopper l'apprentissage quand il ne progresse plus
early_stop = EarlyStopping(
                                patience=6,     # Nombre d'époques sans amélioration avant arrêt
                                min_delta=0.001,
                                mode='min',
                                monitor='val_loss',    # On surveille la perte sur l'ensemble de validation
                                restore_best_weights=True)

# Entraînement du modèle
history_model = model_nasnet.fit(
    train_ds,
    validation_data=val_ds,
    epochs=40,
    callbacks=[reduceLR, save, early_stop]
)

# Sauvegarde du modèle
np.save(rep_model+"CNN_TL_model_nasnet_history.npy", history_model.history)
