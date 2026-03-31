'''

Modèle CNN avec Modèle d'ensemble (soft voting sur ResNet50, EfficientNetB0 et NasNet)

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

# Récupération des 3 modèles TL
model_resnet = tf.keras.models.load_model(rep_model + "CNN_TL_model.keras")
model_efficientnetb0 = tf.keras.models.load_model(rep_model + "CNN_TL_efficientb0_model.keras")
model_nasnet = tf.keras.models.load_model(rep_model + "CNN_TL_nasnet_model.keras")

# On gèle pour ne pas les entraîner 
model_resnet.trainable = False
model_efficientnetb0.trainable = False
model_nasnet.trainable = False

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# Define a wrapper for Keras models to be compatible with scikit-learn's VotingClassifier
class SKLearnKerasClassifier(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier" # Add this line to declare it as a classifier

    def __init__(self, model):
        self.model = model
        self.classes_ = None # Will be set during fit

    def fit(self, X, y):
        # For pre-trained Keras models, fit primarily sets the classes_ attribute.
        # The actual model training is assumed to have happened already.
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        # Keras model.predict takes numpy array X as input
        predictions = self.model.predict(X) # Removed verbose=0
        # For hard voting, we need class labels (argmax of probabilities/logits)
        return np.argmax(predictions, axis=-1)

    def predict_proba(self, X):
        # If the Keras model outputs probabilities directly
        return self.model.predict(X) # Removed verbose=0

# Function to extract images and labels from a tf.data.Dataset into NumPy arrays
def extract_data_from_dataset(dataset):
    images_list = []
    labels_list = []
    for images, labels in dataset.unbatch():
        images_list.append(images.numpy())
        labels_list.append(labels.numpy())
    return np.array(images_list), np.array(labels_list)

# Extract training data
X_train, y_train = extract_data_from_dataset(train_ds)

# Wrap the Keras models
wrapped_resnet = SKLearnKerasClassifier(model_resnet)
wrapped_efficientnetb0 = SKLearnKerasClassifier(model_efficientnetb0)
wrapped_nasnet = SKLearnKerasClassifier(model_nasnet)

def get_predictions_and_labels(dataset):
    true_labels = []
    pred_labels = []

    for images, labels in dataset:
        # Get probability predictions from each wrapped model
        preds_resnet = wrapped_resnet.predict_proba(images)
        preds_efficientnetb0 = wrapped_efficientnetb0.predict_proba(images)
        preds_nasnet = wrapped_nasnet.predict_proba(images)

        # Manually perform soft voting by averaging probabilities
        avg_predictions = (preds_resnet + preds_efficientnetb0 + preds_nasnet) / 3

        # Determine the final class label based on the highest average probability
        pred_labels.extend(np.argmax(avg_predictions, axis=-1))

        true_labels.extend(labels.numpy())

    return np.array(true_labels), np.array(pred_labels)

y_true, y_pred = get_predictions_and_labels(test_ds)

# Afficher le rapport de classification
print(classification_report(y_true, y_pred))