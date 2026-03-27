# Fichier pour la définition des fonctions utilisées dans app.py de Streamlit


#####################################################################################################################################
#####################################################################################################################################

# Import des librairies nécessaires :

import pandas as pd
import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import shap
import io
from io import BytesIO
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import image_dataset_from_directory, load_img, img_to_array
from tensorflow.keras.layers import Conv2D
from contextlib import redirect_stdout

#####################################################################################################################################
#####################################################################################################################################

# Fonction pour charger les 2 datasets de l'appli


@st.cache_resource # chargement du dataset principal
def load_df_main_dataset(rep_data, main_dataset):
    df_main_dataset = pd.read_csv(rep_data + main_dataset, sep =  ",")
    return df_main_dataset


@st.cache_resource # chargement du dataset avec seulement les 30 espèces françaises prinicpales
def load_df_dataset_30(rep_data, dataset_30):
    df_dataset_30 = pd.read_csv(rep_data + dataset_30, sep =  ",")
    return df_dataset_30



#####################################################################################################################################
#####################################################################################################################################

# Fonction pour le chargement du Dataset Images

@st.cache_resource
def load_data(rep_img):

    # train_ds et val_ds pour DNN et CNN

    train_ds = image_dataset_from_directory(
        rep_img,
        validation_split=0.2,
        subset="training",
        seed=42,
        batch_size=64,
        image_size=(224, 224)
    )

    val_ds = image_dataset_from_directory(
        rep_img,
        validation_split=0.2,
        subset="validation",
        seed=42,
        batch_size=64,
        image_size=(224, 224)  
    )

    # Nombre de lot dans l'ensemble d'entraînement
    print("Nombre de batch dans train_ds:", train_ds.cardinality().numpy())

    # Nombre de lot dans l'ensemble de validation
    print("Nombre de batch dans val_ds:", val_ds.cardinality().numpy())

    # Class_names
    class_names = train_ds.class_names
    print(class_names)

    # train_img_flat, train_label, val_img_flat, val_label pour RF et KNN

    # Normalisation
    train_ds_norm = train_ds.map(lambda x, y: (x / 255.0, y))
    val_ds_norm = val_ds.map(lambda x, y: (x / 255.0, y))

    # Fonction pour extraire les images et labels
    def dataset_to_numpy(dataset):
        images, labels = [], []
        for batch_images, batch_labels in dataset:
            images.append(batch_images.numpy())
            labels.append(batch_labels.numpy())
        return np.concatenate(images), np.concatenate(labels)

    X_train, y_train = dataset_to_numpy(train_ds_norm)
    X_val, y_val = dataset_to_numpy(val_ds_norm)

    # Aplatissement des images pour les modèles scikit-learn
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)

    print("Shape de X_train_flat :", X_train_flat.shape)

    return train_ds, val_ds, X_train_flat, y_train, X_val_flat, y_val, class_names 


#####################################################################################################################################
#####################################################################################################################################

# Fonction pour le chargement Modèle :

def loading_model(rep_model, model_file):
    model_path = rep_model + model_file
    if model_path.endswith(".pkl"):
        return joblib.load(model_path), "sklearn"
    elif model_path.endswith(".h5"):
        return load_model(model_path), "keras"
    elif model_path.endswith(".keras"):
        return tf.keras.models.load_model(model_path, compile=False), "keras"
    else:
        raise ValueError("Format de modèle non reconnu")


#####################################################################################################################################
#####################################################################################################################################



# Fonction pour le chargement de l’historique (loss/accuracy)

def load_training_history(rep_model, model_name):
    history_path = rep_model + str(model_name.split('.')[0]) + "_history.npy"
    try:
        history = np.load(history_path, allow_pickle=True).item()
        st.sidebar.markdown('🍄 :green[Historique d’entraînement chargé]')
        return history
    except:
        st.sidebar.markdown('🍄 :green[Aucun historique d’entraînement trouvé pour ce modèle]')
        return None



#####################################################################################################################################
#####################################################################################################################################


# Fonction pour les prédictions DNN et CNN

def get_predictions_and_labels(dataset, model):
    true_labels = []
    
    images_list = [] # pour les analyser et afficher plus tard car pas le même ordre que dans val_ds unbatch

    pred_labels_1, pred_labels_2, pred_labels_3 = [], [], []
    pred_scores_1, pred_scores_2, pred_scores_3 = [], [], []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)  # shape (batch_size, num_classes)

        # indices triés du plus probable au moins probable
        top3_idx = np.argsort(preds, axis=-1)[:, -3:][:, ::-1]
        top3_scores = np.take_along_axis(preds, top3_idx, axis=-1) # pour récupérer les scores correspondants aux index classés

        pred_labels_1.extend(top3_idx[:, 0])
        pred_labels_2.extend(top3_idx[:, 1])
        pred_labels_3.extend(top3_idx[:, 2])

        pred_scores_1.extend(top3_scores[:, 0])
        pred_scores_2.extend(top3_scores[:, 1])
        pred_scores_3.extend(top3_scores[:, 2])

        true_labels.extend(labels.numpy())

        images_list.extend(images.numpy())

    return (
        np.array(true_labels),
        np.array(pred_labels_1), np.array(pred_scores_1),
        np.array(pred_labels_2), np.array(pred_scores_2),
        np.array(pred_labels_3), np.array(pred_scores_3),
        np.array(images_list),
    )

#####################################################################################################################################
#####################################################################################################################################

# Fonction permettant d'afficher 4 images par espèces

def show_n_images(dataset, class_names, specie_choice): # dataset = train_ds


    true_labels = []   
    images_list = [] # pour les analyser et afficher plus tard car pas le même ordre que dans val_ds unbatch

    for images, labels in dataset:
        true_labels.extend(labels.numpy())
        images_list.extend(images.numpy())        


    indice = class_names.index(specie_choice)


    liste_idx = []                                               # On initialise la liste pour les futurs index des photos
    for j in range(len(true_labels)) :                                # On regarde toutes les labellisations 
        if int(true_labels[j]) == indice and len(liste_idx) == 0 :   # Si on est sur une image qui correspond à l'espèce recherchée et qu'on est sur le n° d'image souhaitée
            liste_idx.append(j)                                  # On ajoute l'index à la liste pour afficher l'image plus tard
        elif int(true_labels[j]) == indice and len(liste_idx) == 1 :
            liste_idx.append(j) 
        elif int(true_labels[j]) == indice and len(liste_idx) == 2 :
            liste_idx.append(j)
        elif int(true_labels[j]) == indice and len(liste_idx) == 3 :
            liste_idx.append(j)
        elif int(true_labels[j]) == indice and len(liste_idx) == 4 :
            liste_idx.append(j)    
        j += 1

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(
                images_list[liste_idx[0]].astype("uint8"),
                width=200
                )
    with col2:           
        st.image(
                images_list[liste_idx[1]].astype("uint8"),
                width=200
                )
    with col3:            
        st.image(
                images_list[liste_idx[2]].astype("uint8"),
                width=200
                )  
    with col4:                         
        st.image(
                images_list[liste_idx[3]].astype("uint8"),
                width=200
                )           
    with col5:                         
        st.image(
                images_list[liste_idx[4]].astype("uint8"),
                width=200
                )  



#####################################################################################################################################
#####################################################################################################################################


# Fonction permettant d'afficher le top 3 des prédictions de l'image (Pour analyse d'image unitaire)

def affichage_top3(model, dataset, class_names, image_test):
    (
        _,   # pas besoin de true_label, car c'est une photo test, le vrai label est dans le titre
        pred_labels_1, pred_scores_1,
        pred_labels_2, pred_scores_2,
        pred_labels_3, pred_scores_3,
        images_list
    ) = get_predictions_and_labels(dataset, model) # Récupère la fonction qui permet d'avoir le top-3, les scores et l'image


    st.markdown(
            f"""
            <div style="
            max-width: 600px;
            background-color: black;
            text-align: left;
            ">
                <p style="
                font-size:8px;
                color = #004B23;
                ">
                    => Photo : {image_test}\n
                    Prédiction 1 : {class_names[pred_labels_1[0]]}, Score : {round(float(pred_scores_1[0]), 2)}\n
                    Prédiction 2 : {class_names[pred_labels_2[0]]}, Score : {round(float(pred_scores_2[0]), 2)}\n
                    Prédiction 3 : {class_names[pred_labels_3[0]]}, Score : {round(float(pred_scores_3[0]), 2)}\n
            """,
            unsafe_allow_html=True
        )

    st.image(
            images_list[0].astype("uint8"),
            width=200
        )

    st.markdown("---")


#####################################################################################################################################
#####################################################################################################################################


# Affichage de la matrice de confusion

def conf_mat_affichage(y_true, y_pred, class_names) :
    cm = confusion_matrix(y_true, y_pred, normalize='true') 
    #st.subheader("Matrice de confusion du modèle :")
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(cm, cmap='Blues', annot=True, cbar=False, fmt=".2f")
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names, rotation=0)
    plt.ylabel('Vrais labels')
    plt.xlabel('Labels prédits')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)


#####################################################################################################################################
#####################################################################################################################################

# fonction pour récupérer le summary et l'afficher dans streamlit

def get_model_summary(model):
    stream = io.StringIO()              # Crée un tampon mémoire pour stocker du texte
    with redirect_stdout(stream):       # Capture les affichages print() dans ce tampon
        model.summary()                 # Écrit le résumé dans stream au lieu de la console
    summary_str = stream.getvalue()     # Récupère le texte capturé sous forme de chaîne
    return summary_str



#####################################################################################################################################
#####################################################################################################################################

# Affichage des courbes d'entrainement

def training_curves(history_model) :
    #st.subheader("Courbes d'entraînement :")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].plot(history_model['loss'])     # pas besoin de .history car history_model obtenu à partir de load_training_history() est un dico des .item()
    ax[0].plot(history_model['val_loss'])
    ax[0].set_title('Model loss by epoch')
    ax[0].set_ylabel('loss')
    ax[0].set_xlabel('epoch')
    ax[0].legend(['train', 'val'], loc='best')

    ax[1].plot(history_model['accuracy'])
    ax[1].plot(history_model['val_accuracy'])
    ax[1].set_title('Model accuracy by epoch')
    ax[1].set_ylabel('accuracy')
    ax[1].set_xlabel('epoch')
    ax[1].legend(['train', 'val'], loc='best')
    
    st.pyplot(fig)



#####################################################################################################################################
#####################################################################################################################################

# Fonction pour afficher n images labelisées dans la classe choisie :

def images_show(images_list, y_pred_1, y_pred_2, y_pred_3, y_score_1, y_score_2, y_score_3, y_true, n_img_show, specie_choice, class_names, verdict):

    # Indices des images sélectionnées
    if verdict == "Exactes" : # Si lesprédictions sont exactes
        selection = np.where((y_pred_1 == y_true) & (y_true == class_names.index(specie_choice)))[0]
    else : # Si les prédictions sont fausses
        selection = np.where((y_pred_1 != y_true) & (y_true == class_names.index(specie_choice)))[0]


    liste_index = [] 

    for _, idx in enumerate(selection[:n_img_show]):

        liste_index.append(int(idx))
        
        st.markdown(
            f"""
            <div style="
            max-width: 600px;
            background-color: black;
            text-align: left;
            ">
                <p style="
                font-size:8px;
                color = #004B23;
                ">
                    => Vrai label : {class_names[y_true[idx]]}\n
                    Prédiction 1 : {class_names[y_pred_1[idx]]}, Score : {round(float(y_score_1[idx]), 2)}\n
                    Prédiction 2 : {class_names[y_pred_2[idx]]}, Score : {round(float(y_score_2[idx]), 2)}\n
                    Prédiction 3 : {class_names[y_pred_3[idx]]}, Score : {round(float(y_score_3[idx]), 2)}\n
            """,
            unsafe_allow_html=True
        )

        st.image(
            images_list[idx].astype("uint8"),
            width=200
        )

        st.markdown("---")

    return liste_index # retourne la sélection des images souhaitées



#####################################################################################################################################
#####################################################################################################################################

# Fonction pour créer le dico Latin - Français

@st.cache_resource
def dico_lat_fra(rep_data, taxons):

    data = pd.read_csv(rep_data + taxons, sep = ';', encoding='latin-1')

    dico_1 = {}
    dico_2 = {}

    for i in range(len(data)):
        dico_1[str(data.iloc[i, 0]) + " // " + str(data.iloc[i, 1])] = str(data.iloc[i, 0])
        
        dico_2[str(data.iloc[i, 0])] = str(data.iloc[i, 0]) + " // " + str(data.iloc[i, 1])

    return dico_1, dico_2


#####################################################################################################################################
#####################################################################################################################################


# Fonction Grad_Cam

def grad_cam(image, model, layer_name):     # Analyse une seule image à chaque fois
    # Récupérer la couche convolutive
    layer = model.get_layer(layer_name)
    
    # Créer un modèle qui génère les sorties de la couche convolutive et les prédictions
    grad_model = Model(inputs=model.input, outputs=[layer.output, model.output])

    # Ajout d'une dimension de batch
    image = tf.expand_dims(image, axis=0)

    # Calcul des gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        predicted_class = tf.argmax(predictions[0])  # Classe prédite
        loss = predictions[:, predicted_class]  # Perte pour la classe prédite

    # Gradients des scores par rapport aux sorties de la couche convolutive
    grads = tape.gradient(loss, conv_outputs)

    # Moyenne pondérée des gradients pour chaque canal
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Pondération des activations par les gradients calculés
    conv_outputs = conv_outputs[0]  # Supprimer la dimension batch
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalisation de la carte de chaleur
    heatmap = tf.maximum(heatmap, 0)  # Se concentrer uniquement sur les valeurs positives
    heatmap /= tf.math.reduce_max(heatmap)  # Normaliser entre 0 et 1
    heatmap = heatmap.numpy()  # Convertir en tableau numpy pour la visualisation

   # Redimensionner la carte de chaleur pour correspondre à la taille de l'image d'origine
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (image.shape[1], image.shape[2])).numpy()
    heatmap_resized = np.squeeze(heatmap_resized, axis=-1) # supprimer la dimension de taille 1 à la fin du tableau heatmap_resized

    # Colorier la carte de chaleur avec une palette (par exemple, "jet")
    heatmap_colored = plt.cm.jet(heatmap_resized)[..., :3] # Récupérer les canaux R, G, B 

    superimposed_image = heatmap_colored * 0.7 + image[0].numpy() / 255.0

    return np.clip(superimposed_image, 0, 1), predicted_class


#####################################################################################################################################
#####################################################################################################################################


# Fonction Affichage Grad_Cam

# Prend en entrée les images à analyser et le modèle

def show_grad_cam(images_analyse, model):


    
    number_of_images = images_analyse.shape[0]

    conv_layers = [layer.name for layer in model.layers if isinstance(layer, Conv2D)]
    
    for _, layer in enumerate(conv_layers):

        st.markdown(
                f"""
                <p style='color:#004B23; font-size:16px; font-weight:600;'>
                    Grad-CAM {layer}
                </p>
                """,
                unsafe_allow_html=True
            )


        cols = st.columns(number_of_images) # divise la ligne en colonnes pour affichage des images

        for n, col in enumerate(cols): # correspond au nb d'images par layer

            # Obtenir l'image avec la carte de chaleur superposée
            grad_cam_image, predicted_class = grad_cam(images_analyse[n], model, layer)
            
            # Afficher l'image avec Grad-CAM           
            with col:
                st.image(grad_cam_image, width=200)

        
    st.markdown("---")



#####################################################################################################################################
#####################################################################################################################################

# Fonction SHAP

# Prend en entrée les images à analyser, le modèle et les class_names

def shap_function(images_analyse, model, class_names) :

    # SHAP attend un tableau numpy en entrée, donc permet de convertir le tenseur tensorflow si c'en est un
    # (et les images test sont des np.ndarray)
    images_analyse = images_analyse.numpy() if hasattr(images_analyse, 'numpy') else np.array(images_analyse)

    # prédictions pour obtenir le top k par image
    preds = model.predict(images_analyse, verbose=0)  # (n, C)
    k = 5
    topk_idx = np.argsort(preds, axis=1)[:, ::-1][:, :k]  # (n, k)

    # labels (n, k) = noms des classes correspondants au top-k
    labels = np.take(np.array(class_names, dtype=object), topk_idx)

    # SHAP explainer
    masker = shap.maskers.Image("inpaint_telea", images_analyse[0].shape)

    def f(x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        return model(x, training=False)

    explainer = shap.Explainer(f, masker, output_names=class_names)


    # Calculer les valeurs SHAP pour les images qu'on veut expliquer 
    shap_values = explainer(images_analyse, max_evals=500, outputs=shap.Explanation.argsort.flip[:k])

    # Créer la figure SHAP
    plt.figure()
    shap.image_plot(shap_values, labels=labels)

    # Convertir la figure en image pour st.image()
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0) # Pour retourner au début du fichier en mémoire avant de le lire
    plt.close()  # ferme la figure matplotlib pour éviter l'affichage multiple

    # Afficher avec Streamlit
    st.image(buf, caption="Visualisation SHAP", use_container_width=True)


#####################################################################################################################################
#####################################################################################################################################

# Fonction Affichage Cartes des features

# Prend en entrée les images à analyser, le modèle

def show_feature_maps(images_analyse, model):
    # Récupérer les noms des couches de convolution
    conv_layers = [layer.name for layer in model.layers if isinstance(layer, Conv2D)]

    # Si plusieurs images, on ne garde que la première
    if len(images_analyse.shape) == 4:
        images_analyse = images_analyse[0]

    for j, layer in enumerate(conv_layers):
        # Créer un modèle intermédiaire
        conv_model = Model(inputs=model.input, outputs=model.get_layer(layer).output)

        # Gérer la forme d'entrée
        if len(images_analyse.shape) == 3:
            image_batch = tf.expand_dims(images_analyse, axis=0)
        elif len(images_analyse.shape) == 4:
            image_batch = images_analyse
        else:
            raise ValueError(f"Forme inattendue pour images_analyse : {images_analyse.shape}")

        # Prédire les feature maps
        feature_maps = conv_model.predict(image_batch, verbose=0)
        feature_maps = np.squeeze(feature_maps)

        # Corriger l’ordre des dimensions si besoin
        if feature_maps.ndim == 3 and feature_maps.shape[0] != feature_maps.shape[1]:
            # Cas (N, H, W)
            feature_maps = np.transpose(feature_maps, (1, 2, 0))

        if feature_maps.ndim != 3:
            raise ValueError(f"Forme inattendue pour feature_maps ({layer}) : {feature_maps.shape}")

        num_filters = feature_maps.shape[-1]
        grid_size = int(np.ceil(np.sqrt(num_filters)))

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten()

        for i in range(num_filters):
            ax = axes[i]
            ax.imshow(feature_maps[..., i], cmap='viridis')
            ax.axis("off")
            ax.set_title(f'Filtre {i+1}', fontsize=8)

        # Masquer les subplots vides
        for k in range(num_filters, len(axes)):
            axes[k].axis("off")

        plt.suptitle(f'Feature maps – Couche : {layer}', fontsize=14)
        plt.tight_layout()

        # Convertir en image pour Streamlit
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        plt.close(fig)

        # Afficher dans Streamlit
        st.image(buf, caption=f"Feature maps de la couche '{layer}'", use_container_width=True)



#####################################################################################################################################
#####################################################################################################################################

# Fonction Affichage Barplot


def barplot_display(count_input, titre, labelx, labely):

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=count_input.values, y=count_input.index, palette='viridis', ax=ax)

    ax.set_title(titre)
    ax.set_xlabel(labelx)
    ax.set_ylabel(labely)

    for i in range(len(ax.containers)) :
        ax.bar_label(ax.containers[i])

    # Convertir en image pour Streamlit
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close(fig)

    # Afficher dans Streamlit
    st.image(buf, width=700)



#####################################################################################################################################
#####################################################################################################################################

# Fonction Affichage Countplot

def countplot_display(count_input, titre, labelx, labely):

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x = count_input, ax=ax)

    ax.set_title(titre)
    ax.set_xlabel(labelx)
    ax.set_ylabel(labely)


    # Convertir en image pour Streamlit
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close(fig)

    # Afficher dans Streamlit
    st.image(buf, width=700)

