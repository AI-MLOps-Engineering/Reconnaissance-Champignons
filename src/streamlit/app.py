import pandas as pd
import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory, load_img, img_to_array

import base64

from functions import load_df_main_dataset, load_df_dataset_30, load_data, loading_model, load_training_history, get_predictions_and_labels, conf_mat_affichage, training_curves, affichage_top3, show_grad_cam, shap_function, show_feature_maps, get_model_summary, dico_lat_fra, images_show, barplot_display, countplot_display, show_n_images
from settings import rep_data, rep_img, rep_img_2, rep_img_3, rep_model, models, main_dataset, dataset_30, taxons, n_img_show, dico_taxons


######################################################################################################################

# Config Streamlit

st.set_page_config(page_title="ChamPy Classifier", layout="wide")

def get_base64_of_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Image de fond
image_base64 = get_base64_of_image(rep_img_2 + "Fond.png")

st.markdown(f"""
    <style>
        .stApp {{
            background-image: linear-gradient(rgba(255,255,255,0.7), rgba(255,255,255,0.7)),
                              url("data:image/jpg;base64,{image_base64}");
            background-size: 100%;
            background-position: center;
            background-repeat: no-repeat;
        }}
    </style>
""", unsafe_allow_html=True)


######################################################################################################################

# Chargement des datasets 


df_main_dataset = load_df_main_dataset(rep_data, main_dataset) # chargement du dataset principal
df_dataset_30 = load_df_dataset_30(rep_data, dataset_30) # chargement du dataset avec seulement les 30 espèces françaises prinicpales



######################################################################################################################


# Chargement des images
train_ds, val_ds, X_train_flat, y_train, X_val_flat, y_val, class_names = load_data(rep_img)


######################################################################################################################

# Création du dictionnaire Latin - Français pour le nom des champignons

lat_fra, fra_lat = dico_lat_fra(rep_data, taxons)

######################################################################################################################


# Side barre

# Choix du modèle
model_choice = st.sidebar.selectbox(
    "Choix du modèle :", list(models.keys())
)


model_file = models[model_choice]

model_loaded, model_type = loading_model(rep_model, model_file)


st.sidebar.markdown(f'🍄 :green[Modèle {model_choice} chargé]')

history_model_loaded = load_training_history(rep_model, model_file)

st.sidebar.markdown("---")

# Menu
menu = st.sidebar.radio("Navigation", ["Accueil", "Introduction / Contexte", "Objectifs", "Exploration des données", "Etapes de réalisation", "Modèles", "Interprétabilité",
                                       "Test sur une image", "Conclusion", "Remerciements"])

st.sidebar.markdown("---")



######################################################################################################################
######################################################################################################################

# Page d'accueil

if menu == "Accueil":

    image_datascientest = get_base64_of_image(rep_img_2 + "DataScientest.png")

    st.markdown(
            f"""
            <div style="text-align: right;">
                <img src="data:image/png;base64,{image_datascientest}" 
                    style="width:15%; height:auto; border-radius:10px;">
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("---")

    st.markdown(
                    f"""
                    <p style='color:#004B23; font-size:30px; font-weight:1000; text-align: center'>
                        Projet Reconnaissance de champignons <br> - ChamPy Classifier -
                    </p>                    
                    """,
                    unsafe_allow_html=True
                )
    st.markdown("---")


    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:25px; font-weight:1000; text-align: left'>
                        Cursus : Data Scientist <br>
                        <br>			
                        Etudiants : <br>
                        &emsp;FOCRAUD Loïc (MLE Jan-25), <br>
                		&emsp;RENVOISÉ Quentin (DS Avr-25)
                    </p>                    
                    """,
                    unsafe_allow_html=True
                )
    st.markdown("---")
    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:25px; font-weight:1000; text-align: right'>
                        <br>
                        <br>			
                        <br>
                        fev-2026&emsp;<br>
                    </p>                    
                    """,
                    unsafe_allow_html=True
                )
    st.markdown("---")



######################################################################################################################
######################################################################################################################

# Page Introduction / Contexte

if menu == "Introduction / Contexte":

# Titre
    st.markdown(
                    f"""
                    <p style='color:#004B23; font-size:30px; font-weight:1000; text-align: center'>
                        - Introduction / Contexte-
                    </p>                    
                    """,
                    unsafe_allow_html=True
                )
    st.markdown("---")


# Intro
    st.markdown(
                    f"""
                    <p style='color:#400080; font-size:20px; font-weight:800; text-align: left'></b>
                        Introduction :
                    <b></p>                    
                    """,
                    unsafe_allow_html=True
                ) 
    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        L’objectif du présent projet est de faire de la reconnaissance de champignons à l’aide d'algorithmes de computer vision (deep-learning). Les champignons présentant un grand nombre d’espèces et de genre, ainsi une grande variété de caractéristiques biologiques et nutritives, il est intéressant de savoir les <u>classifier</u> précisément à l’aide de la computer vision.<br>
                    </p>                    
                    """,
                    unsafe_allow_html=True
                )
    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        A moyen terme ces travaux pourront aboutir à une brique constitutive d’une application visant à automatiser l'identification des espèces fongiques.
                        Laquelle trouvera sa place dans plusieurs secteurs tels que le contrôle qualité en industrie agroalimentaire, l’assistance aux mycologues professionnels, le développement d'applications mobiles pour les cueilleurs amateurs, …<br><br>
                    </p>                    
                    """,
                    unsafe_allow_html=True
                )

 # Du point de vue technique :   
    st.markdown(
                    f"""
                    <p style='color:#400080; font-size:20px; font-weight:800; text-align: left'></b>
                        Du point de vue technique :
                    <b></p>                    
                    """,
                    unsafe_allow_html=True
                )  
    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        Le projet exploite les algorithmes de Machine Learning et Deep Learning, notamment les réseaux de neurones denses (DNN) et réseaux de neurones convolutifs (CNN).<br>
                        L'utilisation de modèles pré-entraînés tels que ResNet pourront être utilisés afin de capitaliser sur des représentations visuelles déjà apprises, tout en les adaptant à la spécificité des caractéristiques morphologiques des champignons (texture du chapeau, forme du pied, présence de lamelles ou de pores).<br><br>
                    </p>                    
                    """,
                    unsafe_allow_html=True
                )
# Du point de vue économique :   
    st.markdown(
                    f"""
                    <p style='color:#400080; font-size:20px; font-weight:800; text-align: left'></b>
                        Du point de vue économique :
                    <b></p>                    
                    """,
                    unsafe_allow_html=True
                )  
    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        Le marché des champignons représentait plus de 70 milliards de dollars en 2024 (source : <a href="https://www.imarcgroup.com/mushroom-market">https://www.imarcgroup.com</a>).<br>
                        L'automatisation de l'identification pourrait réduire significativement les coûts de contrôle qualité.<br>
                        Elle pourrait également permettre de diminuer les risques d'intoxication liés aux erreurs d'identification, représentant un enjeu de santé publique majeur => Plus de 1 000 empoisonnements aux champignons sont enregistrés chaque année par les centres antipoison français (source : <a href="https://www.foodsafetynews.com/2019/12/thousands-poisoned-by-mushrooms-in-france-in-recent-years/">https://www.foodsafetynews.com</a>).<br><br>
                    </p>                    
                    """,
                    unsafe_allow_html=True
                )

# Du point de vue éthique et sécuritaire :   
    st.markdown(
                    f"""
                    <p style='color:#400080; font-size:20px; font-weight:800; text-align: left'></b>
                        Du point de vue éthique et sécuritaire :
                    <b></p>                    
                    """,
                    unsafe_allow_html=True
                ) 

    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        Le respect des critères éthiques et de sécurité est un enjeu essentiel du projet. Il sera très important d’émettre à l’issue de la classification des réserves ou avertissements forts sur les natures comestibles ou toxiques de classes identifiées.<br>
                        Une classification imprécise peu constituer des risques importants pour la santé (source : <a href="https://www.courrierinternational.com/article/vu-du-royaume-uni-les-applications-pour-cueilleurs-de-champignons-fleurissent-en-france-les-intoxications-aussi_225538#:~:text=Des%20risques%20%C3%A9lev%C3%A9s%20d'erreur,m%C3%AAme%20p%C3%A9riode%20l'ann%C3%A9e%20pr%C3%A9c%C3%A9dente.">https://www.courrierinternational.com</a>).<br><br>
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )


######################################################################################################################
######################################################################################################################

# Page Objectifs

if menu == "Objectifs":

# Titre
    st.markdown(
                    f"""
                    <p style='color:#004B23; font-size:30px; font-weight:1000; text-align: center'>
                        - Objectifs -
                    </p>                    
                    """,
                    unsafe_allow_html=True
                )
    st.markdown("---")

# Principaux objectifs
    st.markdown(
                    f"""
                    <p style='color:#400080; font-size:20px; font-weight:800; text-align: left'></b>
                        Principaux objectifs :
                    <b></p>                    
                    """,
                    unsafe_allow_html=True
                )

    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        &emsp;•	Développer un modèle de classification capable d'identifier les variables des différents taxons (famille, genre, espèce, ...) de champignons avec une précision suffisante.<br>
                        &emsp;•	Identifier le meilleur taxon afin d'obtenir un compromis satisfaisant entre nombre de suffisant de photos, nombre de classes, performance du modèle et pertinence d'utilisation.<br>
                        &emsp;•	Dans notre cas, après itérations, nous avons abouti à une cible de 30 espèces les plus répandues en France.<br>
                        &emsp;•	Optimiser le modèle pour un déploiement mobile (contraintes de taille et de vitesse d'inférence)<br><br>
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )

# Niveau d'expertise de l'équipe
    st.markdown(
                    f"""
                    <p style='color:#400080; font-size:20px; font-weight:800; text-align: left'></b>
                        Niveau d'expertise de l'équipe : 
                    <b></p>                    
                    """,
                    unsafe_allow_html=True
                )
    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        L'expertise de l’équipe d’étudiants couvre les aspects techniques du deep learning. Un appui pour l’expertise en biologie (connaissance de la taxonomie) est nécessaire.<br><br>
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )

# Interactions avec des experts métiers : 
    st.markdown(
                    f"""
                    <p style='color:#400080; font-size:20px; font-weight:800; text-align: left'></b>
                        Interactions avec des experts métiers : 
                    <b></p>                    
                    """,
                    unsafe_allow_html=True
                )
    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        Collaboration avec Mme Laetitia ROSTAING (professeure en biologie) qui a aidé à comprendre quels taxons sont identifiables par reconnaissance visuels et ceux qui ne le sont pas, permettant ainsi d’orienter la stratégie d’augmentation de données.<br><br>
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )


# Projets similaires
    st.markdown(
                    f"""
                    <p style='color:#400080; font-size:20px; font-weight:800; text-align: left'></b>
                        Projets similaires : 
                    <b></p>                    
                    """,
                    unsafe_allow_html=True
                )
    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        &emsp;•	Application « Champignouf »  : <a href="https://apps.apple.com/fr/app/champignouf/id1227854971">https://apps.apple.com/fr/app/champignouf</a><br>
                        &emsp;•	Application « Champignon Pro » : <a href="https://apps.apple.com/fr/app/champignons-pro/id523607704">https://apps.apple.com/fr/app/champignons-pro</a><br>
                        &emsp;•	Application « PlantNet » (identification botanique) : <a href="https://plantnet.org/">https://plantnet.org</a><br>
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )



######################################################################################################################
######################################################################################################################

# Page Exploration des données


if menu == "Exploration des données":


    # Titre
    st.markdown(
                    f"""
                    <p style='color:#004B23; font-size:30px; font-weight:1000; text-align: center'>
                        - Exploration des données -
                    </p>                    
                    """,
                    unsafe_allow_html=True
                )
    st.markdown("---")

# Jeux de données utilisés
    st.markdown(
                    f"""
                    <p style='color:#400080; font-size:20px; font-weight:800; text-align: left'></b>
                        Jeux de données utilisés : 
                    <b></p>                    
                    """,
                    unsafe_allow_html=True
                )
    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        <b>• Dataset principal :</b> « observations_mushroom.csv » issu du site <a href="https://mushroomobserver.org/">https://mushroomobserver.org/</a>. Site très complet sur les champignons avec informations, Texte descriptif du jeu de données, avec des informations sur la volumétrie, le contenu et la qualité + lien vers où récupérer les données.<br><br>
                        <b>• Disponibilité des données :</b> Les données issues de Mushroom Observer sont accessibles librement. A noter cependant que les images ne sont plus téléchargeables directement sur le site depuis peu (le téléchargement unitaire et le scrapping restent possibles). Celles-ci ont pu être récupérées directement auprès de Datascientest qui les avait préalablement téléchargées (<a href="https://assets-datascientest.s3.eu-west-1.amazonaws.com/datasets/mushroom_dataset.zip">https://assets-datascientest.s3.eu-west-1.amazonaws.com/datasets/mushroom_dataset.zip</a><br><br>
                        <b>• Volumétrie :</b> Au total 646 524 images basse résolution (320 x 240 pixels en moyenne), représentant plusieurs milliers d’espèces avec une distribution très déséquilibrée (voir partie B.2/). Taille totale du dataset : 15 GB environ.<br><br>
                        <b>• Datasets complémentaires :</b><br>
                            &emsp;&emsp;-  Top 30 des espèces de champignons les plus répandues en France (comestibles et toxiques)<br>
                            &emsp;&emsp;- Enrichissement à l’aide des sources globales suivantes :<br>
                            &emsp;&emsp;&emsp;-	Wikipedia<br>
                            &emsp;&emsp;&emsp;-	Google image<br><br>
                      </p>                    
                    """,
                    unsafe_allow_html=True
                )

# Jeux de données utilisés
    st.markdown(
                    f"""
                    <p style='color:#400080; font-size:20px; font-weight:800; text-align: left'></b>
                        Pertinence des variables : 
                    <b></p>                    
                    """,
                    unsafe_allow_html=True
                )
    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        <b>Variables les plus pertinentes :</b><br>
                            &emsp;<b>• Données images (Caractéristiques visuelles et morphologique) : </b>couleur dominante, texture, présence et taille du chapeau, courbure du pied, …<br>
                            &emsp;<b>• Indice de confiance de la classification humaine (feature « confidence »)</b><br><br>
                        <b>Variable cible : </b>Classification multi-classe avec hiérarchie taxonomique intégrée (règne → phylum → classe → ordre → famille → genre → espèce).<br><br>    
                        <b>Particularités du jeu de données :</b><br><br>
                            &emsp;<b><u>• Complexité de la taxonomie des champignons : </u></b><br>
                            &emsp;Les champignons peuvent être classifiés selon les 7 taxons suivants, que l'on retrouve également dans notre dataset « observations_mushroom.csv » :<br>
                            &emsp;&emsp;- Le règne (ou kingdom) => variable 'gbif_info/kingdom'<br>
                            &emsp;&emsp;- Le phylum => variable 'gbif_info/phylum'<br>
                            &emsp;&emsp;- La classe => variable 'gbif_info/class'<br>
                            &emsp;&emsp;- L'ordre => variable 'gbif_info/order'<br>
                            &emsp;&emsp;- La Famille => variable 'gbif_info/family'<br>
                            &emsp;&emsp;- Le Genre (ou genus) => variable 'gbif_info/genus',<br>
                            &emsp;&emsp;- L'Espèce => variable 'gbif_info/species'<br>
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(rep_img_2+'taxons.png')
    with col2:
        st.image(rep_img_2+'taxons_2.png', width=500)


    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                         &emsp;Sources :<br>
                            &emsp;- <a href="https://fr.wikipedia.org/wiki/R%C3%A8gne_(biologie)">https://fr.wikipedia.org/wiki/R%C3%A8gne_(biologie)</a><br>
                            &emsp;- <a href="https://cultiver-les-champignons.com/classification-champignons/">https://cultiver-les-champignons.com/classification-champignons/</a><br><br>
                        Les taxons « règnes », « phylum » et « classe » ne peuvent pas être identifiés par une simple reconnaissance visuelle (nécessite analyse moléculaire, microscopique ou chimique).<br>
                        Les taxons « ordre » et « famille » sont difficilement identifiables par une simple reconnaissance visuelle (peu de critères visuels robustes).<br>
                        Les taxons « genre » et « espèce » sont identifiables par reconnaissance visuelle (nécessite important jeu de données).<br><br>    
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )


    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        <b><u>• Dispersion des données : </u></b>quel que soit le taxon dans notre jeu de données, les variables sont extrêmement dispersées.<br><br>
                            &emsp;Count value des taxons :<br>
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )

    cols = ['gbif_info/kingdom', 'gbif_info/phylum', 'gbif_info/class', 'gbif_info/order', 'gbif_info/family', 'gbif_info/genus','gbif_info/species']
    report_main_dataset = df_main_dataset[cols].describe(include='object')
    st.dataframe(report_main_dataset)

    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        &emsp;Analyse de la répartition des taxons :<br>
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )


    # Dataframe pour analyse règne (kingdom)
    analyse_regne = df_main_dataset.groupby('gbif_info/kingdom')['gbif_info/kingdom'].count()
    analyse_regne = pd.DataFrame(analyse_regne.describe())

    # Dataframe pour analyse phylum (phylum)
    analyse_phylum = df_main_dataset.groupby('gbif_info/phylum')['gbif_info/phylum'].count()
    analyse_phylum = pd.DataFrame(analyse_phylum.describe())

    # Dataframe pour analyse classe (class)
    analyse_classe = df_main_dataset.groupby('gbif_info/class')['gbif_info/class'].count()
    analyse_classe = pd.DataFrame(analyse_classe.describe())

    # Dataframe pour analyse ordre (order))
    analyse_ordre = df_main_dataset.groupby('gbif_info/order')['gbif_info/order'].count()
    analyse_ordre = pd.DataFrame(analyse_ordre.describe())

    # Dataframe pour analyse famille (family)
    analyse_famille = df_main_dataset.groupby('gbif_info/family')['gbif_info/family'].count()
    analyse_famille = pd.DataFrame(analyse_famille.describe())

    # Dataframe pour analyse genre (genus)
    analyse_genre = df_main_dataset.groupby('gbif_info/genus')['gbif_info/genus'].count()
    analyse_genre = pd.DataFrame(analyse_genre.describe())

    # Dataframe pour analyse species (espèces)
    analyse_espece = df_main_dataset.groupby('gbif_info/species')['gbif_info/species'].count()
    analyse_espece = pd.DataFrame(analyse_espece.describe())


    # Concaténation
    analyse = pd.concat([analyse_regne, analyse_phylum, analyse_classe, analyse_ordre, analyse_famille, analyse_genre, analyse_espece], axis = 1)

    analyse = analyse.rename(columns = dico_taxons)

    st.dataframe(analyse)


    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        <br>
                        <b><u>• Forte variabilité intra-espèce : </u></b>un même champignon peut présenter des aspects très différents selon son stade de maturité<br><br>                 
                    Espèce Amanita Muscaria juvénile Vs mature (images 14107 et 479291 du dataset) :<br>
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )


    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(rep_img_3+'14107.jpg')
    with col2:
        st.image(rep_img_3+'479291.jpg')


    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        <br>
                        <b><u>• Qualité hétérogène et parasitage : </u></b>images d’amateurs vs. photos scientifiques. Présence de mains ou d’objets sur les photos (image 441128 du dataset).<br><br>                 
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )


    st.image(rep_img_3+'441128.jpg')


    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        <br>
                        <b>Limitations identifiées : </b><br>
                        &emsp;- Sur et sous-représentation de certaines espèces<br>
                        &emsp;- Faible indice de confiance pour certaines images<br>
                        &emsp;- Peu d’images de stades juvéniles<br>
                        &emsp;- Angles « dessous du chapeau » sous représentés<br><br>                 
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )

# Pre-processing et feature engineering
    st.markdown(
                    f"""
                    <p style='color:#400080; font-size:20px; font-weight:800; text-align: left'></b>
                        Pre-processing et feature engineering : 
                    <b></p>                    
                    """,
                    unsafe_allow_html=True
                )

    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        <b>Nettoyage et traitement</b><br><br>
                        &emsp;<b>- Filtre sur l’indice de confiance (feature « confidence » du dataset)</b><br><br>
                        &emsp; &emsp;Distribution de la variable :<br>          
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )


    titre_c = "Répartition des classes"
    labelx_c = "Classes"
    labely_c = "Nombre"
    countplot_display(df_main_dataset['gbif_info/confidence'], titre_c, labelx_c, labely_c)


    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                         &emsp; &emsp;=> Choix de filtrer au seuil de 92%, qui est un bon compromis entre qualité et nombre de valeurs à exploiter.<br><br>
                        <b>- Classement par script des fichiers images en sous-répertoires relatifs à la variable cible :</b> afin de permettre d'utiliser l'utilitaire Keras <b>image_dataset_from_directory</b> pour appeler les images dans les futurs modèles et gagner en rapidité d'exécution.<br><br>
                        <b>- Over-sampling et Under-sampling :<b> Rééquilibrage des classes afin d’améliorer l’entrainement du modèle.<br><br>
                        <b>Transformations :</b><br>
                            &emsp;<b>•	Normalisation</b> des données images par 1/255 (en lien avec le nb de pixels qui va de 0 à 255) => optimisation de l’entrainement.<br>
                            &emsp;<b>•	Redimensionnement</b> à 224px x 224px, format standard, bon compromis qualité et performance.<br>
                            &emsp;<b>•	Data augmentation :</b> Etirement, Zoom, Retournement, Rotation, Contraste. Afin de permettre une plus grande variété de données, et améliorer l’entrainement.<br><br>
                            &emsp;=> Toutes ces transformations sont réalisées par le pipeline Keras et sont intégrées directement dans le modèle.<br><br>        
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )

# Visualisations et Statistiques
    st.markdown(
                    f"""
                    <p style='color:#400080; font-size:20px; font-weight:800; text-align: left'></b>
                        Visualisations et Statistiques : 
                    <b></p>                    
                    """,
                    unsafe_allow_html=True
                )


    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        <b><u>Visualisations :</u><b><br><br>
                        Visualisations du dataset final, constitué de la fusion du dataset "champignons_france_top30.csv" et du dataset initial "observations_mushroom.csv" (filtré avec l'indice de confiance >= 92%) :<br><br>
                        Chiffres en nombre d’observations, avant rééquilibrage des classes.<br><br>
                        <u>Analyse des variables :</u><br>
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )

    cols_2 = cols + ['Nom commun', 'Statut', 'Habitat typique']

    report_taxons = df_dataset_30[cols_2].describe(include='object')
    st.dataframe(report_taxons)

    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                    <br>
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )

    # Liste déroulante pour affichage du barplot :

    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        <b><u>Affichage du du Top 30 Espèces France, répartition par :</u><b>
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )


    repart_fr = st.selectbox("",
                           ["ordre", "famille", "genre", "Nom commun", "Habitat typique", "Statut"])
    

    if repart_fr == "ordre" :
        repart = "gbif_info/" + "order"
    elif repart_fr == "famille" :
        repart = "gbif_info/" + "family"
    elif repart_fr == "genre" :
        repart = "gbif_info/" + "genus"
    else :
        repart = repart_fr


    # Affichage barplot

    ordres = df_dataset_30[repart].value_counts()

    titre_b = f"Top 30 Espèces France, répartition par {repart_fr} (en nb d'observations)"
    labelx_b = "Nombre d'observations"
    labely_b = repart_fr
    barplot_display(ordres, titre_b, labelx_b, labely_b)

    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                    <br>
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )



# Affichage par espèce :

    # Liste déroulante pour le choix de l'espèce à afficher

    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        <b><u>Affichage par espèce :</u><b>
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )
    specie_choice = st.selectbox("", sorted(list(lat_fra.keys())))

    specie_choice = lat_fra[specie_choice]

    show_n_images(train_ds, class_names, specie_choice)   


# Relations entre variables

    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        <br><br>
                        <b><u>Relations entre variables :</u><b><br>
                        Sans objet. Nos modèles tournent pour classifier la variable cible 'target' relative aux espèces de champignons d’une part, et les données images d’autre part.<br><br>
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )

# Distribution & outliers / Analyse statistique

    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        <b><u>Identification d'outliers visuels :</u><b><br><br>
                        photo microscope (image 103116 du dataset) :
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )
    st.image(rep_img_3+'103116.jpg', width=200)


    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        <br><br>
                        présence d’une règle (image 634315 du dataset) :
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )
    st.image(rep_img_3+'634315.jpg', width=200)

    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        <br><br>
                        présence d’une main (image 635996 du dataset) :
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )
    st.image(rep_img_3+'635996.jpg', width=200)



    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        <br><br>
                        présence d’autres végétaux (image 34224 du dataset) :
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )
    st.image(rep_img_3+'34224.jpg', width=200)


    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        <br><br>
                        <b>=> Détection d'anomalies via l'utilisation de modèles pré-entrainés (ResNet50, YOLO).<b><br>
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )



######################################################################################################################
######################################################################################################################

# Page Etapes de réalisation


if menu == "Etapes de réalisation":


# Titre
    st.markdown(
                    f"""
                    <p style='color:#004B23; font-size:30px; font-weight:1000; text-align: center'>
                        - Etapes de réalisation du projet -
                    </p>                    
                    """,
                    unsafe_allow_html=True
                )
    st.markdown("---")

# Classification du problème
    st.markdown(
                    f"""
                    <p style='color:#400080; font-size:20px; font-weight:800; text-align: left'></b>
                            Classification du problème : 
                    <b></p>                    
                    """,
                    unsafe_allow_html=True
                )

    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        <b><u>Type de problème :</u><b><br>
                        Ce projet relève d’un problème de classification supervisée et plus précisément de classification multi classe. En effet, l’objectif est de prédire, à partir d’une image de champignon, à quelle espèce celui-ci appartient, parmi les 30 espèces les plus courantes sur le territoire métropolitain français.<br><br>
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )

    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        <b><u>Tâche de machine learning :</u><b><br>
                        La tâche correspond à de la reconnaissance visuelle de champignons (espèces biologiques « fongiques » ni animales, ni végétales), c’est-à-dire à une tâche de computer-vision.<br><br>
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )
    
    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        <b><u>Métrique principale :<b></u><br>
                        La métrique principale d’évaluation choisie est la <b>précision<b>.<br>
                        Ce choix provient du contexte sensible de sécurité alimentaire que peut présenter l’application du modèle. Il est en effet indispensable de minimiser les faux positifs, afin d’éviter qu’un champignon vénéneux soit classé à tort comme comestible.
                        Une forte précision garantit que les prédictions positives sont fiables, même si cela implique de manquer certaines espèces comestibles. Un taux un peu plus important de faux négatifs peut donc être toléré dans ce cas.
                        <br><br>
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )

    image_edible = get_base64_of_image(rep_img_2 + "Edible.jpg")
    st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/jpg;base64,{image_edible}" 
                    style="width:20%; height:auto; border-radius:10px;">
            </div>
            """,
            unsafe_allow_html=True
        )



    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        <br>
                        <b><u>Autres métriques utilisées :</u><b><br>
                            &emsp;<b>•	Rappel (recall) :</b> mesure la capacité du modèle à identifier toutes les occurrences d’une espèce.<br>
                            &emsp;<b>•	F1-score :</b> moyenne harmonisée entre précision et rappel.<br>
                            &emsp;<b>•	Accuracy globale :</b> indique la proportion d’images correctement classées toutes espèces confondues.<br>
                            &emsp;<b>•	Matrice de confusion :</b> permet d’identifier les espèces fréquemment confondues.<br>
                            &emsp;•	Une <b>Matrice de confusion spécifique</b> pour différencier les performances entre espèces comestibles et vénéneuses a également été mise en place.
                        <br><br>
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )


# Choix du problème et optimisation
    st.markdown(
                    f"""
                    <p style='color:#400080; font-size:20px; font-weight:800; text-align: left'></b>
                            Choix du problème et optimisation : 
                    <b></p>                    
                    """,
                    unsafe_allow_html=True
                )


    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        <b><u>Algorithmes testés :<b></u><br>
                            &emsp;•	Modèles traditionnels de machine learning :<br>
                            &emsp;&emsp;o	Random Forest<br>
                            &emsp;&emsp;o	K-Nearest Neighbors (KNN)<br>
                            &emsp;•	Modèles de Deep Learning :<br>
                            &emsp;&emsp;o	Deep Neural Network (DNN)<br>
                            &emsp;&emsp;o	Convolutional Neural Network (CNN)<br>
                            &emsp;&emsp;o	Convolutional Neural Network (CNN) avec Transfer Learning<br><br>
                            &emsp;Les modèles classiques (RF, KNN) ont servi de bases pour estimer les performances atteignables sans apprentissage profond.<br><br>
                            &emsp;Cependant, la nature visuelle et complexe des données rend ces modèles moins performants que les architectures convolutives, mieux adaptées à la détection de motifs et textures.<br><br>
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )
    

    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        <b><u>Modèle retenu :</u><b><br>
                        Le modèle CNN a été retenu comme modèle final. Celui-ci a été entrainé de plusieurs manières différentes :<br>
                            &emsp;•	Selon un jeu de données filtrées par ResNet50, sans transfer learning puis avec transfer learning ResNet50.<br>
                            &emsp;•	Selon un jeu de données filtrées par YOLOv11, sans transfer learning puis avec transfer learning MobileNet V2.<br><br>
                            &emsp;Résultats obtenus <u>sans</u> transfer learning :<br>
                            &emsp;•	Une précision moyenne de 0.66 (ResNet50), 0.61 (YOLOv11)<br>
                            &emsp;•	Une accuracy globale de 0.63 (ResNet50), 0.58 (YOLOv11)<br>
                            &emsp;•	Des scores F1 supérieurs à 0.80 pour certaines classes bien représentées dans les deux cas.<br><br>
                            &emsp;Résultats obtenus <u>avec</u> transfer learning :<br>
                            &emsp;•	Une précision moyenne de 0.89 (ResNet50), 0.87 (MobileNet V2)<br>
                            &emsp;•	Une accuracy globale de 0.88 (ResNet50), 0.86 (MobileNet V2)<br>
                            &emsp;•	Des scores F1 de 1.00 pour certaines classes bien représentées dans les deux cas.<br><br>
                            &emsp;Ce modèle se distingue par :<br>
                            &emsp;•	Sa capacité à extraire automatiquement des caractéristiques discriminantes des images (textures, formes, couleurs)<br>
                            &emsp;•	Des résultats visuellement cohérents confirmés par les méthodes d’interprétabilité (Grad-CAM, SHAP, Feature Maps)<br>
                            &emsp;•	Et une meilleure généralisation que les modèles classiques sur le jeu de test.<br><br>
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )



    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        <b><u>Optimisation des paramètres :</u><b><br>
                        Une phase d’optimisation a été réalisée via un ajustement manuel pour déterminer les meilleurs hyperparamètres :<br>
                            &emsp;•	nombre de couches convolutives<br>
                            &emsp;•	taille des filtres<br>
                            &emsp;•	taux de dropout<br><br>
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )
    

    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        <b><u>Modèles avancés :</u><b><br>
                        Un CNN profond (Deep Learning) a été testé car les modèles classiques atteignaient rapidement une saturation de performance.<br>
                        Le bagging et le boosting n’ont pas été retenus : bien qu’efficaces sur des données tabulaires, ils ne sont pas adaptés à la complexité spatiale et visuelle des images.<br><br>
                     </p>                   
                    """,
                    unsafe_allow_html=True
                )
    

# Interprétation des résultats
    st.markdown(
                    f"""
                    <p style='color:#400080; font-size:20px; font-weight:800; text-align: left'></b>
                            Interprétation des résultats : 
                    <b></p>                    
                    """,
                    unsafe_allow_html=True
                )


    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        <b><u>Analyse des erreurs :</u><b><br>
                        Malgré un équilibrage préalable du dataset via de l’over/under sampling et de la data augmentation (rotations, zoom, étirements, …), l’analyse de la matrice de confusion a permis d’identifier les espèces les plus souvent confondues entre elles.<br><br>
                        Cette étape a permis d’observer que certaines confusions concernaient des espèces visuellement très proches, notamment les Amanita Rubescens et les Coprinus Comatus.<br><br>
                     </p>                   
                    """,
                    unsafe_allow_html=True
                )

    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        Amanita Rubescens (images 438142 et 155606 du dataset) :
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(rep_img_3+'438142.jpg')
    with col2:
        st.image(rep_img_3+'155606.jpg')


    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        <br><br>
                        Coprinus Comatus (images 4381696 et 497972 du dataset) :
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(rep_img_3+'81696.jpg')
    with col2:
        st.image(rep_img_3+'497972.jpg')


    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        <br><br>
                        <b><u>Techniques d’interprétabilité :</u><b><br>
                        Afin de s’assurer que le modèle apprend bien les caractéristiques pertinentes :<br>
                            &emsp;<b>•	Grad-CAM :</b> a été utilisé pour visualiser les zones de l’image sur lesquelles le CNN fonde sa prédiction. Les activations montrent que le modèle se concentre bien sur des zones cohérentes (chapeau, pied, lamelles).<br>
                            &emsp;<b>•	SHAP :</b> a permis d’évaluer l’importance des caractéristiques extraites pour chaque prédiction.<br>        
                            &emsp;<b>•	Features Map :</b> ont confirmé la bonne hiérarchie d’apprentissage des filtres (formes simples en bas niveau, textures et détails spécifiques en haut niveau).<br><br>
                        Ces analyses confirment la <b>bonne interprétabilité</b> et fiabilité visuelle du modèle.
                        a permis d’évaluer l’importance des caractéristiques extraites pour chaque prédiction.<br><br>
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )


    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        <b><u>Améliorations significatives :</u><b><br>
                        Les principales améliorations de performance ont été obtenues grâce à :<br>
                            &emsp;<b>•	L’utilisation d’un CNN :</b> (apprentissage automatique des caractéristiques visuelles)<br>
                            &emsp;<b>•	L’analyse Grad-CAM :</b> permettant de vérifier et ajuster la cohérence des zones d’attention<br>
                            &emsp;<b>•	Le recours à une architecture pré-entraînée</b> a permis une amélioration très significative (+0.25 points d’accuracy globale).<br><br>         
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )



######################################################################################################################
######################################################################################################################

# Page Modèles

if menu == "Modèles":

    st.markdown(
    f"""
    <p style='color:#004B23; font-size:22px; font-weight:600;'>
        Modèle sélectionné : {model_choice}
    </p>
    """,
    unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown(
                    f"""
                    <p style='color:#004B23; font-size:30px; font-weight:1000; text-align: center'>
                        - Performance du modèle -
                    </p>                    
                    """,
                    unsafe_allow_html=True
                )
    st.markdown("---")


    # Prédictions
    if model_type == "sklearn": # RF, KNN
        y_pred_1 = model_loaded.predict(X_val_flat) 
    
        # Rapport de classification
        st.markdown(
                f"""
                <p style='color:#004B23; font-size:22px; font-weight:600;'>
                    Rapport de classification :
                </p>
                """,
                unsafe_allow_html=True
            )
        
        report = classification_report(y_val, y_pred_1, target_names=class_names, output_dict=True)
        st.dataframe(report)
        st.markdown("---")

        # Accuracy
        accuracy = np.mean(y_pred_1 == y_val)
        st.markdown(
                    f"""
                    <p style='color:#004B23; font-size:22px; font-weight:600;'>
                        Accuracy :
                    </p>
                    <p style='color:#004B23; font-size:30px; font-weight:1000;'>
                        {accuracy*100:.2f}%
                    </p>                    
                    """,
                    unsafe_allow_html=True
                )
        st.markdown("---")
        
        # Matrice de confusion
        st.markdown(
                f"""
                <p style='color:#004B23; font-size:22px; font-weight:600;'>
                    Matrice de confusion du modèle :
                </p>
                """,
                unsafe_allow_html=True
            )
        conf_mat_affichage(y_val, y_pred_1, class_names)   
        st.markdown("---")
    
    else:   # Si modèle Keras (DNN, CNN)
        
        (
            y_true,
            y_pred_1, y_score_1,
            y_pred_2, y_score_2,
            y_pred_3, y_score_3,
            images_list
        ) = get_predictions_and_labels(val_ds, model_loaded)

    
        # Rapport de classification
        st.markdown(
                f"""
                <p style='color:#004B23; font-size:22px; font-weight:600;'>
                    Rapport de classification :
                </p>
                """,
                unsafe_allow_html=True
            )

        report = classification_report(y_true, y_pred_1, target_names=class_names, output_dict=True)
        st.dataframe(report)
        st.markdown("---")


        # Accuracy
        accuracy = np.mean(y_pred_1 == y_true)
        st.markdown(
                    f"""
                    <p style='color:#004B23; font-size:22px; font-weight:600;'>
                        Accuracy :
                    </p>
                    <p style='color:#004B23; font-size:30px; font-weight:1000;'>
                        {accuracy*100:.2f}%
                    </p>                    
                    """,
                    unsafe_allow_html=True
                )
        st.markdown("---")


        # Matrice de confusion
        st.markdown(
                f"""
                <p style='color:#004B23; font-size:22px; font-weight:600;'>
                    Matrice de confusion du modèle :
                </p>
                """,
                unsafe_allow_html=True
            )       
        conf_mat_affichage(y_true, y_pred_1, class_names)   
        st.markdown("---")    

        # Affichage du résumé du modèle
        st.markdown(
                f"""
                <p style='color:#004B23; font-size:22px; font-weight:600;'>
                    Affichage du résumé du modèle :
                </p>
                """,
                unsafe_allow_html=True
            )

        summary_text = get_model_summary(model_loaded)
        with open("summary.txt", "w", encoding="utf-8") as fichier:

            summary_text = summary_text.replace("┏", " ").replace("━", " ").replace("┳", " ").replace("┓", " ")
            summary_text = summary_text.replace("┡", " ").replace("╇", " ").replace("┩", " ")
            summary_text = summary_text.replace("├", " ").replace("─", " ").replace("┼", " ").replace("┤", " ")
            summary_text = summary_text.replace("└", " ").replace("┴", " ").replace("┘", " ")
            fichier.write(summary_text)


        st.markdown(
            f"""
            <div style="
            max-width: 600px;
            background-color: black;
            margin: auto;
            text-align: left;
            ">
                <p style="
                font-size:8px;
                color = #004B23;
                ">
                    {summary_text}
            """,
            unsafe_allow_html=True
        )
        st.markdown("---")
        

        # Affichage des courbes d’entraînement
        st.markdown(
                f"""
                <p style='color:#004B23; font-size:22px; font-weight:600;'>
                    Affichage des courbes d’entraînement :
                </p>
                """,
                unsafe_allow_html=True
            )
        
        training_curves(history_model_loaded)
        st.markdown("---")


######################################################################################################################
######################################################################################################################


# Page Interprétabilité


elif menu == "Interprétabilité":

    st.markdown(
    f"""
    <p style='color:#004B23; font-size:22px; font-weight:600;'>
        Modèle sélectionné : {model_choice}
    </p>
    """,
    unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown(
                    f"""
                    <p style='color:#004B23; font-size:30px; font-weight:1000; text-align: center'>
                        - Interprétabilité du modèle -
                    </p>                    
                    """,
                    unsafe_allow_html=True
                )
    st.markdown("---")

    if model_type == "sklearn":

        st.markdown(
                f"""
                <p style='color:#ce0000; font-size:22px; font-weight:600;'>
                    Merci de sélectionner un modèle DNN ou CNN
                </p>
                """,
                unsafe_allow_html=True
            )       


    else:   # Si modèle Keras (DNN, CNN)
        
        (
            y_true,
            y_pred_1, y_score_1,
            y_pred_2, y_score_2,
            y_pred_3, y_score_3,
            images_list
        ) = get_predictions_and_labels(val_ds, model_loaded)    

        # Choix de l'espèce

        st.markdown(
                    f"""
                    <p style='color:#004B23; font-size:22px; font-weight:600;'>
                        Choix de l'espèce :
                    </p>
                    """,
                    unsafe_allow_html=True
                )
        specie_choice = st.selectbox("", sorted(list(lat_fra.keys())))

        specie_choice = lat_fra[specie_choice]

        st.markdown("---")


        # Bonne / Mauvaise prédiction

        st.markdown(
                    f"""
                    <p style='color:#004B23; font-size:22px; font-weight:600;'>
                        Prédictions :
                    </p>
                    """,
                    unsafe_allow_html=True
                )

        verdict = st.selectbox("", ["Exactes", "Erronées"])

        st.markdown("---")


        # Aperçu des résultats

        st.markdown(
                    f"""
                    <p style='color:#004B23; font-size:22px; font-weight:600;'>
                        Aperçu des résultats :
                    </p>
                    """,
                    unsafe_allow_html=True
                )

        liste_index = images_show(images_list, y_pred_1, y_pred_2, y_pred_3, y_score_1, y_score_2, y_score_3, y_true, n_img_show, specie_choice, class_names, verdict)


        # Méthode Grad-CAM

        st.markdown(
                f"""
                <p style='color:#004B23; font-size:22px; font-weight:600;'>
                    Méthode Grad-CAM (uniquement modèle CNN)
                </p>
                """,
                unsafe_allow_html=True
            )

        liste_index = np.array(liste_index)
        images_analyse = images_list[liste_index]
        show_grad_cam(images_analyse, model_loaded)


        # Méthode SHAP
        
        st.markdown(
                f"""
                <p style='color:#004B23; font-size:22px; font-weight:600;'>
                    Méthode SHAP
                </p>
                """,
                unsafe_allow_html=True
            )
        
        shap_function(images_analyse, model_loaded, class_names)

        st.markdown("---") 


        # Méthode Features Map
        
        st.markdown(
                f"""
                <p style='color:#004B23; font-size:22px; font-weight:600;'>
                    Méthode Fatures Map (uniquement modèle CNN)
                </p>
                """,
                unsafe_allow_html=True
            )
        
        show_feature_maps(images_analyse, model_loaded)

        st.markdown("---")



######################################################################################################################
######################################################################################################################



# Page Téléchargement d'une image

elif menu == "Test sur une image":

    st.markdown(
    f"""
    <p style='color:#004B23; font-size:22px; font-weight:600;'>
        Modèle sélectionné : {model_choice}
    </p>
    """,
    unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown(
                    f"""
                    <p style='color:#004B23; font-size:30px; font-weight:1000; text-align: center'>
                        - Test sur une image -
                    </p>                    
                    """,
                    unsafe_allow_html=True
                )
    st.markdown("---")


    if model_type == "sklearn":

        st.markdown(
                f"""
                <p style='color:#ce0000; font-size:22px; font-weight:600;'>
                    Merci de sélectionner un modèle DNN ou CNN
                </p>
                """,
                unsafe_allow_html=True
            )    

    else :

        uploaded_file = st.file_uploader("Upload une image :", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:


            #st.image(uploaded_file, width=200)

            # Nom de l'image
            img_name = uploaded_file.name



            img_tensor = tf.image.resize(tf.image.decode_image(uploaded_file.read(), channels=3), [224, 224])
            img_tensor = tf.expand_dims(img_tensor, axis=0)

            single_ds = tf.data.Dataset.from_tensor_slices((img_tensor, [0])).batch(1)

            affichage_top3(model_loaded, single_ds, class_names, img_name)


            # Méthode Grad-CAM

            st.markdown(
                    f"""
                    <p style='color:#004B23; font-size:22px; font-weight:600;'>
                        Méthode Grad-CAM (uniquement modèle CNN)
                    </p>
                    """,
                    unsafe_allow_html=True
                )


            show_grad_cam(img_tensor, model_loaded)


            # Méthode SHAP
            
            st.markdown(
                    f"""
                    <p style='color:#004B23; font-size:22px; font-weight:600;'>
                        Méthode SHAP
                    </p>
                    """,
                    unsafe_allow_html=True
                )
            
            shap_function(img_tensor, model_loaded, class_names)

            st.markdown("---") 


            # Méthode Features Map
            
            st.markdown(
                    f"""
                    <p style='color:#004B23; font-size:22px; font-weight:600;'>
                        Méthode Fatures Map (uniquement modèle CNN)
                    </p>
                    """,
                    unsafe_allow_html=True
                )
            
            show_feature_maps(img_tensor, model_loaded)

            st.markdown("---")


######################################################################################################################
######################################################################################################################



# Page Conclusion

elif menu == "Conclusion":

    
# Titre
    st.markdown(
                    f"""
                    <p style='color:#004B23; font-size:30px; font-weight:1000; text-align: center'>
                        - Conclusion -
                    </p>                    
                    """,
                    unsafe_allow_html=True
                )
    st.markdown("---")


    st.markdown(
                    f"""
                    <p style='color:#000000; font-size:18px; font-weight:500; text-align: left'>
                        Le projet démontre qu’un CNN correctement entraîné et interprété est capable de classifier efficacement des espèces de champignons à partir d’images, tout en respectant la contrainte essentielle de fiabilité des prédictions comestibles.<br><br>
                        Le modèle constitue une base robuste pour une application d’aide à la reconnaissance de champignons, à condition d’intégrer une gestion du risque adaptée aux cas incertains :<br><br>
                     </p>                    
                    """,
                    unsafe_allow_html=True
                )

    
    mc_output = get_base64_of_image(rep_img_2 + "output.png")
    st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{mc_output}" 
                    style="width:50%; height:auto; border-radius:10px;">
            </div>
            """,
            unsafe_allow_html=True
        )


######################################################################################################################
######################################################################################################################



# Page Remerciements

elif menu == "Remerciements":

    # Titre
    st.markdown(
                    f"""
                    <p style='color:#004B23; font-size:30px; font-weight:1000; text-align: center'>
                        - Remerciements -
                    </p>                    
                    """,
                    unsafe_allow_html=True
                )
    st.markdown("---")

    thanks = get_base64_of_image(rep_img_2 + "Thanks.png")
    st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{thanks}" 
                    style="width:50%; height:auto; border-radius:10px;">
            </div>
            """,
            unsafe_allow_html=True
        )
    

    