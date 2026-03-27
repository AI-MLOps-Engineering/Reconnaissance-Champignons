# Paramétrage


# Définition des chemins d'accès
rep_img = 'C:\\Users\\spreg\\Documents\\DataScience_images_Especes\\' # Chemin vers l'ensemble des images de champignons, triées par espèce
rep_img_2 = "C:\\Users\\spreg\\Documents\\DataScience\\avr25cds_reconnaissance_champignons\\images\\" # Chemin vers les photos tests
rep_img_3 = 'C:\\Users\\spreg\\Documents\\DataScience_images\\Fungi\\' # Chemin vers les photos tests, répertoire brut
rep_model = "C:\\Users\\spreg\\Documents\\DataScience\\avr25cds_reconnaissance_champignons\\models\\" # Chemin vers les sauvegardes de modèles
rep_data = "C:\\Users\\spreg\\Documents\\DataScience\\avr25cds_reconnaissance_champignons\\datasets\\" # Chemin vers les datasets csv



# Liste des modèles disponibles
models = {
    "Random_Forest": "Random_Forest_model.pkl",
    "KNN": "KNN_model.pkl",
    "DNN": "dnn_model.h5",
    "CNN": "cnn_model.h5",
    "CNN2": "cnn2_model.h5",
    "CNN_TL_ResNet50": "cnn_tl_model.keras",
    "CNN_TL_EfficientB0": "cnn_tl_model_efficientb0.keras",
    "CNN_TL_NasNet": "cnn_tl_model_nasnet.keras",
    "Ensemble": "cnn_tl_model_nasnet.keras"
}


# Nom des fichiers

main_dataset = 'observations_mushroom.csv'  # Dataset principal
dataset_30 = 'dataset_30_species.csv' # Dataset avec seulement les 30 espèces françaises prinicpales
taxons = 'champignons_france_top30.csv' # Fichier contenant les taxons de référence


# Dictionnaire des taxons
dico_taxons = {'gbif_info/kingdom' : 'regne',
                'gbif_info/phylum' : 'phylum',
                'gbif_info/class' : 'classe',
                'gbif_info/order' : 'ordre',
                'gbif_info/family' : 'famille',
                'gbif_info/genus' : 'genre',
                'gbif_info/species' : 'espece'}


# Nombre d'images par espèces à afficher dans la data exploration
n_species_img = 4


# Nombre d'images à afficher pour l'interprétabilité
n_img_show = 4




