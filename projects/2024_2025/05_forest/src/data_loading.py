from pathlib import Path
import os
import shutil
from New_prog import *

import numpy as np
import pandas as pd
import huggingface_hub
import matplotlib.pyplot as plt
import tifffile

def download_data():
    out_folder = "data/raw/forest-plot-analysis"
    repository = "remote-sensing-ense3-grenoble-inp/forest-plot-analysis"

    cwd = Path(__file__).resolve().parents[1]
    target_directory = cwd / out_folder
    if not target_directory.exists():
        try:
            target_directory.mkdir(parents=True, exist_ok=True)
            huggingface_hub.snapshot_download(
                repo_id=repository,
                repo_type="dataset",
                local_dir=target_directory,
                token=os.getenv("HUGGINGFACE_TOKEN"),
            )
        except Exception as e:
            shutil.rmtree(target_directory)
            raise ValueError(
                f"Error downloading repository." +
                f"{e}"
            )

#def visualize_data(a,RVB):
    
        #Création des chemins fichiers
        
    cwd = Path(__file__).resolve().parents[1] #défini le chemin commun aux fichiers à ouvrir
    filename_hsi = f"data/raw/forest-plot-analysis/data/hi/{a}.tif" #Défini le nom de l'image à ouvrir
    filename_label = "data/raw/forest-plot-analysis/data/gt/df_pixel.csv" #Défini le nom de la carte des espèces
    file_hsi = cwd / filename_hsi #défini le chemin complet du fichier image à ouvrir
    file_label = cwd / filename_label #défini le chemin complet du fichier pixels
    
        #Gestion de l'affichage de l'image
    
    rgb_channels = RVB #Défini quelle gamme de couleur sera affichée
    plot_id = str(a) #défini le numéro de l'image à ouvrir
    
    hsi = tifffile.TiffFile(file_hsi) #Lit l'image et la traduit en Aray
    hsi = hsi.asarray() #Transforme l'image en array
    
    H, W, B = hsi.shape
    print("La taille de l'image ",a," est : ",H,"pixels de haut, ",W,"pixels de large, et ",B,"données spectrales") #Affiche la taille de l'image à afficher
    
    hsi = hsi[:, :, rgb_channels] #n'affiche que l'image avec la gamme de fréquence sélectionnée
    hsi = hsi / hsi.max() #divise l'image par le maximum du spectre -> ratio pour toutes les fréquences
    
        #Gestion des données spectrales
    
#    sp_data = hsi.reshape(-1, B)
    print("Le spectre du pixel 10 est : ",hsi[10,10,:])
    
    
        #Gestion de la carte des espèces colorée
    
    df = pd.read_csv(file_label) #Lis le fichier csv Pixels
    df = df[df["plotid"] == plot_id] #Sélectionne l'id des arbres 
    img_label = np.zeros((H, W, 3)) #crée la matrice en 3D avec : la Hauteur, la Largeur (matrice 2D), la profondeur donc les n gammes de spectre

    tree_ids = df["specie"].unique() #Pas sûr mais enlève les doublons d'espèce dans le fichier csv
    color_map = {tree_id: color for tree_id, color in zip(tree_ids, plt.cm.tab20.colors)} #attribue une couleur à chaque espèce
    for _, row in df.iterrows(): #Pour chaque ligne dans les lignes du tableau CSV
        color = color_map[row["specie"]] #défini une variable couleur en fonction de l'espèce lue
        img_label[int(row["row"]), int(row["col"])] = color #atttribue à la position (ligne,colonne) la couleur de l'espèce, ie dessine la carte des espèces en couleurs

        #Affichage de l'image et de la carte
    fig, ax = plt.subplots(1, 2) #crée les graphes à montrer

    ax[0].imshow(hsi) #Affiche l'image rgb
    ax[0].set_title("Hyperspectral image") 
    ax[1].imshow(img_label) #Affiche la carte couleur des 
    ax[1].set_title("Labels map")
    
    # legend_elements = [
    #     mpatches.Patch(color=color, label=class_name)
    #     for class_name, color in color_map.items()
    # ]
    # ax.legend(
    #     handles=legend_elements,
    #     loc='upper right',
    #     bbox_to_anchor=(1.25, 1),
    #     title="Classes",
    # )
   
def main():
    visualize_data("1",[64, 28, 15])
    show_moy_sp("1")

if __name__ == "__main__":
    main()