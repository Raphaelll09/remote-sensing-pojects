"""
show.py

Module d'affichage et de visualisation pour l’exploration et l’analyse des données :
- Affichage des spectres par pixel ou par espèce
- Affichage des images hyperspectrales RVB
- Affichage des cartes de classes (espèces d’arbres)
- Visualisation des performances des modèles de classification

Fonctionnalités :
- Tracés des spectres moyens par espèce, avec incertitude à 95 %
- Superposition RVB + vérité terrain
- Matrices de confusion et courbes de scores comparatives (accuracy, f1)
- Interface compatible avec `models.py` pour le diagnostic de performances

Fonctions principales :
    show_image_RVB()       : Affiche une image HSI selon bandes RVB
    show_map_specie()      : Affiche la carte d'espèces (vérité terrain)
    show_spectre_moy()     : Affiche un spectre moyen avec incertitude
    show_scores()          : Affiche comparativement les scores des algos

Projet : Classification d’espèces d’arbres par télédétection (HSI + LiDAR)
"""

#%% Librairies

from data import *
from models import *

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#%% Fonctions révisées

def show_spectre_pixel(pixel: np.ndarray,f: np.ndarray,pos: list[int],B: int):
    
    """
    Affiche le spectre d’un pixel individuel sur un graphique.

    Args:
        pixel (np.ndarray): Intensités spectrales du pixel (160 bandes).
        f (np.ndarray): Longueurs d’onde correspondantes (400–1000 nm).
        pos (list[int]): Coordonnées [x, y] du pixel dans l’image.
        B (int): Nombre total de bandes spectrales (typiquement 160).

    Returns:
        None
    """
    
    x,y=pos
    plt.plot(f,pixel)
    plt.title("Spectre du pixel ("+str(x)+","+str(y)+") avec "+str(B)+" longueurs d'ondes")
    plt.xlabel("Fréquence")
    plt.ylabel("Intensité")
    plt.show()
 

def show_image_RVB(image_id: str,RVB: list[int], size=False, ax=None):
    
    """
    Affiche une image hyperspectrale dans une composition couleur (RVB).

    Args:
        image_id (str): Identifiant de l’image.
        RVB (list[int]): Indices des bandes à utiliser pour le rouge, vert et bleu.
        size (bool): Si True, affiche la taille de l’image (H, W, B).
        ax (matplotlib.axes.Axes, optional): Axe sur lequel afficher l’image.

    Returns:
        None
    """
    
    hsi=HSI(image_id)
    
    H, W, B = hsi.shape
    
    if size : print("La taille de l'image est : ",H,"pixels de haut, ",W,"pixels de large, et ",B,"données spectrales")
    
    for j in RVB :
        if (j<0 or j>=B) :
            print("Une des bandes RVB n'est pas comprise entre 0 et",B)
            return 0
    
    hsi = hsi[:, :, RVB] #n'affiche que l'image avec la gamme de fréquence sélectionnée
    hsi = hsi / hsi.max() #divise l'image par le maximum du spectre -> ratio pour toutes les fréquences

    if ax==None :  
        fig, ax = plt.subplots(1, 1)
        ax.imshow(hsi) #Affiche la carte couleur des arbres
        ax.set_title(f"Image n°{image_id} RVB")
        plt.show()
    else :
        ax.imshow(hsi) #Affiche la carte couleur des arbres
        ax.set_title(f"Image n°{image_id} RVB")
        
     
def show_map_specie(image_id: str, ax=None, alph=None):
    
    """
    Affiche une carte couleur représentant les espèces d’arbres par pixel.

    Args:
        image_id (str): Identifiant de l’image.
        ax (matplotlib.axes.Axes, optional): Axe sur lequel dessiner la carte.
        alph (float, optional): Niveau de transparence pour superposition.

    Returns:
        None
    """
    
    mapname="df_pixel"
    
    hsi=HSI(image_id)
    H,W,B=hsi.shape
    
    gt=GT(mapname)
    
    gt = gt[gt["plotid"] == image_id] #Sélectionne l'id des arbres 
    img_label = np.zeros((H, W, 3)) #crée la matrice en 3D avec : la Hauteur, la Largeur (matrice 2D), la profondeur donc les n gammes de spectre
    
    tree_ids = gt["specie"].unique() #Pas sûr mais enlève les doublons d'espèce dans le fichier csv
    color_map = {tree_id: color for tree_id, color in zip(tree_ids, plt.cm.tab20.colors)} #attribue une couleur à chaque espèce
    for _, row in gt.iterrows(): #Pour chaque ligne dans les lignes du tableau CSV
        color = color_map[row["specie"]] #défini une variable couleur en fonction de l'espèce lue
        img_label[int(row["row"]), int(row["col"])] = color #atttribue à la position (ligne,colonne) la couleur de l'espèce, ie dessine la carte des espèces en couleurs
    
    if ax==None :
        fig, ax = plt.subplots(1, 1)
        ax.imshow(img_label)
        ax.set_title(f"Color Map Image n°{image_id}")
        plt.show()
        
    elif alph==None :
        ax.imshow(img_label) #Affiche la carte couleur des arbres    
        legend_elements = [mpatches.Patch(color=color, label=class_name) for class_name, color in color_map.items()]
        ax.legend(handles=legend_elements,loc='upper right',bbox_to_anchor=(1.2, 1),title="Classes")
            
    else:
        ax.imshow(img_label,alpha=alph)
    ax.set_title(f"Color Map Image n°{image_id}")    
    

def visualize_data(image_id: str,RVB: list[int],alpha=None):
    
    """
    Affiche côte à côte ou superposées :
    - L’image en composition RVB
    - La carte des espèces d’arbres

    Args:
        image_id (str): Identifiant de la parcelle/image.
        RVB (list[int]): Bandes à utiliser pour la représentation RVB.
        alpha (float or None): Si défini, applique une superposition semi-transparente.

    Returns:
        None
    """
    
    if alpha==None:
        fig, (ax1,ax2)=plt.subplots(1,2)
        show_image_RVB(image_id,RVB,0,ax1)
        show_map_specie(image_id,ax2)
        
    else:
        fig, ax = plt.subplots()
        show_image_RVB(image_id,RVB,0,ax)
        show_map_specie(image_id,ax,0.2)
    
    plt.show()  
    

def show_spectre_moy(specie: str, Sp_mean: np.ndarray, Sp_std: np.ndarray,f: np.ndarray, std=False):
    
    """
    Affiche le spectre moyen d’une espèce avec son intervalle de confiance.

    Args:
        specie (str): Code de l’espèce (ex: "PIAB").
        Sp_mean (np.ndarray): Moyenne spectrale (160 valeurs).
        Sp_std (np.ndarray): Incertitude à 95% (160 valeurs).
        f (np.ndarray): Longueurs d’onde correspondantes.

    Returns:
        None
    """
    
    plt.plot(f,Sp_mean, label=specie)
    if std : 
        plt.fill_between(f,Sp_mean-Sp_std, Sp_mean+Sp_std,alpha=0.3,label='Confiance à 95%')
    plt.title("Moyenne du spectre de l'espèce : "+specie)
    plt.xlabel("Longueur d'onde (nm)")
    plt.ylabel("Intensité relative")
    plt.legend()
    plt.show()
    

def show_spectres_img(d_Spe: dict, f: np.ndarray, ax, fill=True):
    
    """
    Affiche les spectres moyens de toutes les espèces d’une image.

    Args:
        d_Spe (dict): Dictionnaire {espèce: (moyenne, incertitude)}.
        f (np.ndarray): Longueurs d’onde (160 bandes).
        ax (matplotlib.axes.Axes): Sous-figure sur laquelle tracer.
        fill (bool): Si True, ajoute l’enveloppe d’incertitude.

    Returns:
        None
    """
    
    for Spe in d_Spe :
        ax.plot(f,d_Spe[Spe][0],label=Spe)
        if fill : 
            ax.fill_between(f,d_Spe[Spe][0]-d_Spe[Spe][1], d_Spe[Spe][0]+d_Spe[Spe][1],alpha=0.3)
#           show_sp_curve(Spe,d_Spe[Spe][0],d_Spe[Spe][1],f)

def test_pixel_norm(image_id: str,row: int,col: int):
    
    """
    Teste et affiche à quelle espèce appartient un pixel donné, via distance euclidienne.

    Args:
        image_id (str): Identifiant de l’image.
        row (int): Ligne du pixel.
        col (int): Colonne du pixel.

    Returns:
        None
    """
    
    hsi=HSI(image_id)
    d_spe,f=show_spectre_moy(image_id,hsi)
    pixel=hsi[row][col]
    Id,norm=spectre_norm_dico(d_spe,pixel)
    print("Le pixel de coordonnées ",str((row,col))," correspond probablement à l'espèce :",Id,",avec une norme de :",str(norm))


def show_moy_spectre(image_id: str):
    
    """
    Affiche tous les spectres moyens des espèces d’une image.

    Args:
        image_id (str): Identifiant de la parcelle.

    Returns:
        None
    """
    
    hsi=HSI(image_id)
    fig, ax=plt.subplots()
    
    d_Spe,f=species_spectre_moy(image_id,hsi)
    show_spectre_img(d_Spe,f,ax)
    ax.legend()
    plt.title(f"Spectre moyen des espèces sur le l'image {image_id}")
    plt.show()

    
#%% Machine Learning

def show_res_Alg(accuracy,classification_report,confusion_matrix):
    
    """
    Affiche les résultats de classification d’un modèle.

    Args:
        accuracy (float): Score d’exactitude.
        classification_report (dict): Rapport détaillé des scores par classe.
        confusion_matrix (np.ndarray): Matrice de confusion (classe réelle vs prédite).

    Returns:
        None
    """
    
    print("Accuracy:", accuracy)
    print(classification_report)
    print(confusion_matrix)
    
def show_scores(data_accuracy: dict, data_macro: dict, data_weighted: dict, comp: str):
    
    """
        Affiche un graphique comparatif des scores des modèles avec ou sans traitement.

        Args:
            data_accuracy (dict): Résultats d’accuracy.
            data_macro (dict): Résultats F1 macro-averaged.
            data_weighted (dict): Résultats F1 weighted.
            comp (str): Nom du traitement comparé ("Scaler", "LiDAR", etc.)

        Returns:
            None
    """
    
    def show_accuracy(data: dict, ax, score: str, comp: str):
        
        df = pd.DataFrame(data)

        
        sns.barplot(data=df, x='Modèle', y=score, hue=comp, palette='Set2',ax=ax)

        ax.set_title(score+' des modèles avec et sans '+comp)
        ax.set_ylim(0, 1)
        ax.set_ylabel(score)
        ax.set_xlabel('Modèle')
        ax.legend(title='Données prétraitées')
    
    #figsize=(8, 5)
    fig, (ax1,ax2,ax3)=plt.subplots(1,3)
    show_accuracy(data_accuracy, ax1, "Accuracy",comp)
    show_accuracy(data_macro, ax2, "Macro avg",comp)
    show_accuracy(data_weighted, ax3, "Weighted avg",comp)
    
    plt.tight_layout()
    plt.show()
    
