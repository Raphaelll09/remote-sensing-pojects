#%% Librairies

from data import *
from pathlib import Path

import laspy
import tifffile
import math as m
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.stats import sem, t
from scipy.signal import savgol_filter

#%% Fonctions révisées

def show_spectre_pixel(pixel,f,pos,B):
    
    x,y=pos
    plt.plot(f,pixel)
    plt.title("Spectre du pixel ("+str(x)+","+str(y)+") avec "+str(B)+" longueurs d'ondes")
    plt.xlabel("Fréquence")
    plt.ylabel("Intensité")
    plt.show()
 

def show_image_RVB(image_id,RVB, size=False, ax=None):
    
    """Allow to visualize an image from an hsi array
    
    Also shows the 
    It takes for arguments :
        -hsi (array): the 3D file to visualize
        -RVB (list): a list giving the band selected
        -i (bool): (optional) if i=0 doesn't show the size of the image
    Returns :
        - 0 : if there is any error
        - 1 : if the image has been displayed
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
        
     
def show_map_specie(image_id, ax=None, alph=None):
    
    """Show the colored map of the tree distribution on an image.
    Needs a .csv coming from the folder : data/raw/forest-plot-analysis/data/gt/ 
    Arguments :
        - filename (str): the name of the folder containing the datas for the map
        - hsi (array): the image the map will be based on, necessarry for the dimensions
    Returns :
        - 0: if there is an error
        - 1: if the map has been displayed"""
    
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
    

def visualize_data(image_id,RVB,alpha=None):
    
    """"Cette fonction permet d'afficher une image dans la bande de couleur RVB désirée, ainsi que la carte couleur.
    Arguments :
    - image_id (str) : le nom de l'image à afficher
    - RVB (list) : la bande de couleur à afficher
    - alpha (None) : si un autre paramètre est donné que None, affiche l'image et la carte couleur superposées
    """
    
    if alpha==None:
        fig, (ax1,ax2)=plt.subplots(1,2)
        show_image_RVB(image_id,RVB,0,ax1)
        show_map(image_id,ax2)
        
    else:
        fig, ax = plt.subplots()
        show_image_RVB(image_id,RVB,0,ax)
        show_map(image_id,ax,0.2)
    
    plt.show()  
    

def show_spectre_moy(specie, Sp_mean,Sp_std,f):
    
    """ Affiche la courbe moyenne de "specie" avec son incertitude. 
    Arguments :
    - specie (str) : le nom de l'espèce dont on veut afficher le spectre
    - Sp_mean (array) : le spectre moyen de l'espèce à afficher
    - Sp_std (array) : l'incertitde à95%
    - f (array) : longueurs d'onde
    """
    
    plt.plot(f,Sp_mean,label=specie)
    plt.fill_between(f,Sp_mean-Sp_std, Sp_mean+Sp_std,alpha=0.3)
    plt.title("Moyenne du spectre de l'espèce : "+specie)
    plt.show()
    

def show_spectre_img(d_Spe, f, ax, fill=True):
    
    """"Permet d'afficher tous les spectres moyens d'une image avec leur incertitude (ou non) sur le même graphe.
    Arguments :
    - d_spe (dico) : dictionnaire des espèces contenus dans une image
    - f (array) : longueur d'onde
    - ax (axe de plt) : le subplot sur lequel afficher les courbes
    """
    
    for Spe in d_Spe :
        ax.plot(f,d_Spe[Spe][0],label=Spe)
        if fill : 
            ax.fill_between(f,d_Spe[Spe][0]-d_Spe[Spe][1], d_Spe[Spe][0]+d_Spe[Spe][1],alpha=0.3)
#           show_sp_curve(Spe,d_Spe[Spe][0],d_Spe[Spe][1],f)

def test_pixel_norm(image_id,row,col):
    
    """Permet de tester à quelle espèce un pixel appartient"""
    
    hsi=HSI(image_id)
    d_spe,f=species_spectre_moy(image_id,hsi)
    pixel=hsi[row][col]
    Id,norm=sp_norm_dico(d_spe,pixel)
    print("Le pixel de coordonnées ",str((row,col))," correspond probablement à l'espèce :",Id,",avec une norme de :",str(norm))



#%%
    
def show_moy_sp(image_id):
    
    """"Affiche toutes les courbes moyennes des espèces de l'image "image_id". """
    
    hsi=HSI(image_id)
    fig, ax=plt.subplots()
    
    d_Spe,f=sp_species_moy(image_id,hsi)
    show_spectre_img(d_Spe,f,ax)
    ax.legend()
    plt.title(f"Spectre moyen des espèces sur le l'image {image_id}")
    plt.show()
    

#%% Machine Learning

def show_res(accuracy,classification_report,confusion_matrix):
    
    print("Accuracy:", accuracy)
    print(classification_report)
    print(confusion_matrix)
    
    
def show_accuracy(data):
    
    df_acc = pd.DataFrame(data)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_acc, x='Modèle', y='Accuracy', hue='Scaler', palette='Set2')

    plt.title('Accuracy des modèles avec et sans Scaler')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.xlabel('Modèle')
    plt.legend(title='Données prétraitées')
    plt.tight_layout()
    plt.show()

