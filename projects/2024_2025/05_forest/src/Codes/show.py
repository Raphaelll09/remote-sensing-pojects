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

import models as md
from data import *

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata


#%% Exploitation des données

def show_spectre_pixel(pixel: np.ndarray,f: np.ndarray,pos: list[int],B: int):
    
    """
    Plot the spectral signature of a single pixel.

    Args:
        pixel (np.ndarray): Pixel spectral intensities (length B).
        f (np.ndarray): Wavelength array (400–1000 nm, length B).
        pos (list[int]): Pixel coordinates [row, col].
        B (int): Number of bands.

    Returns:
        None
    """
    
    plt.plot(f,pixel)
    plt.title(f"Pixel spectrum at {pos} ({B} wavelenghts)")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.show(block=False)
 

def show_image_RVB(image_id: str,RVB=[65, 29, 16], size=False, ax=None):
    
    """
    Display a hyperspectral image using specified RGB bands.

    Args:
        image_id (str): Plot identifier.
        RVB (list[int]): Band indices for Red, Green, Blue.
        size (bool): If True, print image dimensions.
        ax (matplotlib.axes.Axes): Axes to plot on (optional).

    Returns:
        None
    """
    
    hsi=HSI(image_id)
    
    H, W, B = hsi.shape
    
    if size : print(f"Image size: {H}x{W} with {B} bands")
    
    for band in RVB :
        if (band<0 or band>=B) :
            raise ValueError(f"Band index {band} out of range [0, {B-1}]")
    
    hsi = hsi[:, :, RVB] #n'affiche que l'image avec la gamme de fréquence sélectionnée
    hsi = hsi / hsi.max() #divise l'image par le maximum du spectre -> ratio pour toutes les fréquences

    if ax is None :  
        fig, ax = plt.subplots(1, 1)
        ax.imshow(hsi) #Affiche la carte couleur des arbres
        ax.set_title(f"RGB Composite (bands {RVB}) – Image {image_id}")
        plt.show(block=False)
    else :
        ax.imshow(hsi) #Affiche la carte couleur des arbres
        ax.set_title(f"Image n°{image_id} RVB")
        
     
def show_map_specie(image_id: str, ax=None, alph=None):
    
    """
    Display a color map of tree species labels for an image.

    Args:
        image_id (str): Plot identifier.
        ax (matplotlib.axes.Axes): Axes to plot on (optional).
        alph (float): Transparency for overlay (optional).

    Returns:
        None
    """
    
    hsi=HSI(image_id)
    H, W, _=hsi.shape
    
    gt=GT("df_pixel")
    gt = gt[gt["plotid"] == image_id]
    
    img_label = np.zeros((H, W, 3)) 
    
    color_map = {tree_id: color for tree_id, color in zip(d_spec, plt.cm.tab20.colors)} #attribue une couleur à chaque espèce
    
    for _, row in gt.iterrows(): #Pour chaque ligne dans les lignes du tableau CSV
        if row["specie"]!='ND':
            color = color_map[row["specie"]] #défini une variable couleur en fonction de l'espèce lue
        img_label[int(row["row"]), int(row["col"])] = color #atttribue à la position (ligne,colonne) la couleur de l'espèce, ie dessine la carte des espèces en couleurs
    
    legend_elements = [mpatches.Patch(color=color, label=class_name) for class_name, color in color_map.items()]
    
    if ax==None :
        fig, ax = plt.subplots(1, 1)
        ax.imshow(img_label)
        ax.set_title(f"Color Map Image {image_id}")
        plt.show(block=False)
        
    elif alph==None :
        ax.imshow(img_label) #Affiche la carte couleur des arbres    
        
    else:
        ax.imshow(img_label,alpha=alph)

    ax.legend(handles=legend_elements,loc='upper right',bbox_to_anchor=(1.2, 1),title="Classes")
    ax.set_title(f"Color Map Image {image_id}")    
    

def visualize_data(image_id: str, RVB=[65, 29, 16], alpha=None):
    
    """
    Display side‐by‐side or overlay of HSI RGB composite and species map.

    Args:
        image_id (str): Plot identifier.
        RVB (list[int]): Band indices for Red, Green, Blue channels.
        alpha (float, optional): If provided, overlay species map with transparency.

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
    
    plt.show(block=False)  
    

def show_spectre_moy(specie: str, Sp_mean: np.ndarray, Sp_std: np.ndarray,f: np.ndarray, std=False):
    
    """
    Plot the mean spectrum of a species with confidence interval shading.

    Args:
        specie (str): Species code.
        Sp_mean (np.ndarray): Mean spectral values.
        Sp_std (np.ndarray): Uncertainty (±95%) values.
        f (np.ndarray): Wavelength array.
        std (bool): If True, shade the confidence interval.

    Returns:
        None
    """
    
    plt.plot(f,Sp_mean, label=specie)
    if std : 
        plt.fill_between(f,Sp_mean-Sp_std, Sp_mean+Sp_std,alpha=0.3,label='Confiance à 95%')
    plt.title(f"Mean Spectrum – {specie}")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (normalized)")
    plt.legend()
    plt.show(block=False)
    

def show_spectres_img(image_id: str, fill=True, normalize=False):
    
    """
    Plot mean spectra of all species in an image.

    Args:
        image_id (str): Plot identifier.
        fill (bool): If True, include confidence envelopes.
        normalize (bool): If True, plot normalized intensities.

    Returns:
        None
    """
    
    fig, ax=plt.subplots()
    
    f,d_Spe = md.species_spectre_moy(image_id, normalize=normalize)
    
    for Spe in d_Spe :
        ax.plot(f,d_Spe[Spe][0],label=Spe)
        if fill : 
            ax.fill_between(f,d_Spe[Spe][0]-d_Spe[Spe][1], d_Spe[Spe][0]+d_Spe[Spe][1],alpha=0.3)
#           show_sp_curve(Spe,d_Spe[Spe][0],d_Spe[Spe][1],f)
    plt.legend(title="Species")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (relative)" if normalize else "Intensity")
    plt.title(f"Mean Spectra – Image {image_id}")
    plt.show(block=False)

#%% Visualisation :

def chamrousse(image_id: str):
    
    """
    Visualize tree crowns (circles) and trunk positions from Chamrousse ground‐truth.

    Args:
        image_id (str): Plot identifier.

    Returns:
        None
    """
    
    df=GT("Chamrousse")
    num=df["plotid"]
    df=df[num==image_id]
    fam,espy,x,y,r,h=df["family"], df["specie"],df["x"],df["y"],df["radius"],df["height"]

    def cercle(xc, yc, r):
        theta = np.linspace(0, 2 * np.pi, 100)
        return (r * np.cos(theta) + xc, r * np.sin(theta) + yc)
    for xi, yi, ri in zip(x, y, r):
        x_circ, y_circ = cercle(xi, yi, ri)
        plt.plot(x_circ, y_circ, 'b')
        plt.axis('equal')
    
    plt.scatter(x, y, color='red')  
    plt.title("Arbes et leur canopée sur l'image "+image_id)
    plt.xlabel("Coordonnés en x")
    plt.ylabel("Coordonnées en y")
    plt.show(block=False)
   
def lidar_data(image_id: str):
    
    """
    Plot a LiDAR-derived canopy height map for an image.

    Args:
        image_id (str): Plot identifier.

    Returns:
        None
    """
    
    df=LiD(image_id)
    X, Y, Z=df["0"], df["1"], df["7"]
    mask= Z!= 0
    plt.figure()
    plt.tricontourf(X[mask], Y[mask], Z[mask], levels=50, cmap='terrain')
    plt.scatter(X[mask], Y[mask], c=Z[mask], s=5, cmap='terrain')
    plt.colorbar(label="Height (m)")
    plt.title(f"LiDAR Canopy Height – Image {image_id}")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.show(block=False)

#%% Machine Learning

def show_res_Alg(accuracy,classification_report,confusion_matrix):
    
    """
    Print classification results for a trained model.

    Args:
        accuracy (float): Overall accuracy score.
        classification_report (str): Text report of per-class scores.
        confusion_matrix (np.ndarray): Confusion matrix (true vs predicted).

    Returns:
        None
    """
    
    print(f"Accuracy: {accuracy:.3f}")
    print(classification_report)
    print("Confusion Matrix:\n", confusion_matrix)
    
def show_scores(data_accuracy: dict, data_macro: dict, data_weighted: dict, comp: str):
    
    """
    Plot comparative bar charts of model scores (accuracy, F1) with/without preprocessing.

    Args:
        data_accuracy (dict): {'Model': [...], 'Accuracy': [...], comp: [...]}
        data_macro (dict): {'Model': [...], 'Macro avg': [...], comp: [...]}
        data_weighted (dict): {'Model': [...], 'Weighted avg': [...], comp: [...]}
        comp (str): Description of the preprocessing compared (e.g., "Scaler").

    Returns:
        None
    """
    
    def plot_score(data: dict, ax, score: str, comp: str):
        
        df = pd.DataFrame(data)
        
        sns.barplot(data=df, x='Model', y=score, hue=comp, palette='Set2',ax=ax)

        ax.set_title(f"{score} with/without {comp}")
        ax.set_ylim(0, 1)
        ax.set_ylabel(score)
        ax.set_xlabel("Classifier")
        ax.legend(title='Données prétraitées')
    
    #figsize=(8, 5)
    fig, (ax1,ax2,ax3)=plt.subplots(1,3)
    plot_score(data_accuracy, ax1, "Accuracy",comp)
    plot_score(data_macro, ax2, "Macro avg",comp)
    plot_score(data_weighted, ax3, "Weighted avg",comp)
    
    plt.tight_layout()
    plt.show(block=False)
    
def show_colormap(image_id, filename, gT=False, ax=None, alph=None):
    
    """
    Display a color map of predicted (or ground-truth) species for an image.

    Args:
        image_id (str): Plot identifier.
        filename (str): Model filename (without .joblib) for predictions.
        gT (bool): If True, overlay only ground-truth locations.
        ax (matplotlib.axes.Axes): Axes to plot on (optional).
        alph (float): Transparency if overlayed (optional).

    Returns:
        None
    """
    
    row,col,pred = md.predict_img(image_id, filename, gt=gT)
    
    gt=GT("df_pixel")
    gt=gt[gt["plotid"] == image_id]
    
    hsi=HSI(image_id)
    H,W,B=hsi.shape
    img_label = np.zeros((H, W, 3))
    
    color_map = {tree_id: color for tree_id, color in zip(d_spec, plt.cm.tab20.colors)}
    
    for i in range(len(row)):
        if pred[i]!='ND':#Pour chaque ligne dans les lignes du tableau CSV
            color = color_map[pred[i]] #défini une variable couleur en fonction de l'espèce lue
        img_label[int(row[i]), int(col[i])] = color #atttribue à la position (ligne,colonne) la couleur de l'espèce, ie dessine la carte des espèces en couleurs
    
    legend_elements = [mpatches.Patch(color=color, label=class_name) for class_name, color in color_map.items()]
    
    if ax==None :
        fig, ax = plt.subplots()
        ax.imshow(img_label)
        ax.set_title(f"Predictions (Image {image_id})")
        plt.legend(handles=legend_elements,loc='upper right',bbox_to_anchor=(1.2, 1),title="Classes")
        plt.show(block=False)
        
    elif alph==None :
        ax.imshow(img_label) #Affiche la carte couleur des arbres    
        ax.set_title(f"Predictions (Image {image_id})" if not gT else f"Ground Truth (Image {image_id})")
    else:
        ax.imshow(img_label,alpha=alph)
        ax.set_title(f"Reality/Prediction superposition (Image {image_id})")
        
    ax.legend(handles=legend_elements,loc='upper right',bbox_to_anchor=(1.13, 1),title="Classes")
    

def compare_prediction(image_id: str, filename: str, alph=False):
    
    """
    Compare ground truth vs. predicted species maps side by side.

    Args:
        image_id (str): Plot identifier.
        filename (str): Model filename (without .joblib) for predictions.
        alph (bool): If True, do semi-transparent overlay comparison.

    Returns:
        None
    """
    
    if alph:
        fig,(ax1,ax2)=plt.subplots(1,2)
        show_colormap(image_id, filename, gT=True, ax=ax1,alph=alph)
        show_colormap(image_id, filename, ax=ax2)
        
    else :
        fig,(ax1,ax2,ax3)=plt.subplots(1,3)
        show_colormap(image_id, filename, gT=True, ax=ax2)
        show_colormap(image_id, filename, ax=ax3)
        
    show_map_specie(image_id,ax=ax1)    
    
    plt.tight_layout()      # Ajuste les sous-graphes pour éviter que le texte ne se chevauche
    plt.subplots_adjust(top=1.25)
    plt.suptitle(f"Prediction of the Algorithm {filename} on Image {image_id}",fontsize=20, fontweight='bold')
    plt.show(block=False)
    