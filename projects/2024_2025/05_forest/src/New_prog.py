from pathlib import Path
import os
import shutil
import copy

import numpy as np
import pandas as pd
import huggingface_hub
import matplotlib.pyplot as plt
import tifffile
import matplotlib.patches as mpatches


keys=["ABAL","ACPS","BEPE","BEsp","COAV","FASY","FREX","PIAB","PICE","PIUN","POTR","SOAR","SOAU"]
d_spec= {k : None for k in keys}


def sp_pixel(hsi, x, y, i=0):
    
    """This program allows to access and display the specter contained in a pixel from a tif image
    It takes for arguments :
        - hsi (array): the 3D array containing the image (opened with tifffile)
        - x (int): the x coordinate of the pixel you look for
        - y (int): the y coordinate of the pixel
        - i (bool): (optionnal argument) the indicator if you want to show the graph (i=1) or not (i=0). It is natively set to not show
    It returns : 
        - sp (array): the intensity of the specter
        - f  (array): the specter
    """
    
    H, W, B=hsi.shape #Get the shape of the image

    if x<0 or x>=W: 
        print("x is out of bound (W=",str(W),")")
        return 0

    elif (y<0 or y>=H): 
        print("y is out of bound (H=",str(H),")")
        return 0

    pixel=hsi[y][x][:]
    f=np.linspace(400,1000,B)

    if i==1 :
        plt.plot(f,pixel)
        plt.title("Pixel's Specter ("+str(x)+","+str(y)+") with "+str(B)+" fréquences")
        plt.xlabel("Frequence")
        plt.ylabel("Intensity")
        plt.show()
 
    return pixel,f

def open_hsi(filename):
    """Allow to open an hyper spectral image under the .tif format.
    The file to open has to be in the exact folder : data/raw/forest-plot-analysis/data/hi/
    It takes for arguments :
        - filename (str): The name of the file to open
    It returns :
        -hsi (array): a 3D array representing the image
    """
    cwd = Path(__file__).resolve().parents[1]
    
    filename_hsi ="data/raw/forest-plot-analysis/data/hi/"+filename+".tif"
    file_hsi = cwd / filename_hsi
    
    hsi = tifffile.TiffFile(file_hsi) #Lit l'image et la traduit en Aray
    hsi = hsi.asarray()
    return hsi

def show_image(id_image,RVB, i=1, ax=None):
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
    hsi=open_hsi(id_image)
    H, W, B = hsi.shape
    if i==1 : print("La taille de l'image est : ",H,"pixels de haut, ",W,"pixels de large, et ",B,"données spectrales")
    for j in RVB :
        if (j<0 or j>=B) :
            print("Une des bandes RVB n'est pas comprise entre 0 et",B)
            return 0
    
    hsi = hsi[:, :, RVB] #n'affiche que l'image avec la gamme de fréquence sélectionnée
    hsi = hsi / hsi.max() #divise l'image par le maximum du spectre -> ratio pour toutes les fréquences

    if ax==None :  
        fig, ax = plt.subplots(1, 1)
    ax.imshow(hsi) #Affiche la carte couleur des arbres
    ax.set_title(f"Image n°{id_image} RVB")

def show_map(id_image,ax=0,alph=None):
    """Show the colored map of the tree distribution on an image.
    Needs a .csv coming from the folder : data/raw/forest-plot-analysis/data/gt/ 
    Arguments :
        - filename (str): the name of the folder containing the datas for the map
        - hsi (array): the image the map will be based on, necessarry for the dimensions
    Returns :
        - 0: if there is an error
        - 1: if the map has been displayed"""
    mapname="df_pixel"
    
    cwd = Path(__file__).resolve().parents[1] #défini le chemin commun aux fichiers à ouvrir
    filename_label = "data/raw/forest-plot-analysis/data/gt/"+mapname+".csv" #Défini le nom de la carte des espèces
    file_label = cwd / filename_label #défini le chemin complet du fichier pixels
    
    hsi=open_hsi(id_image)
    H,W,B=hsi.shape
    
    df = pd.read_csv(file_label) #Lis le fichier csv Pixels
    df = df[df["plotid"] == id_image] #Sélectionne l'id des arbres 
    img_label = np.zeros((H, W, 3)) #crée la matrice en 3D avec : la Hauteur, la Largeur (matrice 2D), la profondeur donc les n gammes de spectre
    
    tree_ids = df["specie"].unique() #Pas sûr mais enlève les doublons d'espèce dans le fichier csv
    color_map = {tree_id: color for tree_id, color in zip(tree_ids, plt.cm.tab20.colors)} #attribue une couleur à chaque espèce
    for _, row in df.iterrows(): #Pour chaque ligne dans les lignes du tableau CSV
        color = color_map[row["specie"]] #défini une variable couleur en fonction de l'espèce lue
        img_label[int(row["row"]), int(row["col"])] = color #atttribue à la position (ligne,colonne) la couleur de l'espèce, ie dessine la carte des espèces en couleurs
    
    if ax==None :
        fig, ax = plt.subplots(1, 1)
        ax.imshow(img_label)

    elif alph==None :
        ax.imshow(img_label) #Affiche la carte couleur des arbres    
        legend_elements = [mpatches.Patch(color=color, label=class_name) for class_name, color in color_map.items()]
        ax.legend(handles=legend_elements,loc='upper right',bbox_to_anchor=(1.2, 1),title="Classes")
            
    else:
        ax.imshow(img_label,alpha=alph)
    ax.set_title(f"Color Map Image n°{id_image}")

def visualize_data(id_image,RVB,alpha=None):
    
    hsi=open_hsi(id_image)
    if alpha==None:
        fig, (ax1,ax2)=plt.subplots(1,2)
        show_image(id_image,RVB,0,ax1)
        show_map(id_image,ax2)
    else:
        fig, ax = plt.subplots()
        show_image(id_image,RVB,0,ax)
        show_map(id_image,ax,0.3)
    
    plt.show()
    
def pixels_species(image_id,specie):
    
    hsi=open_hsi(image_id)
    
    mapname="df_pixel"
    cwd = Path(__file__).resolve().parents[1] #défini le chemin commun aux fichiers à ouvrir
    mapname_label = "data/raw/forest-plot-analysis/data/gt/"+mapname+".csv" #Défini le nom de la carte des espèces
    map_label = cwd / mapname_label #défini le chemin complet du fichier pixels
    
    df = pd.read_csv(map_label) #Lis le fichier csv Pixels
    df = df[df["plotid"] == image_id]#Sélectionne l'id des arbres 

    if specie not in df["specie"].iloc() : 
        return 0
    
    df = df[df["specie"] == specie]
    
    row,col=df["row"],df["col"]
    
    Positions=np.zeros((len(row),2))
    for i in range(len(Positions)):
        Positions[i]=[row.iloc[i],col.iloc[i]]
    return Positions
        
def sp_pixel_moy(spe_pos,hsi):
    
    H, W, B=hsi.shape
    Sp_moy=np.zeros(B)
    
    for i in range(len(spe_pos)):
        x,y=int(spe_pos[i][1]),int(spe_pos[i][0])
        sp,f=sp_pixel(hsi,x,y)
        Sp_moy=Sp_moy+sp
        
    Sp_moy=Sp_moy/len(spe_pos)
    
    return Sp_moy

def sp_species_moy(image_id,hsi):
    
    H, W, B=hsi.shape
    f=np.linspace(400,1000,B)
    d_Spe={}
    
    for Spe in d_spec :
        Positions=pixels_species(image_id,Spe)
        if type(Positions)!=int:
            Sp_moy=sp_pixel_moy(Positions,hsi)
            d_Spe[Spe]=Sp_moy
    return d_Spe,f
    
def show_sp_dico(d_Spe,f,ax):
    for Spe in d_Spe :
        ax.plot(f,d_Spe[Spe],label=Spe)
        
def show_moy_sp(id_image):
    
    hsi=open_hsi(id_image)
    fig, ax=plt.subplots()
    hsi=open_hsi(id_image)
    d_Spe,f=sp_species_moy(id_image,hsi)
    show_sp_dico(d_Spe,f,ax)
    ax.legend()
    plt.title(f"Spectre moyen des espèces sur le l'image {id_image}")
    plt.show()
    
    
