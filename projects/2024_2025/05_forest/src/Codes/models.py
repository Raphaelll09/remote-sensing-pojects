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

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE


#%% Traitement de données

def pixel(image_id, x, y):
    
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
    
    hsi=HSI(image_id)
    H, W, B=hsi.shape #Get the shape of the image

    if x<0 or x>=W: 
        print("x is out of bound (W=",str(W),")")
        return 0

    elif (y<0 or y>=H): 
        print("y is out of bound (H=",str(H),")")
        return 0

    pix=hsi[y][x][:]
    f=np.linspace(400,1000,B)

    return pix,f,[x,y],B


def pixels_species(image_id,specie):
    
    """"Renvoie un array contenant la liste des points d'une image correspondant à l'espèce "specie",
    selon le fichier df_pixel.
    Arguments :
    - image_id (str): le nom de l'image dont on veut afficher la position des pixels
    - specie (str) : le nom de l'espèce dont on veut afficher la position des pixels
    Output :
    - Positions (array): array contenant n coordonnées correspondant aux positions de chaque pixels correspondant à l'espèce "specie",
    dans l'image "image_id".  
    """
        
    gt=GT("df_pixel")
    gt=gt[gt["plotid"] == image_id]
    
    if specie not in gt["specie"].values : 
        return 0
    
    gt = gt[gt["specie"] == specie]
    
    Positions = gt[["row", "col"]].to_numpy()
    
    return Positions


def nb_pixel(tree_id):
    
    gt=GT("df_tree")
    gt=gt[gt["treeid"] == tree_id]
    
    nb_pixel=gt["npix"].values[0]
    
    return nb_pixel


#%% Algorithmes naïfs

def pixel_spectre_moy(spe_pos,hsi):
    
    """"Renvoie la courbe moyenne de chacun des pixels de spe_pos, ainsi que l'erreur à la moyenne à 95%.
    Arguments :
    - spe_pos (array): array contenant la position de chacun des pixels d'une espèce correspondant à l'image du fichier hsi
    - hsi (array): l'array de l'image liée à la position des pixels
    Output :
    - Sp_mean (array): renvoie la courbe moyenne de tous les pixels de l'espèce
    - Sp_std (array) : renvoie l'incertitude à 95% 
    """
    
    rows = spe_pos[:, 0].astype(int)
    cols = spe_pos[:, 1].astype(int)
    
    spectra = hsi[rows, cols, :]
    
    sp_mean = np.mean(spectra, axis=0)
    stderr = sem(spectra, axis=0)
    t_val = t.ppf(0.975, df=len(spe_pos) - 1)
    sp_std = t_val * stderr  # incertitude à 95%
    
    return sp_mean, sp_std


def species_spectre_moy(image_id,hsi):
    
    """"Renvoie un dictionnaire contenant toutes les espèces contenues dans une image (en clé),
    et leurs courbes spectrale (avec l'incertitude) en objet.
    Arguments :
    - image_id (str) : nom de l'image dont on veut la liste des espèces
    - hsi (array) : pixels de l'image
    Output : 
    - d_Spe (dico) : dictionnaire contenant les tuples [courbe moyenne, incertitude]
    - f (array) : abscisses des longueurs d'onde 
    """
    
    H, W, B=hsi.shape
    f=np.linspace(400,1000,B)
    d_Spe={}
    
    for Spe in d_spec :
        Positions=pixels_species(image_id,Spe)
        if type(Positions)!= int:
            Sp_mean,Sp_std=pixel_spectre_moy(Positions,hsi)
            d_Spe[Spe]=[Sp_mean,Sp_std]
            
    return d_Spe,f


def spectre_norm_dico(d_Spe,sp_pixel):
    
    """Renvoie l'espèce avec laquellle un pixel se rapproche le plus, et renvoie également la distance avec cette espèce.
    Arguments :
    - d_Spe (dico) : dictionnaire des espèces et leur moyenne
    - sp_pixel (array) : pixel à tester
    Output :
    - Id (str) : Espèce avec laquelle se rapproche le plus le pixel
    - N (float) : distance avec l'espèce
    - Cla (array) : liste de tuples ["Espèce", distance]
    """
    
    Id=None
    N=m.inf
    
    for Spe in d_Spe:
        moy=d_Spe[Spe][0]
        Norm=np.linalg.norm(moy-sp_pixel)
        if Norm<N :
            Id=Spe
            N=Norm
        
    return Id,N,Cla


#%% Machine Learning

def train_img(image_id,LiDAR=False):
    
    """Détermine les données d'entraînement d'une image, c'est à dire une liste x avec le spectre de chaque pixel,
    et une y avec le label de chaque pixel (ie à quelle espèce correspond ce pixel). L'ordre est donc important
    Arguments :
    - image_id (str) : l'image sur laquelle on veut s'entraîner
    Outputs :
    - X_train (array) : array de (160,n) où n est le nombre de pixels dont on a l'identité dans une image, et 160 correspond donc au spectre d'une image
    - Y_train (array) : l'identité associée à chacun des pixels
    """
    
    gt=GT("df_pixel")
    gt=gt[gt["plotid"]==image_id]
    hsi=HSI(image_id)
    
    row = gt["row"].values
    col = gt["col"].values
    species = gt["specie"].values
    
    pix = hsi[row, col, :]
    
    if LiDAR:
            li = LiD(image_id).to_numpy()
            width = len(li[li[:, 0] == 0])
            
            indices = row * width + col + 1
            lidar_f = li[indices, 2:]
            
            valid = ~np.all(lidar_f == 0, axis=1)
            X = np.hstack([pix[valid], lidar_f[valid]])
            Y = species[valid]
    else:
        X = pix
        Y = species
        
    return X, Y


def train_All_img(LiDAR=False):
    
    image_id=["1b","2","3","3b","4"]
    x,y=train_img("1",LiDAR=LiDAR)

    for im in image_id:
        X_train,Y_train=train_img(im,LiDAR=LiDAR)
        x=np.concatenate((x,X_train),axis=0)
        y=np.concatenate((y,Y_train),axis=None)
 
    return x,y


def train_MLA(image_id=False,LiDAR=False,resample=False,scaler=False,samp_size=0.1):
    
    if image_id :
        X_train,Y_train=train_img(image_id,LiDAR=LiDAR)
    else :
        X_train,Y_train=train_All_img(LiDAR=LiDAR)

    if resample :
        smote = SMOTE(random_state=42)
        X_train, Y_train = smote.fit_resample(X_train, Y_train)
    
    X_train, x_test, Y_train, y_test = train_test_split(X_train, Y_train, test_size=samp_size,random_state=42)
    
    if scaler :
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        x_test = scaler.transform(x_test)
    
    return X_train, x_test, Y_train, y_test


def MLA(algo,X_train, x_test, Y_train, y_test, show=True):
    
    le=LabelEncoder()
    Y_train_enc = le.fit_transform(Y_train)
    y_test_enc = le.transform(y_test)
    
    if algo=="KNN":
        param_grid = {
            'n_neighbors': np.arange(1, 21)
        }
        Alg=KNeighborsClassifier()
        
    elif algo=="RdF":
        param_grid = {
        'n_estimators': [ 300],
        'max_depth': [ 30], 
        'min_samples_split': [2], 
        'min_samples_leaf': [1], 
        'max_features': ['sqrt'],
        'class_weight': [ 'balanced_subsample']
        }
        Alg=RandomForestClassifier()
    
    elif algo=="SVM":
        param_grid = {
            'C': [0.1],
            'max_iter':[10000]
        }
        Alg=LinearSVC()
        
    elif algo == "XGB":
        param_grid = {
            'n_estimators': [300],
            'max_depth': [30],
            'learning_rate': [0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }
        Alg = XGBClassifier(eval_metric='mlogloss')
    
    else : return 0
    
    grid_search = GridSearchCV(Alg, param_grid, cv=2, scoring=make_scorer(f1_score, average='macro'), n_jobs=-1)
    grid_search.fit(X_train, Y_train_enc)

    y_pred = grid_search.predict(x_test)
    y_pred = le.inverse_transform(y_pred)
    y_test_dec = le.inverse_transform(y_test_enc)

    accuracy,cla_report,C_matrix=accuracy_score(y_test_dec, y_pred),classification_report(y_test_dec, y_pred,zero_division=0),confusion_matrix(y_test_dec, y_pred)
    
    if show:
        print("\n\tAlgorithme  : ",algo,"\n")
        print("Meilleurs paramètres pour ",algo,grid_search.best_params_)
        show_res(accuracy,cla_report,C_matrix)
        
    return accuracy,cla_report,C_matrix