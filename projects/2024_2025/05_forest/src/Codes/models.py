"""
models.py

Module principal pour :
- Le traitement des données hyperspectrales et LiDAR
- Le calcul des spectres moyens des espèces
- L'entraînement de modèles de classification supervisée
- La comparaison de plusieurs algorithmes de machine learning

Fonctionnalités :
- Extraction de pixels d'entraînement (HSI seul ou fusion HSI+LiDAR)
- Calcul de spectres moyens et distance euclidienne
- Implémentation de modèles : KNN, Random Forest, SVM, XGBoost
- Utilisation de SMOTE pour le rééquilibrage
- Pipeline complet d’entraînement/test avec normalisation optionnelle

Fonctions principales :
    train_img()        : Préparation X, y pour une image
    train_All_img()    : Préparation multi-images
    train_MLA()        : Split train/test avec options (scaling, LiDAR, resampling)
    MLA()              : Entraînement + GridSearch + évaluation
    algo_comparison()  : Comparaison multi-algorithmes avec/ sans prétraitement

Projet : Classification d’espèces d’arbres par télédétection (HSI + LiDAR)
"""
#%% Librairies
import show as sh
from data import *
from show import *

import math as m
import numpy as np

from scipy.stats import sem, t

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler


from imblearn.over_sampling import SMOTE


#%% Traitement de données

def pixel(image_id: str, x: int, y:int, show=False):
    
    """
    Extrait le spectre d'un pixel donné dans une image hyperspectrale.

    Args:
        image_id (str): Identifiant de l'image à interroger (ex : "1", "1b").
        x (int): Coordonnée en colonne (axe horizontal) du pixel.
        y (int): Coordonnée en ligne (axe vertical) du pixel.

    Returns:
        Tuple[np.ndarray, np.ndarray, list[int], int]:
            - spectre (B,) : Valeurs spectrales du pixel.
            - f (B,) : Longueurs d'onde associées (400–1000 nm).
            - [x, y] : Coordonnées du pixel.
            - B (int) : Nombre total de bandes spectrales.
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

    if show:
        show_spectre_pixel(pix,f,[x,y],B)
    
    return pix,f,B


def pixels_specie(image_id: str, specie: str):
    
    """
    Récupère toutes les coordonnées des pixels d'une espèce donnée dans une image.

    Args:
        image_id (str): Identifiant de la parcelle (ex : "1b", "2").
        specie (str): Code de l'espèce recherchée (ex : "PIAB", "FASY").

    Returns:
        np.ndarray | int:
            - Tableau (n, 2) des positions [row, col] des pixels associés à l'espèce.
            - Retourne 0 si l'espèce n'existe pas dans cette image.
    """
        
    gt=GT("df_pixel")
    gt=gt[gt["plotid"] == image_id]
    
    if specie not in gt["specie"].values : 
        return 0
    
    gt = gt[gt["specie"] == specie]
    
    Positions = gt[["row", "col"]].to_numpy()
    
    return Positions


def nb_pixel(tree_id: str):
    
    """
    Retourne le nombre de pixels correspondant à la couronne d’un arbre donné.

    Args:
        tree_id (int): Identifiant unique d’un arbre dans df_tree.csv.

    Returns:
        int: Nombre de pixels associés à l’arbre (champ "npix").
    """
    
    gt=GT("df_tree")
    gt=gt[gt["treeid"] == tree_id]
    
    nb_pixel=gt["npix"].values[0]
    
    return nb_pixel


#%% Algorithmes naïfs

def pixel_spectre_moy(image_id: str, specie: str, normalize=False, std=False, show=False):
    
    """
    Calcule le spectre moyen et son incertitude (95%) pour un ensemble de pixels.

    Args:
        spe_pos (np.ndarray): Tableau (n, 2) de positions [row, col] des pixels.
        hsi (np.ndarray): Image hyperspectrale (H, W, B).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Sp_mean (B,) : Spectre moyen.
            - Sp_std (B,) : Intervalle de confiance à 95% (basé sur erreur standard).
    """
    spe_pos=pixels_specie(image_id,specie)
    hsi=HSI(image_id)
    H,W,B=hsi.shape
    
    rows = spe_pos[:, 0].astype(int)
    cols = spe_pos[:, 1].astype(int)
    
    spectra = hsi[rows, cols, :]
    f=np.linspace(400,1000,B)
    
    if normalize:
        max_vals = np.max(spectra, axis=1, keepdims=True)
        max_vals[max_vals == 0] = 1  
        spectra = spectra / max_vals
    
    sp_mean = np.mean(spectra, axis=0)
    
    if std:
        stderr = sem(spectra, axis=0)
        t_val = t.ppf(0.975, df=len(spe_pos) - 1)
        sp_std = t_val * stderr # incertitude à 95%
    else :
        sp_std=None
    
    if show:
        sh.show_spectre_moy(specie, sp_mean,sp_std, f, std=std)
    
    return sp_mean, sp_std, f


def species_spectre_moy(image_id: str):
    
    """
    Calcule le spectre moyen et l'incertitude pour chaque espèce dans une image.

    Args:
        image_id (str): Identifiant de la parcelle.
        hsi (np.ndarray): Image hyperspectrale 3D (H, W, B).

    Returns:
        Tuple[dict, np.ndarray]:
            - d_Spe : Dictionnaire {espèce: (spectre moyen, incertitude)}.
            - f : Tableau des longueurs d’onde (160 valeurs de 400 à 1000 nm).
    """
    hsi=HSI(image_id)
    
    H, W, B=hsi.shape
    f=np.linspace(400,1000,B)
    d_Spe={}
    
    for Spe in d_spec :
        Positions=pixels_species(image_id,Spe)
        if type(Positions)!= int:
            Sp_mean,Sp_std=pixel_spectre_moy(Positions,hsi)
            d_Spe[Spe]=[Sp_mean,Sp_std]
            
    return d_Spe,f


def spectre_norm_dico(d_Spe: dict, sp_pixel: np.ndarray):
    
    """
    Compare un spectre de pixel à tous les spectres moyens d’espèces, et retourne la plus proche.

    Args:
        d_Spe (dict): Dictionnaire {espèce: (spectre moyen, incertitude)}.
        sp_pixel (np.ndarray): Spectre à tester (160,).

    Returns:
        Tuple[str, float, List[Tuple[str, float]]]:
            - Id : Nom de l'espèce la plus proche.
            - N : Distance euclidienne minimale.
            - Cla : Liste [(espèce, distance)] triée (optionnel, à activer).
    """
    
    Id=None
    N=m.inf
    
    for Spe in d_Spe:
        moy=d_Spe[Spe][0]
        Norm=np.linalg.norm(moy-sp_pixel)
        if Norm<N :
            Id=Spe
            N=Norm
        
    return Id,N


#%% Machine Learning

def train_img(image_id: str,LiDAR=False):
    
    """
    Prépare les données d'entraînement pour une image donnée (HSI seul ou HSI + LiDAR).

    Args:
        image_id (str): Identifiant de l'image (ex: "1", "1b", "3").
        LiDAR (bool): Si True, les données LiDAR sont concaténées aux spectres HSI.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - X_train : Matrice (n_samples, n_features), avec 160 bandes + 24 si LiDAR.
            - Y_train : Tableau (n_samples,) des classes (espèces d'arbres).
    """
    
    gt=GT("df_pixel")
    gt=gt[gt["plotid"]==image_id]
    hsi=HSI(image_id)
    
    row = gt["row"].values
    col = gt["col"].values
    species = gt["specie"].values
    treeids = gt["treeid"].values
    
    pix = hsi[row, col, :]
    
    if LiDAR:
            li = LiD(image_id).to_numpy()
            width = len(li[li[:, 0] == 0])
            
            indices = row * width + col + 1
            lidar_f = li[indices, 2:]
            
            valid = ~np.all(lidar_f == 0, axis=1)
            X = np.hstack([pix[valid], lidar_f[valid]])
            Y = species[valid]
            G = treeids[valid]
        
    else:
        X = pix
        Y = species
        G=treeids
        
    return X, Y, G


def train_All_img(LiDAR=False):
    
    """Prépare les données d'entraînement sur toutes les images."""
    
    image_id=["1b","2","3","3b","4","Premol"]
    x,y,g=train_img("1",LiDAR=LiDAR)

    for im in image_id:
        X_train,Y_train,G_train=train_img(im,LiDAR=LiDAR)
        x=np.concatenate((x,X_train),axis=0)
        y=np.concatenate((y,Y_train),axis=None)
        g=np.concatenate((g,G_train),axis=None)
 
    return x, y, g


def train_MLA(image_id=False, LiDAR=False, split_by_tree=False, resample=False,scaler=False,samp_size=0.1):

    """
    Prépare les données pour entraînement/test d’un modèle machine learning.

    Args:
        image_id (str, optional): Si spécifié, ne prend que cette image, sinon prend tout.
        LiDAR (bool): Si True, ajoute les données LiDAR aux spectres HSI.
        resample (bool): Si True, applique SMOTE pour équilibrer les classes.
        scaler (bool): Si True, applique une standardisation des données (Z-score).
        samp_size (float): Proportion de l’échantillon de test (0.1 = 10%).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - X_train, x_test, Y_train, y_test : Données scindées pour ML.
    """

    
    if image_id :
        X_train,Y_train,G_train=train_img(image_id,LiDAR=LiDAR)
    else :
        X_train,Y_train,G_train=train_All_img(LiDAR=LiDAR)

    if resample :
        smote = SMOTE(random_state=42)
        X_train, Y_train = smote.fit_resample(X_train, Y_train)
    
    if split_by_tree:
        
        gss = GroupShuffleSplit(n_splits=1, test_size=samp_size, random_state=42)
        train_idx, test_idx = next(gss.split(X_train, Y_train, groups=G_train))
        X_train, x_test = X_train[train_idx], X_train[test_idx]
        Y_train, y_test = Y_train[train_idx], Y_train[test_idx]
    else :
        X_train, x_test, Y_train, y_test = train_test_split(X_train, Y_train, test_size=samp_size,random_state=42)
    
    if scaler :
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        x_test = scaler.transform(x_test)
    
    return X_train, x_test, Y_train, y_test


def MLA(algo: str,X_train: np.ndarray, x_test: np.ndarray, Y_train: np.ndarray, y_test: np.ndarray, show=True):
    
    """
    Entraîne un algorithme de classification, cherche les meilleurs hyperparamètres via GridSearchCV,
    et retourne les résultats de performance.

    Args:
        algo (str): Nom de l'algorithme ("KNN", "RdF", "SVM", "XGB").
        X_train (np.ndarray): Données d’entraînement.
        x_test (np.ndarray): Données de test.
        Y_train (np.ndarray): Labels d’entraînement.
        y_test (np.ndarray): Labels de test.
        show (bool): Si True, affiche les scores, le rapport de classification et la matrice de confusion.

    Returns:
        Tuple[float, str, np.ndarray]:
            - accuracy : Précision globale
            - classification_report : Rapport texte des scores par classe
            - confusion_matrix : Matrice de confusion
    """
    
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
    
    grid_search = GridSearchCV(Alg, param_grid, cv=3, scoring=make_scorer(f1_score, average='macro'), n_jobs=-1)
    grid_search.fit(X_train, Y_train_enc)

    y_pred = grid_search.predict(x_test)
    y_pred = le.inverse_transform(y_pred)
    y_test_dec = le.inverse_transform(y_test_enc)

    accuracy,report,report_dico,C_matrix=accuracy_score(y_test_dec, y_pred),classification_report(y_test_dec, y_pred,zero_division=0),classification_report(y_test_dec, y_pred,zero_division=0,output_dict=True),confusion_matrix(y_test_dec, y_pred)
    
    if show:
        print("\n\tAlgorithme  : ",algo,"\n")
        print("Meilleurs paramètres pour ",algo,grid_search.best_params_)
        sh.show_res_Alg(accuracy,report,C_matrix)
    
    return accuracy,report,report_dico,C_matrix


def algo_comparison(Algos=["KNN","RdF","XGB","SVM"],scaler=False,LiDAR=False,show=False,split_by_tree=False):
    
    """
    Compare plusieurs modèles (KNN, Random Forest, XGBoost, SVM) avec et sans traitement.

    Args:
        Algos (list): Liste des algorithmes à tester.
        scaler (bool): Si True, applique une normalisation standard.
        LiDAR (bool): Si True, inclut les variables LiDAR aux spectres HSI.
        show (bool): Si True, affiche les résultats pour chaque modèle.

    Returns:
        Tuple[dict, dict, dict]:
            - data_accuracy : Scores d’accuracy pour chaque modèle.
            - data_macro : Scores F1 macro-averaged.
            - data_weighted : Scores F1 pondérés par effectif de classe.
    """
    
    if scaler:
        comp="Scaler"
    elif LiDAR :
        comp="LiDAR"
    elif split_by_tree:
        comp="split_by_tree"
    else :
        return 0
    
    A1,A2,M1,M2,W1,W2=[],[],[],[],[],[]
    
    X_train, x_test, Y_train, y_test=train_MLA(LiDAR=True)
    X_train_p, x_test_p, Y_train_p, y_test_p=train_MLA(scaler=scaler,LiDAR=LiDAR,split_by_tree=split_by_tree)
    
    for Alg in Algos :
        
        if show : print("\n\t\tSans "+comp+" :\n ")
        
        a,r,r_d,m=MLA(Alg,X_train, x_test, Y_train, y_test, show=show)
        A1+=[a]
        M1+=[r_d["macro avg"]["f1-score"]]
        W1+=[r_d["weighted avg"]["f1-score"]]
        
        if show : print("\n\t\tAvec "+comp+" :\n ")
    
        a,r,r_d,m=MLA(Alg, X_train_p, x_test_p, Y_train_p, y_test_p, show=show)
        A2+=[a]
        M2+=[r_d["macro avg"]["f1-score"]]
        W2+=[r_d["weighted avg"]["f1-score"]]
    
    A,M,W = A1+A2, M1+M2, W1+W2
    
    data_accuracy = {
    'Modèle': Algos * 2,
    'Accuracy': A,  # Sans Scaler puis Avec Scaler
    comp: ['Sans Scaler']*len(Algos) + ['Avec Scaler']*len(Algos)
    }
    
    data_macro = {
    'Modèle': Algos * 2,
    'Macro avg': M,  # Sans Scaler puis Avec Scaler
    comp: ['Sans Scaler']*len(Algos) + ['Avec Scaler']*len(Algos)
    }
    
    data_weighted = {
    'Modèle': Algos * 2,
    'Weighted avg': W,  # Sans Scaler puis Avec Scaler
    comp: ['Sans Scaler']*len(Algos) + ['Avec Scaler']*len(Algos)
    }
    
    return data_accuracy, data_macro, data_weighted


