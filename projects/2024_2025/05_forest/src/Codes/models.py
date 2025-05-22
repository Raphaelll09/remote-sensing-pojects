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
    train_img()        : Préparation i, j pour une image
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

from joblib import dump, load

from imblearn.over_sampling import SMOTE

dico_Alg={
    "KNN":KNeighborsClassifier(n_neighbors=3),
    "RdF":RandomForestClassifier(n_estimators=300, max_depth=30, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', class_weight='balanced_subsample'),
    "XGB":XGBClassifier(eval_metric='mlogloss',n_estimators=300, max_depth=30, learning_rate=0.1,subsample=0.8,colsample_bytree=0.8),
    "SVM":LinearSVC(C=0.1,max_iter=10000),
    }

#%% Traitement de données

def pixel(image_id: str, i: int, j: int, show=False):
    
    """
    Extract the spectrum of a pixel at coordinates (i, j) from an HSI image.

    Args:
        image_id (str): Image identifier (e.g., "1", "1b").
        i (int): Row index of the pixel.
        j (int): Column index of the pixel.
        show (bool): If True, plot the spectrum for visualization.

    Returns:
        Tuple[np.ndarray, np.ndarray, int]:
            - spectrum (np.ndarray): Spectral values of the pixel (length B).
            - wavelengths (np.ndarray): Wavelengths corresponding to each band.
            - B (int): Number of spectral bands.
    """
    
    hsi=HSI(image_id)
    H, W, B=hsi.shape #Get the shape of the image

    if j<0 or j>=W: 
        print("i is out of bound (W=",str(W),")")
        return 0

    elif (i<0 or i>=H): 
        print("j is out of bound (H=",str(H),")")
        return 0

    pix=hsi[i][j][:]
    f=np.linspace(400,1000,B)

    if show:
        sh.show_spectre_pixel(pix,f,[i,j],B)
    
    return pix,f,B

def pixels_specie(image_id: str, specie: str):
    
    """
    Get the pixel coordinates of all occurrences of a given species in an image.

    Args:
        image_id (str): Plot identifier (e.g., "1b", "2").
        specie (str): Species code (e.g., "PIAB", "FASY").

    Returns:
        np.ndarray: Array of shape (n_pixels, 2) containing [row, col] positions,
                    or 0 if the species is not present.
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
    Return the number of pixels belonging to a given tree crown.

    Args:
        tree_id (str): Unique identifier of a tree in df_tree.csv.

    Returns:
        int: Number of pixels (field "npix") associated with that tree.
    """
    
    gt=GT("df_tree")
    gt=gt[gt["treeid"] == tree_id]
    
    nb_pixel=gt["npix"].values[0]
    
    return nb_pixel

#%% Algorithmes naïfs

def pixel_spectre_moy(image_id: str, specie: str, normalize=False, std=False, show=False):
    
    """
    Compute the mean spectrum and (optionally) 95% confidence interval for a species in an image.

    Args:
        image_id (str): Plot identifier.
        specie (str): Species code (e.g., "PIAB").
        normalize (bool): If True, normalize each pixel spectrum before averaging.
        std (bool): If True, compute 95% confidence interval (using standard error).
        show (bool): If True, plot the mean spectrum with confidence interval.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - sp_mean (np.ndarray): Mean spectrum (length B).
            - sp_std (np.ndarray or None): 95% CI values (length B) if std=True, else None.
            - f (np.ndarray): Wavelengths for each band (400–1000 nm).
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
        sh.show_spectre_moy(specie, sp_mean, sp_std, f, std=std)
    
    return sp_mean, sp_std, f

def species_spectre_moy(image_id: str,normalize=False):
    
    """
    Compute the mean spectrum and 95% confidence interval for each species in one image.

    Args:
        image_id (str): Plot identifier (e.g. "1b", "3").
        normalize (bool): If True, normalize each pixel spectrum by its max value.

    Returns:
        Tuple[np.ndarray, dict]:
            - wavelengths (np.ndarray): 1D array of wavelengths (400–1000 nm, length B).
            - d_spe (dict): Mapping species → [mean_spectrum (np.ndarray), ci95 (np.ndarray)].
    """
    
    d_spe={}
    
    gt=GT("df_pixel")
    gt=gt[gt["plotid"] == image_id]
    
    for Spe in gt["specie"]:
        d_spe[Spe]=None
    
    for Spe in d_spe.keys() :
        Sp_mean,Sp_std,f=pixel_spectre_moy(image_id,Spe,normalize=normalize,std=True)
        d_spe[Spe]=[Sp_mean,Sp_std]
    
    return f, d_spe

def spectre_norm_dico(d_Spe: dict, sp_pixel: np.ndarray):
    
    """
    Classify a pixel by the nearest species mean spectrum (Euclidean distance).

    Args:
        d_spe (dict): Mapping {species: [mean_spectrum, std_dev]}.
        sp_pixel (np.ndarray): Pixel spectrum (normalized if needed).

    Returns:
        Tuple[str, float, dict]: 
            - nearest_species (str): Closest species code.
            - distance (float): Euclidean distance to that species' mean.
            - all_distances (dict): Distances to all species {species: distance}.
    """
    
    pix=sp_pixel/np.max(sp_pixel)
    Id=None
    Norm_min=m.inf
    Norm_dico={}
    
    for Spe in d_Spe:

        moy=d_Spe[Spe][0]
        Norm=np.linalg.norm(moy-pix)
        
        Norm_dico[Spe]=float(Norm)
        
        if Norm<Norm_min :
            Id=Spe
            Norm_min=Norm
        
    return Id, Norm_min, Norm_dico

def test_pixel_norm(image_id: str,row: int,col: int, show=False):
    
    """
    Classify a single pixel via Euclidean distance to species’ mean spectra.

    Args:
        image_id (str): Plot identifier.
        row (int): Pixel row index.
        col (int): Pixel column index.
        show (bool): If True, print the predicted species and distance.

    Returns:
        None
    """
    
    hsi=HSI(image_id)
    f, d_Spe=species_spectre_moy(image_id)
    
    pixel=hsi[row][col]
    
    Id,Norm_min,Norms=spectre_norm_dico(d_Spe,pixel)
    
    if show:
        print("Le pixel de coordonnées ",str((row,col))," correspond probablement à l'espèce :",Id,",avec une norme de :",str(Norm_min))

def test_norm_image(image_id: str, score=None):
    
    """
    Evaluate baseline distance‐to‐mean classifier over all labeled pixels in one image.

    Args:
        image_id (str): Plot identifier.
        score (dict, optional): Running tallies {species: [total, correct, accuracy]}.

    Returns:
        dict: Updated scores per species: [n_total, n_correct, accuracy].
    """
    
    gt=GT("df_pixel")
    gt=gt[gt["plotid"] == image_id]
    
    f, d_spe=species_spectre_moy(image_id)
    row,col,specie=gt["row"],gt["col"],gt["specie"]
    
    if score==None:
        score={k:[0,0,0] for k in d_spe.keys()}
    
    for k in range(len(specie)):
        
        i,j,spe=row.iloc[k],col.iloc[k],specie.iloc[k]
        pix,f,B=pixel(image_id,i,j)
        Id,Norm_min,Norms=spectre_norm_dico(d_spe,pix)

        score[spe][0]+=1
        if spe==Id: score[spe][1]+=1
    
    for Spe in score.keys() :
        if score[Spe][1]:
            score[Spe][2]=round(score[Spe][1]/score[Spe][0],3)
    
    return score
        
def test_norm_All():
    
    image_id=["1","1b","2","3","3b","4"]
    score={k : [0,0,0] for k in keys}
    
    for img in image_id:
        test_norm_image(img,score)
        
    return score
        
#%% Machine Learning

def train_img(image_id: str,LiDAR=False, pos=False):
    
    """
    Prepare training data from one image.

    Args:
        image_id (str): Plot identifier.
        LiDAR (bool): If True, include LiDAR features concatenated to HSI.
        pos (bool): If True, also return pixel positions (row, col).

    Returns:
        If pos=False:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - X (samples x features): Feature matrix (HSI bands [+ LiDAR]).
                - Y (samples,): Species labels.
                - G (samples,): Tree IDs (for grouping).
                - f (samples,): Family labels.
        If pos=True, additionally returns:
            row (np.ndarray), col (np.ndarray): Pixel coordinates.
    """
    
    gt=GT("df_pixel")
    gt=gt[gt["plotid"]==image_id]
    hsi=HSI(image_id)
    
    row = gt["row"].values
    col = gt["col"].values
    specie = gt["specie"]
    treeids = gt["treeid"].values
    family = gt["family"].values
    
    pix = hsi[row, col, :]
    
    if LiDAR:
            li = LiD(image_id).to_numpy()
            width = len(li[li[:, 0] == 0])
            
            indices = row * width + col + 1
            lidar_f = li[indices, 2:]
            
            valid = ~np.all(lidar_f == 0, axis=1)
            x = np.hstack([pix[valid], lidar_f[valid]])
            y = specie[valid]
            G = treeids[valid]
            f = family[valid]
            row=row[valid]
            col=col[valid]
        
    else:
        x = pix
        y = specie
        G = treeids
        f = family
    
    if pos:
        return row,col,x,y
    
    return x, y, G, f

def train_full_img(image_id: str):
    
    """
    Extract all ground‐truth & LiDAR points from one plot to build a combined feature array.

    Args:
        image_id (str): Plot identifier.

    Returns:
        np.ndarray: Array of shape (n_points, 2 + B + 24):
            [row, col, HSI_bands…, LiDAR_features…] for every valid LiDAR pixel.
    """
    
    hsi=HSI(image_id)
    li = LiD(image_id).to_numpy()
    

    x=li[:,0]
    y=li[:,1]
    lidar_f = li[:,2:]
    valid = ~np.all(lidar_f == 0, axis=1)
    x=x[valid]
    y=y[valid]
    lidar_f=lidar_f[valid]

    def pix(hsi,x,y,lidar_f):
        pix=[]
        for i in range(len(x)):
            X,Y=int(x[i]),int(y[i])
            spectre=hsi[X,Y,:]
            lid=lidar_f[i,:]
            som=np.concatenate(([X,Y],spectre,lid))
            pix.append(som)
        
        return np.array(pix)
    
    pix=pix(hsi,x,y,lidar_f)

    return pix
    

def train_All_img(LiDAR=False):
    
    """
    Prepare training data by concatenating multiple images.

    Args:
        LiDAR (bool): If True, include LiDAR features.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            Concatenated (X, Y, G, f) across all plots.
    """
    
    image_id=["1b","2","3","3b","4","Premol"]
    x,y,g,f=train_img("1",LiDAR=LiDAR)

    for im in image_id:
        X,Y,G,F=train_img(im,LiDAR=LiDAR)

        x=np.concatenate((x,X),axis=0)
        y=np.concatenate((y,Y),axis=None)
        g=np.concatenate((g,G),axis=None)
        f=np.concatenate((f,F),axis=None)
 
    return x, y, g, f

def train_MLA(image_id=False, LiDAR=False, split_by_tree=False, resample=False,scaler=False,samp_size=0.1):

    """
    Prepare train/test split for machine learning experiments.

    Args:
        image_id (str, optional): Specific image to use; if False, use all images.
        LiDAR (bool): If True, include LiDAR features.
        split_by_tree (bool): If True, ensure no tree's pixels appear in both train and test.
        resample (bool): If True, apply SMOTE oversampling to balance classes:contentReference[oaicite:0]{index=0}.
        scaler (bool): If True, apply standard scaling (Z-score).
        samp_size (float): Proportion of data for the test set (default 0.1).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            (X_train, X_test, Y_train, Y_test).
    """

    if image_id :
        X_train,Y_train,G_train,f_train=train_img(image_id,LiDAR=LiDAR)
    else :
        X_train,Y_train,G_train,f_train=train_All_img(LiDAR=LiDAR)

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

def MLA_model(algo: str, X_train: np.ndarray, Y_train: np.ndarray, Grid_S=True):

    """
    Train a classifier (optionally with GridSearch for hyperparameters).

    Args:
        algo (str): Algorithm key ("KNN", "RdF", "SVM", "XGB").
        X_train (np.ndarray): Training feature matrix.
        Y_train (np.ndarray): Training labels (integer-encoded).
        Grid_S (bool): If True, perform 3-fold GridSearchCV for parameter tuning.

    Returns:
        sklearn estimator: Trained classifier (GridSearchCV-wrapped if Grid_S=True).
    """
    
    if Grid_S:
        
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
            Alg=XGBClassifier(eval_metric='mlogloss')
                
        else : raise ValueError(f"Unknown algorithm '{algo}' for MLA_model.")
        
        Alg=GridSearchCV(Alg, param_grid, cv=3, scoring=make_scorer(f1_score, average='macro'), n_jobs=-1)
        
    else: Alg=dico_Alg[algo]

    Alg.fit(X_train,Y_train)
        
    return Alg

def MLA(algo: str, X_train: np.ndarray, x_test: np.ndarray, Y_train: np.ndarray, y_test: np.ndarray, show=True):
    
    """
    Train and evaluate a classifier, returning accuracy and reports.

    Args:
        algo (str): Algorithm key ("KNN", "RdF", "SVM", "XGB").
        X_train, X_test (np.ndarray): Training and test features.
        Y_train, Y_test (np.ndarray): Training and test labels.
        show (bool): If True, print scores and confusion matrix.

    Returns:
        Tuple[float, str, dict, np.ndarray]:
            - accuracy: Overall accuracy.
            - report: Text classification report.
            - report_dict: Classification report as dict.
            - conf_matrix: Confusion matrix array.
    """
    
    le=LabelEncoder()
    Y_train_enc = le.fit_transform(Y_train)
    y_test_enc = le.transform(y_test)

    grid_search=MLA_model(algo, X_train, Y_train_enc)
    
    y_pred = grid_search.predict(x_test)
    y_pred = le.inverse_transform(y_pred)
    y_test_dec = le.inverse_transform(y_test_enc)

    accuracy,report,report_dico,C_matrix=accuracy_score(y_test_dec, y_pred),classification_report(y_test_dec, y_pred,zero_division=0),classification_report(y_test_dec, y_pred,zero_division=0,output_dict=True),confusion_matrix(y_test_dec, y_pred)
    
    if show:
        print(f"\n=== Algorithm: {algo} ===")
        print("Best parameters:",algo,grid_search.best_params_)
        sh.show_res_Alg(accuracy,report,C_matrix)
    
    return accuracy,report,report_dico,C_matrix

def algo_comparison(Algos=["KNN","RdF","XGB","SVM"],LiD_nat=True, scaler=False,LiDAR=False,show=False,split_by_tree=False):
    
    """
    Compare multiple classifiers with and without preprocessing (e.g., scaling or LiDAR).

    Args:
        Algos (list): Algorithm keys to compare.
        LiD_nat (bool): If True, baseline uses LiDAR by default; else uses only HSI.
        scaler (bool): If True, apply standard scaling in the second comparison.
        LiDAR (bool): If True, include LiDAR features in the second comparison.
        show (bool): If True, print results for each model.
        split_by_tree (bool): If True, split data by tree ID groups.

    Returns:
        Tuple[dict, dict, dict]: Dictionaries of accuracy, macro-F1, and weighted-F1 for each model,
                                 with and without the specified preprocessing.
    """
    
    if scaler:
        comp="Scaler"
    elif LiDAR :
        comp="LiDAR"
    elif split_by_tree:
        comp="split_by_tree"
    else :
        raise ValueError("Specify a comparison mode (scaler or LiDAR or split_by_tree).")
    
    A1,A2,M1,M2,W1,W2=[],[],[],[],[],[]
    
    if LiD_nat :
        LiD=True
    else : 
        LiD=LiDAR
    
    X_train, x_test, Y_train, y_test=train_MLA(LiDAR=LiD_nat)
    X_train_p, x_test_p, Y_train_p, y_test_p=train_MLA(scaler=scaler,LiDAR=LiD,split_by_tree=split_by_tree)
    
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
    'Model': Algos * 2,
    'Accuracy': A,  # Sans Scaler puis Avec Scaler
    comp: ['without '+comp]*len(Algos) + ['With '+comp]*len(Algos)
    }
    
    data_macro = {
    'Model': Algos * 2,
    'Macro avg': M,  # Sans Scaler puis Avec Scaler
    comp: ['without '+comp]*len(Algos) + ['With '+comp]*len(Algos)
    }
    
    data_weighted = {
    'Model': Algos * 2,
    'Weighted avg': W,  # Sans Scaler puis Avec Scaler
    comp: ['without '+comp]*len(Algos) + ['With '+comp]*len(Algos)
    }
    
    return data_accuracy, data_macro, data_weighted

def train_n_save_Alg(algo: str, filename: str):
    
    """
    Train a classifier on all data and save it to disk as a .joblib file.

    Args:
        algo (str): Algorithm key (e.g., "RdF").
        filename (str): Output filename (without extension).

    Returns:
        None
    """
    
    X_train, x_test, Y_train, y_test = train_MLA(LiDAR=True)
    
    le=LabelEncoder()
    Y_train_enc = le.fit_transform(Y_train)
    Alg=MLA_model(algo, X_train, Y_train_enc, Grid_S=False)
    
    dump(le, f"{filename}_le.joblib")
    dump(Alg, f"{filename}.joblib")
    print(f"Model {algo} saved to {filename}.joblib")
    
def predict_img(image_id: str, filename: str, gt=False):
    
    """
    Apply a saved model to predict species on an image.

    Args:
        image_id (str): Plot identifier.
        filename (str): Saved model filename (without .joblib).
        gt (bool): If True, predict only on ground-truth pixels.

    Returns:
        Tuple: (row_indices, col_indices, predicted_species_array).
    """
    
    Alg=load(f"{filename}.joblib")
    le=load(f"{filename}_le.joblib")
    
    if gt:
        row,col,x,y=train_img(image_id,LiDAR=True,pos=True)
    else:
        pix=train_full_img(image_id)
        row,col,x=pix[:,0],pix[:,1],pix[:,2:]
        
    pred_enc=Alg.predict(x)
    pred=le.inverse_transform(pred_enc)
    
    return row,col,pred
    
    
#%% Hiérarchie [PROTOTYPE]

def major_label(Y_train):
    
    """
    Convert species labels to 'majority' vs 'Other' for hierarchical training.

    Args:
        Y_train (np.ndarray): Original species labels.

    Returns:
        np.ndarray: Labels with minority species replaced by 'Other'.
    """
    
    return np.array([y if y in majority else 'Other' for y in Y_train])
    
def train_MLA_h(LiDAR=False, samp_size=0.1):
    
    """
    Prepare train/test split for hierarchical classification (family first).

    Args:
        LiDAR (bool): If True, include LiDAR features.
        samp_size (float): Test set proportion.

    Returns:
        Tuple: (X_train, Y_train, Family_train, X_test, Y_test, Family_test).
    """
    
    X,Y,G,f=train_All_img(LiDAR=LiDAR)
    
    idx=np.arange(len(X))
    train_idx,test_idx = train_test_split(idx, test_size=samp_size,random_state=42, shuffle=True)
    
    X_train, x_test = X[train_idx], X[test_idx]
    Y_train, y_test = Y[train_idx], Y[test_idx]
    F_train, f_test = f[train_idx], f[test_idx]
    
    return X_train, Y_train, F_train, x_test, y_test, f_test

def hierarchical_MLA(algo, LiDAR=False):
    
    """
    Train and evaluate a two-stage hierarchical classifier (family → species).

    First, predict tree family; then predict species within each family.

    Args:
        algo (str): Base algorithm key for classifiers.
        LiDAR (bool): If True, include LiDAR features.

    Returns:
        dict: Dictionary containing true labels, predictions at each stage, and final species predictions.
    """
    
    X_train, Y_train, F_train, x_test, y_test, f_test=train_MLA_h(LiDAR=LiDAR)
    
    le_f=LabelEncoder()
    
    F_train_enc = le_f.fit_transform(F_train)
    f_test_enc = le_f.fit_transform(f_test)
    Alg_f=MLA_model(algo, X_train, F_train_enc)
    f_pred_enc=Alg_f.predict(x_test)
    
    f_pred=le_f.inverse_transform(f_pred_enc)
    F_train=le_f.inverse_transform(F_train_enc)
    
    def train_per_family(algo, X_train, Y_train, F_train, x_test, y_test, f_pred):
        
        major_pred = np.empty(len(x_test), dtype=object)
        
        for fam in ["CONI", "BROA"]:
            
            le_maj = LabelEncoder()
            
            mask_train = F_train == fam
            mask_test = f_pred == fam

            if not np.any(mask_train): continue

            Y_fam = major_label(Y_train[mask_train])
            
            Y_fam_enc = le_maj.fit_transform(Y_fam)
            
            Alg_maj = MLA_model(algo, X_train[mask_train], Y_fam_enc)
            Y_pred_enc = Alg_maj.predict(x_test[mask_test])
            
            Y_pred = le_maj.inverse_transform(Y_pred_enc)

            major_pred[mask_test] = Y_pred

        return major_pred
    
    major_pred=train_per_family(algo, X_train, Y_train, F_train, x_test, y_test, f_pred)
    final_pred = np.array(major_pred)
    
    mask_other = (major_pred == "Other")

    if np.any(mask_other):
        
        X_other = x_test[mask_other]
        f_other = f_pred[mask_other]
        minor_pred=np.empty(len(X_other), dtype=object)
        
        for fam in ["CONI", "BROA"]:
            mask_tr = (F_train == fam) & np.isin(Y_train, minority)
            mask_te = f_other == fam

            if not np.any(mask_tr): continue

            le_min = LabelEncoder()
            Y_min_enc = le_min.fit_transform(Y_train[mask_tr])

            clf_min = MLA_model(algo, X_train[mask_tr], Y_min_enc)
            pred_enc = clf_min.predict(X_other[mask_te])
            pred = le_min.inverse_transform(pred_enc)

            minor_pred[mask_te] = pred
        
        final_pred[mask_other] = minor_pred
        
    
    print("\n=== Final Hierarchical Classification Report ===")    
    print(classification_report(y_test, final_pred, zero_division=0))
    print("Overall Accuracy:", accuracy_score(y_test, final_pred))

    accuracy,report,report_dico,C_matrix=accuracy_score(y_test, final_pred),classification_report(y_test, final_pred,zero_division=0),classification_report(y_test, final_pred,zero_division=0,output_dict=True),confusion_matrix(y_test, final_pred)

    return {
        "true": y_test,
        "pred": final_pred,
        "fam_pred": f_pred,
        "major_pred": major_pred,
        "minor_pred": final_pred[mask_other],
        "mask_other": mask_other,
        "X_test": x_test
        }   
    


