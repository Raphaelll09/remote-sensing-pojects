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

keys=["ABAL","ACPS","BEPE","BEsp","COAV","FASY","FREX","PIAB","PICE","PIUN","POTR","SOAR","SOAU"]
d_spec= {k : None for k in keys}


def spectre_pixel(hsi, x, y, i=0):
    
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
    #pixel=savgol_filter(pixel,window_length=5,polyorder=2)
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
    cwd = Path(__file__).resolve().parents[2]
    
    filename_hsi ="data/raw/forest-plot-analysis/data/hi/"+filename+".tif"
    file_hsi = cwd / filename_hsi
    
    hsi = tifffile.TiffFile(file_hsi) #Lit l'image et la traduit en Aray
    hsi = hsi.asarray()
    return hsi

def show_image(image_id,RVB, i=1, ax=None):
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
    hsi=open_hsi(image_id)
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
    ax.set_title(f"Image n°{image_id} RVB")

def show_map(image_id,ax=0,alph=None):
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
    
    hsi=open_hsi(image_id)
    H,W,B=hsi.shape
    
    df = pd.read_csv(file_label) #Lis le fichier csv Pixels
    df = df[df["plotid"] == image_id] #Sélectionne l'id des arbres 
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
        show_image(image_id,RVB,0,ax1)
        show_map(image_id,ax2)
    else:
        fig, ax = plt.subplots()
        show_image(image_id,RVB,0,ax)
        show_map(image_id,ax,0.2)
    
    plt.show()
    
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
        
    df=open_csv(image_id,"df_pixel",tree=True)
    
    if specie not in df["specie"].iloc() : 
        return 0
    
    df = df[df["specie"] == specie]
    
    row,col=df["row"],df["col"]
    
    Positions=np.zeros((len(row),2))
    for i in range(len(Positions)):
        Positions[i]=[row.iloc[i],col.iloc[i]]
    
    return Positions
        
def nb_pixel(image_id,tree_id):
    
    df=open_csv(image_id,"df_tree",tree=True)
    df=df[df["treeid"] == tree_id]
    
    nb_pixel=df["npix"].iloc[0]
    
    return nb_pixel

def sp_pixel_moy(spe_pos,hsi):
    
    """"Renvoie la courbe moyenne de chacun des pixels de spe_pos, ainsi que l'erreur à la moyenne à 95%.
    Arguments :
    - spe_pos (array): array contenant la position de chacun des pixels d'une espèce correspondant à l'image du fichier hsi
    - hsi (array): l'array de l'image liée à la position des pixels
    Output :
    - Sp_mean (array): renvoie la courbe moyenne de tous les pixels de l'espèce
    - Sp_std (array) : renvoie l'incertitude à 95% 
    """
    
    H, W, B=hsi.shape
    Sp_tot=np.zeros((len(spe_pos),B))
    
    for i in range(len(spe_pos)):
        x,y=int(spe_pos[i][1]),int(spe_pos[i][0])
        sp,f=sp_pixel(hsi,x,y)
        Sp_tot[i]=sp
    
    Sp_mean=np.mean(Sp_tot,axis=0)
    stderr_curve = sem(Sp_tot, axis=0)  # erreur standard de la moyenne
    t_val = t.ppf(0.975, df=len(spe_pos) - 1)  # pour 95% de confiance
    Sp_std = t_val * stderr_curve
    
    return Sp_mean, Sp_std

def sp_species_moy(image_id,hsi):
    
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
            Sp_mean,Sp_std=sp_pixel_moy(Positions,hsi)
            d_Spe[Spe]=[Sp_mean,Sp_std]
            
    return d_Spe,f
  
def show_sp_curve(specie, Sp_mean,Sp_std,f):
    
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

def show_sp_dico(d_Spe,f,ax, i=None):
    
    """"Permet d'afficher toutes les courbes d'une image avec leur incertitude (ou non) sur le même graphe.
    Arguments :
    - d_spe (dico) : dictionnaire des espèces contenus dans une image
    - f (array) : longueur d'onde
    - ax (axe de plt) : le subplot sur lequel afficher les courbes
    """
    
    for Spe in d_Spe :
        ax.plot(f,d_Spe[Spe][0],label=Spe)
        if i!=None : 
            ax.fill_between(f,d_Spe[Spe][0]-d_Spe[Spe][1], d_Spe[Spe][0]+d_Spe[Spe][1],alpha=0.3)
#           show_sp_curve(Spe,d_Spe[Spe][0],d_Spe[Spe][1],f)
        
def show_moy_sp(image_id):
    
    """"Affiche toutes les courbes moyennes des espèces de l'image "image_id". """
    
    hsi=open_hsi(image_id)
    fig, ax=plt.subplots()
    
    d_Spe,f=sp_species_moy(image_id,hsi)
    show_sp_dico(d_Spe,f,ax)
    ax.legend()
    plt.title(f"Spectre moyen des espèces sur le l'image {image_id}")
    plt.show()
    
    
def sp_norm(sp_moy,sp_pixel):
    
    """"Calcule la norme entre le spectre d'un pixel et la moyenne d'un spectre.
    Arguments :
    - sp_moy (array) : le spectre moyen d'une espèce
    - sp_pixel (array) : le spectre du pixel dont on veut déterminer l'espèce
    Output :
    - norm (float) : c'est la norme/la distance euclidienne entre ces deux courbes
    """
    
    norm=np.linalg.norm(sp_moy-sp_pixel)
    return norm


def sp_norm_dico(d_Spe,sp_pixel):
    
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
    Cla=[]
    for Spe in d_Spe:
        Norm=sp_norm(d_Spe[Spe][0],sp_pixel)
        if Norm<N :
            Id=Spe
            N=Norm
        Cla.append([round(float(Norm)),Spe])
    return Id,N,Cla


def test_pixel(image_id,row,col):
    
    """Permet de tester à quelle espèce un pixel appartient"""
    
    hsi=open_hsi(image_id)
    d_spe,f=sp_species_moy(image_id,hsi)
    pixel=hsi[row][col]
    Id,norm,Cla=sp_norm_dico(d_spe,pixel)
    print("Le pixel de coordonnées ",str((row,col))," correspond probablement à l'espèce :",Id,",avec une norme de :",str(norm))

#%% Algorithm k-nn

def open_csv(image_id,mapname,tree=False):
    
    """Permet d'ouvrir le fichier csv des pixels d'une image sous forme d'accès au fichier
    Arguments :
    - image_id (str) : image que l'on veut sélectionner dans le fichier csv
    Output :
    - df : chemin vers l'image sélectionnée du fichier csv 
    """
    
    cwd = Path(__file__).resolve().parents[2] 
    
    mapname_label = "data/raw/forest-plot-analysis/data/gt/"+mapname+".csv"
    map_label = cwd / mapname_label 
    
    df = pd.read_csv(map_label) #Lis le fichier csv Pixels
    if tree:
        df = df[df["plotid"] == image_id]#Sélectionne l'id des arbres 
    
    return df


def train_img(image_id,LiDAR=False):
    
    """Détermine les données d'entraînement d'une image, c'est à dire une liste x avec le spectre de chaque pixel,
    et une y avec le label de chaque pixel (ie à quelle espèce correspond ce pixel). L'ordre est donc important
    Arguments :
    - image_id (str) : l'image sur laquelle on veut s'entraîner
    Outputs :
    - X_train (array) : array de (160,n) où n est le nombre de pixels dont on a l'identité dans une image, et 160 correspond donc au spectre d'une image
    - Y_train (array) : l'identité associée à chacun des pixels
    """
    
    df=open_csv(image_id,"df_pixel",tree=True)
    hsi=open_hsi(image_id)
    K=0
    if LiDAR:
        dl=open_lidar(image_id)
        K=24
        w=len(dl[dl['0']==0])
    
    X=np.zeros((1,160+K))
    Y=[]
    
    specie,row,col=df["specie"],df["row"],df["col"]
    
    n=len(row)
    
    for i in range(n):
        x,y=row.iloc[i],col.iloc[i]
        spectre=np.array(hsi[x,y,:])
        
        #spectre=savgol_filter(spectre,window_length=11,polyorder=2)
        if LiDAR:
            Li=dl.to_numpy()[x*w+y+1]
            lidar=Li[2:]
            if np.any(lidar):
                spectre=np.append(spectre,lidar)
                X=np.append(X,[spectre],axis=0)
                Y.append(specie.iloc[i])
            
        else :
            X=np.append(X,[spectre],axis=0)
            Y.append(specie.iloc[i])
              
    X=np.delete(X,0,axis=0)
           
    X_train=X
    Y_train=np.array(Y)
    print(X_train.shape,Y_train.shape)
    return X_train,Y_train


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
    
    
def algo_comparison(Scaler=False,LiDAR=False):
    
    comp=None
    
    if Scaler:
        comp="Scaler"
    elif LiDAR :
        comp="LiDAR"
    else :
        return 0
    
    print("\n\t\tSans "+comp+" :\n ")
    
    X_train, x_test, Y_train, y_test=train_MLA()

    Ka,Kr,Km=MLA("KNN",X_train, x_test, Y_train, y_test)
    Ra,Rr,Rm=MLA("RdF",X_train, x_test, Y_train, y_test)
    Xa,Xr,Xm=MLA("XGB",X_train, x_test, Y_train, y_test)
    Ca,Cr,Cm=MLA("SVM",X_train, x_test, Y_train, y_test)
    
    X_train, x_test, Y_train, y_test=train_MLA(Scaler=Scaler,LiDAR=LiDAR)

    print("\n\t\tAvec "+comp+" :\n ")
    
    Ka_s,Kr_s,Km_s=MLA("KNN",X_train, x_test, Y_train, y_test)
    Ra_s,Rr_s,Rm_s=MLA("RdF",X_train, x_test, Y_train, y_test)
    Xa_s,Xr_s,Xm_s=MLA("XGB",X_train, x_test, Y_train, y_test)
    Ca_s,Cr_s,Cm_s=MLA("SVM",X_train, x_test, Y_train, y_test)
    
    data = {
    'Modèle': ['KNN', 'RandomForest', 'XGB', 'SVM'] * 2,
    'Accuracy': [Ka, Ra, Xa, Ca, Ka_s, Ra_s, Xa_s, Ca_s],  # Sans Scaler puis Avec Scaler
    'Scaler': ['Sans Scaler']*4 + ['Avec Scaler']*4
    }
    
    return data
    
    
#%% Données Lidar

li_keys=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25']

def open_lidar(lidar_id):
    
    cwd = Path(__file__).resolve().parents[2] 
    mapname_label = "data/raw/forest-plot-analysis/data/lidar_features/raster/img_"+lidar_id+".csv"
    map_label = cwd / mapname_label 
    
    df = pd.read_csv(map_label) #Lis le fichier csv Pixels
    
    return df

    


