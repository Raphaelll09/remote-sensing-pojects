"""
data.py

Module de gestion centralisée des données de télédétection forestière, incluant :
- Chargement des images hyperspectrales (HSI)
- Chargement des données LiDAR rasterisées
- Chargement des fichiers d'annotations de terrain (ground truth)

Fonctionnalités :
- Mise en cache automatique des fichiers pour optimiser les performances
- Accès simplifié aux fichiers CSV et TIF liés à chaque parcelle forestière
- Nettoyage explicite des caches en mémoire

Classes :
    DataCache : Gestionnaire de cache pour les jeux de données utilisés

Variables globales :
    keys : Liste des espèces étudiées (13 classes)
    LiDAR_keys : Liste des noms de colonnes LiDAR (26 variables rasterisées)

Usage :
    c = DataCache()
    hsi = c.get_hsi("1b")
    lidar = c.get_lidar("1b")
    df_pixel = c.get_gt("df_pixel")

Projet : Classification d’espèces d’arbres par télédétection (HSI + LiDAR)
"""

#%% Librairies

from pathlib import Path
import tifffile
import pandas as pd

#%% Variables

keys=["ABAL","ACPS","BEPE","BEsp","COAV","FASY","FREX","PIAB","PICE","PIUN","POTR","SOAR","SOAU"]
d_spec= {k : None for k in keys}

LiDAR_keys=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25']

minority=["ACPS","BEsp","FREX","PICE","POTR","SOAR","SOAU"]
majority=["ABAL","BEPE","COAV","FASY","PIAB","PIUN"]
family=["CONI","BROA"]

#%% Fonctions

class DataCache:
    
    """
    Cache manager for hyperspectral images (HSI), LiDAR features, and ground truth data.

    Attributes:
        hsi (dict): Cached HSI arrays keyed by image ID.
        lidar (dict): Cached LiDAR DataFrames keyed by image ID.
        gt (dict): Cached ground truth DataFrames keyed by filename.
        base (Path): Base path to the raw data directory.
    """
    
    def __init__(self):
        
        self.hsi = {}     # Hyperspectral images
        self.lidar = {}   # LiDAR raster features
        self.gt = {}      # Ground truth CSV (df_pixel, df_tree, Chamrousse)
        
        # Base path (à adapter si nécessaire)
        self.base = Path(__file__).resolve().parents[2] / "data/raw/forest-plot-analysis/data"

    def get_hsi(self, image_id: str):
        
        """
        Load and cache a hyperspectral image (.tif) by its ID.

        Args:
            image_id (str): Plot identifier (e.g., "1", "3b").

        Returns:
            np.ndarray: Hyperspectral image array (height x width x bands).
        """
        
        if image_id not in self.hsi:
            path = self.base / "hi" / f"{image_id}.tif"
            self.hsi[image_id] = tifffile.imread(path)
        return self.hsi[image_id]

    def get_lidar(self, image_id: str):
        
        """
        Load and cache LiDAR raster features for a given image.

        Args:
            image_id (str): Plot identifier.

        Returns:
            pd.DataFrame: LiDAR features for each pixel of the image.
        """
        
        if image_id not in self.lidar:
            path = self.base / "lidar_features" / "raster" / f"img_{image_id}.csv"
            self.lidar[image_id] = pd.read_csv(path)
        return self.lidar[image_id]

    def get_gt(self, filename: str):
        
        """
        Load and cache a ground-truth CSV file by name.

        Args:
            filename (str): CSV file name without extension (e.g., "df_pixel", "df_tree").

        Returns:
            pd.DataFrame: Ground truth data from the specified CSV.
        """
        
        if filename not in self.gt:
            path = self.base / "gt" / f"{filename}.csv"
            self.gt[filename] = pd.read_csv(path)
        return self.gt[filename]

    def clear_all(self):
        
        """
        Clear all cached data (HSI, LiDAR, ground truth).
        """
        self.hsi.clear()
        self.lidar.clear()
        self.gt.clear()

    def clear_hsi(self):
        self.hsi.clear()

    def clear_lidar(self):
        self.lidar.clear()

    def clear_gt(self):
        self.gt.clear()

c=DataCache()

HSI=c.get_hsi
GT=c.get_gt
LiD=c.get_lidar
c.clear_all()
