#%% Librairies

from pathlib import Path
import tifffile
import pandas as pd

#%% Variables

keys=["ABAL","ACPS","BEPE","BEsp","COAV","FASY","FREX","PIAB","PICE","PIUN","POTR","SOAR","SOAU"]
d_spec= {k : None for k in keys}

LiDAR_keys=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25']



#%% Fonctions

class DataCache:
    
    def __init__(self):
        
        self.hsi = {}     # Hyperspectral images
        self.lidar = {}   # LiDAR raster features
        self.gt = {}      # Ground truth CSV (df_pixel, df_tree, Chamrousse)
        
        # Base path (à adapter si nécessaire)
        self.base = Path(__file__).resolve().parents[2] / "data/raw/forest-plot-analysis/data"

    def get_hsi(self, image_id: str):
        
        """Charge l'image HSI (3D) d'une parcelle"""
        if image_id not in self.hsi:
            path = self.base / "hi" / f"{image_id}.tif"
            self.hsi[image_id] = tifffile.imread(path)
        return self.hsi[image_id]

    def get_lidar(self, image_id: str):
        
        """Charge les features LiDAR rasterisées pour une parcelle"""
        if image_id not in self.lidar:
            path = self.base / "lidar_features" / "raster" / f"img_{image_id}.csv"
            self.lidar[image_id] = pd.read_csv(path)
        return self.lidar[image_id]

    def get_gt(self, filename: str):
        
        """Charge un fichier de vérité terrain : df_pixel, df_tree, Chamrousse"""
        if filename not in self.gt:
            path = self.base / "gt" / f"{filename}.csv"
            self.gt[filename] = pd.read_csv(path)
        return self.gt[filename]

    def clear_all(self):
        
        """Vide tous les caches pour libérer la mémoire"""
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
