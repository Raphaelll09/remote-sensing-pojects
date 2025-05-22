# Tree Species Classification by Hyperspectral & LiDAR Remote Sensing

This repository implements an end-to-end pipeline for classifying tree species in forest plots using airborne hyperspectral imagery (HSI) and LiDAR features. It includes data loading, baseline distance-to-mean classifiers, supervised ML models (KNN, SVM, Random Forest, XGBoost), and a hierarchical classifier (family → majority/minority species). Visualization tools enable exploration of spectra, image composites, species maps, LiDAR canopy heights, and classification results.

    **YOU WILL FIND THE CODES IN : *projects\2024_2025\05_forest\src\Codes***

---

## Table of Contents

1. [Motivation &amp; Overview](#motivation--overview)
2. [Repository Structure](#repository-structure)
3. [Data Organization](#data-organization)
4. [Installation &amp; Setup](#installation--setup)
5. [Usage &amp; Demo (`main.py`)](#usage--demo-mainpy)
6. [Module Summaries](#module-summaries)
   - [data.py](#datapy)
   - [models.py](#modelspy)
   - [show.py](#showpy)
7. [Visual Outputs &amp; Examples](#visual-outputs--examples)
8. [Best Practices &amp; Tips](#best-practices--tips)
9. [License &amp; Citation](#license--citation)

---

## Motivation & Overview

Forest health, biodiversity, and carbon accounting depend on identifying tree species. Traditional field surveys are labor‐intensive. Airborne sensors—HSI for spectral signatures and LiDAR for canopy structure—can automate species mapping. This project:

- Extracts pixel‐level spectra and LiDAR features.
- Computes species‐specific mean spectra as a baseline.
- Trains supervised classifiers on HSI alone, LiDAR alone, and combined.
- Implements a hierarchical approach: first predict conifer vs. broadleaf (“family”), then species within each family, handling rare species separately.
- Provides interactive visualizations for RGB composites, species maps, spectra, and performance metrics.

---

## Repository Structure

├─ data.py         # DataCache class for HSI, LiDAR, ground truth

├─ models.py       # Spectral routines, ML training & evaluation, hierarchical classifier

├─ show.py         # Visualization utilities (images, spectra, maps, results)

├─ main.py         # End-to-end demo script

├─ requirements.txt# Python dependencies

└─ README.md       # This documentation

## Data Organization

Place raw data under:

data/raw/forest-plot-analysis/data/
│
├─ hi/ # Hyperspectral TIFF files: `<plotID>`.tif
├─ lidar_features/
│ └─ raster/ # CSV per image: img_`<plotID>`.csv
└─ gt/ # Ground truth CSVs: df_pixel.csv, df_tree.csv, Chamrousse.csv

- **df_pixel.csv**: one row per pixel, with plotID, row, col, specie, treeid, family.
- **df_tree.csv**: crown‐level metadata (`npix` = pixel count).
- **Chamrousse.csv**: circle (x, y, radius) and height info for a study subset.

---

## Installation & Setup

1. **Clone** this repo:

   ```bash
   git clone https://github.com/<your_username>/tree-hsi-lidar.git
   cd tree-hsi-lidar
   ```
2. **Create & activate** a Python ≥3.9 environment:

   ```
   python -m venv venv
   source venv/bin/activate      # Linux / macOS
   venv\Scripts\activate.bat     # Windows
   ```
3. **Install dependencies** :

   ```
   pip install -r requirements.txt
   ```
4. **Verify data** : ensure `data/raw/...` directory is populated.

## Usage & Demo (`main.py`)

Run the full pipeline end-to-end:

```
python main.py

```

This will:

1. Load HSI & LiDAR for plot “1b”, display shapes & species counts.
2. Compute baseline distance‐to‐mean classification on one pixel.
3. Train & evaluate KNN, RF, XGB, SVM on HSI only (with scaling).
4. Show RGB composite, species map, and LiDAR canopy height map.
5. Train & evaluate Random Forest on combined HSI+LiDAR.
6. Run hierarchical classifier (family → species) with XGBoost.

## Module Summaries

* **data.py**
  * `DataCache`: load & cache HSI, LiDAR, GT.
  * Accessors: `HSI()`, `LiD()`, `GT()`.
* **models.py**
  * Spectral analysis: `pixel()`, `pixel_spectre_moy()`, `species_spectre_moy()`, `spectre_norm_dico()`.
  * Baseline tests: `test_pixel_norm()`, `test_norm_image()`.
  * Data prep: `train_img()`, `train_All_img()`, `train_full_img()`.
  * ML pipeline: `train_MLA()`, `MLA_model()`, `MLA()`, `algo_comparison()`.
  * Hierarchy: `hierarchical_MLA()`.
* **show.py**
  * Spectra plots: `show_spectre_pixel()`, `show_spectre_moy()`, `show_moy_spectre()`.
  * Image & map: `show_image_RVB()`, `show_map_specie()`, `visualize_data()`.
  * LiDAR viz: `lidar_data()`.
  * Chamrousse: `chamrousse()`.
  * Results: `show_res_Alg()`, `show_scores()`, `show_colormap()`, `compare_prediction()`.

---

## Visual Outputs & Examples

* **RGB Composite** (bands 65/29/16)
* **Species Map** ground truth
* **LiDAR Canopy Height** contours & scatter
* **Mean Spectra** plots with 95% CI
* **Accuracy & Reports** for each classifier
* **Comparison Charts** (before/after scaling, LiDAR)
* **Hierarchical Report** (family then species)

---

## Best Practices & Tips

* Use **SMOTE** for class imbalance (`train_MLA(resample=True)`).
* Tune hyperparameters via **GridSearchCV** with `scoring='f1_macro'`.
* Consider **PCA** to reduce 160+24 features for speed.
* Replace `print` with `logging` for production code.
* Write **unit tests** for data loaders and key routines.

## License & Citation

This project is released under the MIT License.

If you use it academically, please cite:

> PHILIPPON C.,DELAHAYE J., DESAILLY N., EVRARD R., “Identification of Tree Species in a Forest Area Using Remote Sensing,” 2025.
