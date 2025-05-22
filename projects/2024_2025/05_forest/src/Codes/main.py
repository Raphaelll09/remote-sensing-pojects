import sys
import os
from datetime import datetime

from data import HSI, LiD, GT
import models as md
import show as sh
import matplotlib.pyplot as plt

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def main():
    
    ## !! WARNING !! This algorithm can take few minutes to compile ##
    
    # 0) Prépare dossiers de sortie
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"results_{timestamp}"
    log_dir = os.path.join(out_dir, "logs")
    img_dir = os.path.join(out_dir, "images")
    ensure_dir(log_dir); ensure_dir(img_dir)

    # 1) Redirige stdout/stderr vers un fichier log
    log_path = os.path.join(log_dir, "main_output.log")
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = log_file
    sys.stderr = log_file
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    plot = "1"

    # 2) Inspect raw data
    print(f"[1] HSI shape for plot {plot}: {HSI(plot).shape}")
    print(f"[1] LiDAR shape for plot {plot}: {LiD(plot).shape}")
    counts = GT("df_pixel").query("plotid == @plot")["specie"].value_counts()
    print(f"[1] Species counts:\n{counts.to_dict()}\n")

    # 3) Baseline per-image
    print("[2] Baseline distance-to-mean on single image:")
    scores = md.test_norm_image(plot)
    for sp,(tot,cor,acc) in scores.items():
        print(f"   {sp}: {cor}/{tot} → {acc:.3f}")
    print()

    # 4) Save mean-spectra plot
    print("[3] Saving mean spectra plot…")
    sh.show_spectres_img(plot)
    plt.savefig(os.path.join(img_dir, "mean_spectra.png"))
    plt.close()

    # 5) Save RGB + species overlay
    print("[4] Saving HSI RGB + species overlay…")
    sh.visualize_data(plot, RVB=[65,29,16], alpha=0.3)
    plt.savefig(os.path.join(img_dir, "rgb_species_overlay.png"))
    plt.close()

    # 6) Save LiDAR canopy height map
    print("[5] Saving LiDAR canopy height map…")
    sh.lidar_data(plot)
    plt.savefig(os.path.join(img_dir, "lidar_height_map.png"))
    plt.close()

    # 7) ML on HSI-only
    print("[6] ML on HSI only, with scaling:")
    Xtr, Xte, Ytr, Yte = md.train_MLA(LiDAR=False, scaler=True)
    for alg in ["KNN","RdF","XGB","SVM"]:
        acc, _, _, _ = md.MLA(alg, Xtr, Xte, Ytr, Yte, show=False)
        print(f"   {alg}: {acc:.3f}")
    print()

    # 8) ML on HSI+LiDAR
    print("[7] Random Forest on HSI+LiDAR (with scaling) and confusion matrix:")
    Xtr, Xte, Ytr, Yte = md.train_MLA(LiDAR=True, scaler=True)
    acc_rf, report_rf, _, C_rf = md.MLA("RdF", Xtr, Xte, Ytr, Yte, show=False)
    print(f"   RandomForest accuracy: {acc_rf:.3f}")

    plt.figure(figsize=(6,5))
    sh.show_res_Alg(acc_rf, report_rf, C_rf)
    plt.title("RF Confusion Matrix (HSI+LiDAR)")
    plt.savefig(os.path.join(img_dir, "rf_confusion_matrix.png"))
    plt.close()

    # 9) Comparison plot
    print("[8] Saving comparison bar charts…")
    data_acc, data_mac, data_w = md.algo_comparison(
        Algos=["KNN","RdF","XGB","SVM"],
        LiD_nat=False, scaler=True, LiDAR=False, show=False
    )
    sh.show_scores(data_acc, data_mac, data_w, comp="Scaler")
    plt.savefig(os.path.join(img_dir, "comparison_scores.png"))
    plt.close()

    # 10) Save prediction maps
    print("[9] Saving prediction vs ground truth…")
    md.train_n_save_Alg("XGB", "xgb_hsi_lidar")
    sh.compare_prediction(plot, "xgb_hsi_lidar", alph=True)
    plt.savefig(os.path.join(img_dir, "pred_vs_gt.png"))
    plt.close()

    # 11) End
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr
    print(f"\nAll outputs logged to {log_path}")
    log_file.close()

if __name__ == "__main__":
    main()