from data import *
from show import *
from models import *


c=DataCache()

def main():
     
    pixel_spectre_moy("1","PIAB",normalize=True,std=True,show=True)
    
    # fig, ax=plt.subplots()
    
    # d_Spe,f=species_spectre_moy("1")
    # show_spectre_img(d_Spe, f, ax, fill=False)
    # show_spectre_img(d_Spe, f, ax)
    # data=algo_comparison(LiDAR=True, show=True)
    # show_accuracy(data)
    # data_accuracy, data_macro, data_weighted = algo_comparison(Algos=["RdF"],show=True,LiDAR=True)
    # show_scores(data_accuracy, data_macro, data_weighted,"split_by_tree")
    # visualize_data("Premol",[65,29,16])
    # visualize_data("Premol",[65,29,16],alpha=0.4)
    
if __name__ == "__main__":
    main()