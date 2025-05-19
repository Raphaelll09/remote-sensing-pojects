from New_prog import *
   
def main():
    
    
    # visualize_data("1",[64, 28, 15],alpha=1)
    # show_moy_sp("1")

    # test_pixel("1",34,45)
    
    X,Y=train_All_img()
    print(X.shape,Y.shape)
    X,Y=train_All_img(LiDAR=True)
    print(X.shape,Y.shape)
    
    # X_train, x_test, Y_train, y_test=train_MLA()
    # MLA("KNN",X_train, x_test, Y_train, y_test)
    
    # X_train, x_test, Y_train, y_test=train_MLA(LiDAR=True)
    # MLA("KNN",X_train, x_test, Y_train, y_test)
    
    # np.stack()
    
    
if __name__ == "__main__":
    main()