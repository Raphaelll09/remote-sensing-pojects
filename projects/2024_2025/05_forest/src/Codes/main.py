from data import *
from show import *
from models import *


c=DataCache()

def main():
    
    X,Y=train_All_img()
    print(X.shape,Y.shape)
    X,Y=train_All_img(LiDAR=True)
    print(X.shape,Y.shape)
    
if __name__ == "__main__":
    main()