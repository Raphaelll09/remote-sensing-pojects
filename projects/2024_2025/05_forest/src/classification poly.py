# Import modules
%matplotlib inline
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
# Select random seed
random_state = 0
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import PolynomialFeatures


# # créer les échantillons
# from sklearn.datasets import make_moons
# X,y = make_moons(n_samples=1000, random_state=random_state, noise=0.25) 





from sklearn.model_selection import train_test_split
# We split the initial set in two sets: one, namely the training set, use for training the model, 
# and one, namely the test set, use to compute the validation error
# -> test_size x n_samples for the test set and n_samples x (1- test_size) for the training set
# where test_size is given as a parameter
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state= random_state )
y_train[y_train==0]=-1
y_test[y_test==0]=-1



# # Display the training set
# plt.scatter(X_train[:,0], X_train[:,1],c=y_train)
# plt.grid()



from sklearn import linear_model
from sklearn.preprocessing import  StandardScaler

# Train the Least Squares (Ridge) Classifier with polynomial features
degree = 10# Degree of the polynomial 1 ->20

model = make_pipeline(PolynomialFeatures(degree),StandardScaler(), linear_model.RidgeClassifier(alpha=1e-6))
model.fit(X_train, y_train)

#Plot the decision functions
XX, YY = np.meshgrid(np.linspace(X_train[:,0].min(), X_train[:,0].max(),200),
                      np.linspace(X_train[:,1].min(), X_train[:,1].max(),200))
XY = np.vstack([ XX.flatten(), YY.flatten() ]).T
yp= model.predict(XY)
plt.contour(XX,YY,yp.reshape(XX.shape),[0])
plt.grid()
plt.scatter(X_train[:,0], X_train[:,1], c=y_train)


from sklearn.model_selection import validation_curve


lin_model = linear_model.RidgeClassifier(alpha=1e-6)

degrees= np.arange(1,20)
train_error_rate= np.zeros(degrees.shape)
test_error_rate= np.zeros(degrees.shape)
for i,deg in enumerate(degrees):
    model = make_pipeline(PolynomialFeatures(deg),StandardScaler(), lin_model )
    model.fit(X_train,y_train )
    y_pred= model.predict( X_train )
    train_error_rate = np.mean( y_train != y_pred)
    y_pred= model.predict( X_test )
    test_error_rate = np.mean( y_test != y_pred)
print(y_pred)


# #train_scores, valid_scores = validation_curve(pipe, X, y, "poly__degree", range(1,20), cv=10)
# plt.plot(range(1, 20), train_error_rate, label="Training error")
# plt.plot(range(1, 20), test_error_rate, label="Test error")
# plt.grid()
# plt.xticks(range(1, 20))
# plt.legend()
# plt.xlabel("Polynomial degrees $d$")
# plt.ylabel("Misclassification rate")
# #plt.savefig("train_vs_test_error.pdf")

