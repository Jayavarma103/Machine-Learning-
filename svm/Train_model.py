import numpy as np
import csv
import sys
import pickle
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix

from validate import validate

def import_data():
    X=np.genfromtxt("train_X_svm.csv",delimiter=",",dtype=np.float64,skip_header=1)
    Y=np.genfromtxt("train_Y_svm.csv",delimiter=",",dtype=np.float64)
    return X,Y

def mean_normalize(X):
    for i in range(np.shape(X)[1]):
        X[:,i]=(X[:,i]-np.mean(X[:,i]))/(np.max(X[:,i])-np.min(X[:,i]))
        
    return X

def gaussian_kernel(X1, X2):
    sigma=3
    norm=np.linalg.norm(X1-X2)
    K=np.exp(-norm**2/(2*(sigma**2)))
    return K

def proxy_kernel(X, Y, K=gaussian_kernel):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            gram_matrix[i, j] = K(x, y)
    return gram_matrix

def train_model(X,Y):
    #Y = Y.reshape(len(X),1)
    sv = svm.SVC(C=20,kernel='rbf',gamma=1)
    sv.fit(X,Y)
    pickle.dump(sv, open('MODEL_FILE.sav', 'wb'))
    
def predict_label(X):
    sv = pickle.load(open('MODEL_FILE.sav', 'rb'))
    y_pred=sv.predict(X)
    return y_pred

if __name__=="__main__":
    X,Y=import_data()
    X=mean_normalize(X)
    train_model(X,Y)
    y_pred=predict_label(X)
    print(confusion_matrix(Y,y_pred))
    print(classification_report(Y,y_pred))
    