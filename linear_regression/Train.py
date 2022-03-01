import numpy as np
import csv
import sys

from validate import validate

def import_data():
    X=np.genfromtxt("train_X_lr.csv",delimiter=",",dtype=np.float64,skip_header=1)
    Y=np.genfromtxt("train_Y_lr.csv",delimiter=",",dtype=np.float64)
    return X,Y

def compute_cost(X,Y,W):
    MSE=np.sum((np.matmul(X,W)-Y)**2)
    cost_value=MSE/(2*len(X))
    return cost_value

def compute_gradient_of_cost_function(X, Y, W):
    diff=np.matmul(X,W)-Y
    X_T=np.transpose(X)
    Gr=(np.matmul(X_T,diff))/len(X)
    return np.transpose(Gr)

def optimize_weights_using_gradient_descent(X, Y, W, num_iterations, learning_rate):
    X_T=np.transpose(X)
    for i in range(num_iterations):
        diff=np.matmul(X,W)-Y
        Gr=(np.matmul(X_T,diff))/len(X)
        W=W-(Gr*learning_rate)
        
    return W

def train_model(X,Y):
    X=np.insert(X,0,1,axis=1)
    Y=Y.reshape(len(X),1)
    W=np.zeros((X.shape[1],1))
    W=optimize_weights_using_gradient_descent(X, Y, W, 58000000, 0.0002)
    return W

def save_model(weights,weights_file_name):
    with open(weights_file_name,'w') as weights_file:
        wr=csv.writer(weights_file)
        wr.writerows(weights)
        weights_file.close()
        
        
if __name__=="__main__":
    X,Y=import_data()
    weights=train_model(X,Y)
    save_model(weights, 'WEIGHTS_FILE.csv')