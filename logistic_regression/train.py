import numpy as np
import csv
import sys

from validate import validate

def import_data():
    X=np.genfromtxt("train_X_lg_v2.csv",delimiter=",",dtype=np.float64,skip_header=1)
    Y=np.genfromtxt("train_Y_lg_v2.csv",delimiter=",",dtype=np.float64)
    return X,Y

def sigmoid(Z):
    sigmoid=1/(1+np.exp(-Z))
    return sigmoid

def compute_cost(X, Y, W, b):
    M=len(X)
    Z=np.dot(X,W)+b
    A=sigmoid(Z)
    A[A==1]=0.99999
    A[A==0]=0.00001
    cost=(-1/M)*sum((Y*np.log(A) + (1-Y)*np.log(1-A)))
    return cost[0]

def compute_gradient_of_cost_function(X, Y, W, b):
    M=len(X)
    Z=np.dot(X,W)+b
    A=sigmoid(Z)
    dW=(1/M)*np.dot(X.T,(A-Y))
    db=(1/M)*np.sum(A-Y)
    return dW,db

def optimize_weights_using_gradient_descent(X,Y,W,b,num_iterations,learning_rate):
    A = sigmoid(np.dot(X, W) + b)
    i=0
    last_cost = 0
    while True:
        dw, db = compute_gradient_of_cost_function(X,Y,W,b)
        
        W = W - learning_rate*dw
        b = b - learning_rate*db
        
        new_cost = compute_cost(X, Y, W, b)
        
        if i%1000==0:
            print(new_cost)
        if abs(last_cost - new_cost)<0.0000001:
            print(new_cost,i)
            break
        last_cost = new_cost
        i+=1
        
    return W, b

def train_model(X, Y):
    Y = Y.reshape(len(X),1)
    W = np.zeros((X.shape[1],1))
    b = 0
    W ,b = optimize_weights_using_gradient_descent(X, Y, W, b, 10, 0.0002)
    W = np.append(W,b)
    return W

def predict_labels(X,W,b):
    Z=np.dot(X,W)+b
    A=sigmoid(Z)
    A=A.T[0]
    Y_prediction=np.where(A>=0.5,1,0)
    return Y_prediction

def save_model(weights,weights_file_name):
    with open(weights_file_name,'w',newline='') as weights_file:
        wr=csv.writer(weights_file)
        wr.writerows(weights)
        weights_file.close()
        
if __name__=="__main__":
    train_X,train_Y=import_data()
    weights_array=[]
    for i in range(4):
        class_label=i
        X,Y=get_train_data_for_class(train_X,train_Y,class_label)
        weights=train_model(X,Y)
        weights_array.append(weights)
    save_model(weights_array,'WEIGHTS_FILE.CSV')
    

