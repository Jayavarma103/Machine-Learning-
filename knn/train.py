import numpy as np
import csv
import sys
import math
from validate import validate

def import_data():
    X=np.genfromtxt("train_X_knn.csv",delimiter=",",dtype=np.float64,skip_header=1)
    Y=np.genfromtxt("train_Y_knn.csv",delimiter=",",dtype=np.float64)
    return X,Y

def compute_ln_norm_distance(vector1, vector2, n):
    Ln=0
    v1=np.array(vector1)
    v2=np.array(vector2)
    for i in range(len(v1)):
        Ln+=abs(v1[i]-v2[i])**n
    return Ln**(1/n)

def find_k_nearest_neighbors(train_X, test_example, k, n):
    Dis=[]
    for i in range(len(train_X)):
        Dis.append(compute_ln_norm_distance(train_X[i],test_example,n))
    Dis1=sorted(Dis)
    top_k_indices=[Dis.index(i) for i in Dis1[:k]]
    return top_k_indices

def classify_points_using_knn(train_X, train_Y, test_X, k, n):
    test_Y=[]
    for test_elem_x in test_X:
        top_k_indices=find_k_nearest_neighbors(train_X,test_elem_x,k,n)
        top_knn_labels=[]
        
        for i in top_k_indices:
            top_knn_labels.append(train_Y[i])
        Y_values=list(set(top_knn_labels))
        
        max_count=0
        most_frequent_label=-1
        for y in Y_values:
            count=top_knn_labels.count(y)
            if(count>max_count):
                max_count=count
                most_frequent_label=y
        test_Y.append(most_frequent_label)
    return test_Y

def calculate_accuracy(predicted_Y, actual_Y):
    numerator=0
    for i in range(len(predicted_Y)):
        if (predicted_Y[i]==actual_Y[i]):
            numerator+=1
    
    return numerator/len(predicted_Y)

def get_best_k_using_validation_set(train_X, train_Y, validation_split_percent,n):
    total_observation=len(train_X)
    train_len=math.floor(float(100-validation_split_percent)/100*total_observation)
    validation_X=train_X[train_len:]
    validation_Y=train_Y[train_len:]
    train_X=train_X[0:train_len]
    train_Y=train_Y[0:train_len]
    
    best_k=-1
    best_accuracy=0
    for k in range(1,train_len+1):
        predicted_Y=classify_points_using_knn(train_X,train_Y,validation_X,n,k)
        accuracy=calculate_accuracy(predicted_Y,validation_Y)
        if accuracy>best_accuracy:
            best_k=k
            best_accuracy=accuracy
            
    return best_k

def train_model(X,Y):
    Y=Y.reshape(len(X),1)
    k=get_best_k_using_validation_set(X,Y,30,2)
    return k

if __name__=="__main__":
    X,Y=import_data()
    K=train_model(X,Y)