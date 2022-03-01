import numpy as np
import csv
import sys
import math
import pickle

from validate import validate

def import_data():
    X=np.genfromtxt("train_X_de.csv",delimiter=",",dtype=np.float64,skip_header=1)
    Y=np.genfromtxt("train_Y_de.csv",delimiter=",",dtype=np.float64)
    return X,Y

def mean_normalize(X):
    for i in range(np.shape(X)[1]):
        X[:,i]=(X[:,i]-np.mean(X[:,i]))/(np.max(X[:,i])-np.min(X[:,i]))
        
    return X

def calculate_entropy(Y):
    Y1=list(set(Y))
    T=len(Y)
    sum=0
    for i in Y1:
        p=Y.count(i)/T
        sum+=-p*math.log(p,2)
        
    return sum

def calculate_information_gain(Y_subsets):
    PN=np.concatenate(Y_subsets)
    IG=0
    m=len(PN)
    H_S=calculate_entropy(PN)
    N=np.shape(PN)[0]
    
    for i in range(N):
        IG+=-(len(Y_subsets[i])/m)*calculate_entropy(Y_subsets[i])
    
    return H_S+IG   

def calculate_gain_ratio(Y_subsets):
    PN=np.concatenate(Y_subsets)
    m=len(PN)
    N=np.shape(Y_subsets)[0]
    SI=0
    for i in range(N):
        SI+=-(len(Y_subsets[i])/m)*math.log(len(Y_subsets[i])/m,2)
    
    return calculate_information_gain(Y_subsets)/SI

def calculate_gini_index(Y_subsets):
    gini,gini_index=0,0
    PN=np.concatenate(Y_subsets)
    m=len(PN)
    uni=list(set(PN))
    N=np.size(Y_subsets)[0]
    for i in range(N):
        n=len(Y_subsets[i])
        if n==0:
            continue
        count=[Y_subsets[i].count(j) for j in uni]
        gini=1-sum((k/n)**2 for k in count)
        gini_index+=(n/m)*gini
        
    return gini_index    

def split_data_set(data_X, data_Y, feature_index, threshold):
    left_X=[]
    left_Y=[]
    right_X=[]
    right_Y=[]
    for i in range(len(data_X)):
        if data_X[i][feature_index] < threshold:
            left_X.append(data_X[i])
            left_Y.append(data_Y[i])
        else:
            right_X.append(data_X[i])
            right_Y.append(data_Y[i])

    return left_X, left_Y, right_X, right_Y

def get_best_split(X, Y):
    X=np.array(X)
    best_gini_index=9999
    best_feature=0
    best_threshold=0
    for i in range(len(X[0])):
        thresholds=sorted(set(X[:,i]))
        for j in thresholds:
            left_X,left_Y,right_X,right_Y=split_data_set(X,Y,i,j)
            if len(left_X)==0 or len(right_X)==0:
                continue
            gini_index=calculate_gini_index([left_Y,right_Y])
            if gini_index < best_gini_index:
                best_gini_index,best_feature,best_threshold = gini_index, i, j
    
    return best_feature, best_threshold

class Node:
    def __init__(self, predicted_class, depth):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.depth = depth
        self.left = None
        self.right = None
        
def construct_tree(X, Y, max_depth, min_size, depth):
    Y=np.array(Y)
    classes=list(set(Y))
    predicted_class=classes[np.argmax([np.sum(Y==c) for c in classes])]
    node = Node(predicted_class, depth)
    
    if len(set(Y)) == 1:
        return node
    
    if depth >= max_depth:
        return node
        
    if len(Y) <= min_size:
        return node
        
    feature_index, threshold = get_best_split(X,Y)
    
    if feature_index is None or threshold is None:
        return node
        
    node.feature_index = feature_index
    node.threshold = threshold
    
    left_X, left_Y, right_X, right_Y = split_data_set(X,Y,feature_index,threshold)
    
    node.left = construct_tree(np.array(left_X), np.array(left_Y), max_depth, min_size, depth+1)
    node.right = construct_tree(np.array(right_X), np.array(right_Y), max_depth, min_size, depth+1)
    return node

def print_tree(node):
    if node.left is not None and node.right is not None:
        print("X" + str(node.feature_index) + " " + str(node.threshold))
    if node.left is not None:
        print_tree(node.left)
    if node.right is not None:
        print_tree(node.right)
        
def predict(root, X):
    ypred=[]
    for i in range(len(X)):
        node = root
        if test_X[i][node.feature_index] < node.threshold:
            node = node.left
        
        else: 
            node = node.right
        
        ypred.append(node.predicted_class)
        
    return np.array(ypred)

if __name__=="__main__":
    X,Y=import_data()
    X=mean_normalize(X)
    node = construct_tree(X,Y,7,1,0)
    pickle.dump(node, open('MODEL_FILE.sav', 'wb'))
    
