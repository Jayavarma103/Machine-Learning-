import numpy as np
import csv
import sys
import math
import string 
import json
import collections

from validate import validate

def import_data():
    X=np.genfromtxt("train_X_nb.csv",delimiter='\n',dtype=str)
    Y=np.genfromtxt("train_Y_nb.csv",delimiter='\n',dtype=str)
    return X,Y

def write_model(model,path):
    dump=json.dumps(model)
    with open(path, "w") as outfile:
        outfile.write(dump)
        
def preprocessing(X):
    final=[]
    for s in X:
        s=s.lower()
        l=list(string.ascii_lowercase)
        s=list(s)
        s_new=""

        for i in range(len(s)):
            s_new+=s[i]
            if s[i]==" " and s_new:
                if s_new[-1]!=" ":
                    s_new+=s[i]
        final.append(s_new)

    return final

def compute_prior_probabilities(Y):
    prior_prob=dict()

    prior_prob=collections.Counter(Y)
    for key in prior_prob.keys():
        prior_prob[key]/=len(Y)
    
            
    return prior_prob

def class_wise_words_frequency_dict(X, Y):
    class_wise_frequency_dict=dict()
    for i in range(len(X)):
        words=X[i].split()
        for token_word in words:
            y=Y[i]
            if y not in class_wise_frequency_dict:
                class_wise_frequency_dict[y]=dict()
            if token_word not in class_wise_frequency_dict[y]:
                class_wise_frequency_dict[y][token_word]=0
            class_wise_frequency_dict[y][token_word]+=1
    
    return class_wise_frequency_dict

def get_class_wise_denominators_likelihood(X, Y, alpha):
    class_wise_frequency_dict=class_wise_words_frequency_dict(X,Y)
    
    class_wise_denominator=dict()
    classes=list(set(Y))
    vocabulary=[]
    for c in classes:
        frequency_dict=class_wise_frequency_dict[c]
        class_wise_denominator[c]=sum(list(frequency_dict.values()))
        vocabulary+=list(frequency_dict.keys())
    
    vocabulary=list(set(vocabulary))
    
    for c in classes:
        class_wise_denominator[c]+=alpha*len(vocabulary)
    
    return class_wise_denominator

def train_model():
    X,Y=import_data()
    X=preprocessing(X)
    alpha=1
    model=[list(set(Y)), class_wise_words_frequency_dict(X, Y)]
    model.append(get_class_wise_denominators_likelihood(X, Y, alpha))
    model.append(compute_prior_probabilities(Y))
    model.append(alpha)
    write_model(model, "TRAINED_MODEL.json")
    
if __name__=="__main__":
    train_model()
    
