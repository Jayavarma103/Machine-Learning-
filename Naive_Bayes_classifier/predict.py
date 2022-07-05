import numpy as np
import csv
import sys
import json
import math

from validate import validate

def import_data_and_model(test_X_file_path, model_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter='\n', dtype=str)
    with open(model_file_path, "r") as read_file:
        model=json.load(read_file)
    return test_X, model

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

def compute_likelihood(test_X, c,class_wise_frequency_dict,class_wise_denominators,prior_prob,alpha):
    likelihood=0
    words=test_X.split(" ")
    for word in words:
        count=0
        words_frequency=class_wise_frequency_dict[c]
        if word in words_frequency:
            count=class_wise_frequency_dict[c][word]
        likelihood+=(math.log((count+alpha)/class_wise_denominators[c]))
    
    return likelihood+(math.log(prior_prob[c]))


def predict_target_values(test_X, model):
    [classes, class_wise_frequency_dict,class_wise_denominators,prior_prob,alpha]=model
    pred_Y=[]
    
    for x in test_X:
        best_p=-9999999
        best_c=-1
        for c in classes:
            p=compute_likelihood(x,c,class_wise_frequency_dict,class_wise_denominators,prior_prob,alpha)
            if p>best_p:
                best_p=p
                best_c=c
        pred_Y.append(best_c) 
    return np.array(pred_Y)


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X, model = import_data_and_model(test_X_file_path, "./TRAINED_MODEL.json")
    pred_Y = predict_target_values(test_X, model)
    write_to_csv_file(pred_Y, "predicted_test_Y_nb.csv")    


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_nb.csv") 
