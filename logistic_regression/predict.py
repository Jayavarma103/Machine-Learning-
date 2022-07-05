import numpy as np
import csv
import sys
from train import sigmoid
from validate import validate

def import_data_and_weights(test_X_file_path, weights_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    weights = np.genfromtxt(weights_file_path, delimiter=',', dtype=np.float64)
    return test_X, weights

def sort_(array):
    array0, array1, array2, array3 = array[0], array[1], array[2], array[3]
    out = []
    for (i,j,k,l) in zip(array0,array1,array2,array3):
        m = max(i,j,k,l)
        if m == i:
            out.append(0)
        elif m == j:
            out.append(1)
        elif m == k:
            out.append(2)
        elif m == l:
            out.append(3)
    return out

def predict_target_values(test_X, weights):
    result = []
    for i in range(4):
        W = weights[i].T[:-1]
        b = weights[i].T[-1]
        Z = np.dot(test_X,W) + b
        A = sigmoid(Z).T
        result.append(A)
    pred_Y = np.array(sort_(result))
    return pred_Y

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()

def predict(test_X_file_path):
    test_X, weights = import_data_and_weights(test_X_file_path, "WEIGHTS_FILE.csv")
    pred_Y = predict_target_values(test_X, weights)
    write_to_csv_file(pred_Y, "predicted_test_Y_lg.csv")

if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_lg_v2.csv") 
