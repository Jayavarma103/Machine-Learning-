import numpy as np
import csv
import sys
from train import get_best_k_using_validation_set, calculate_accuracy, classify_points_using_knn, find_k_nearest_neighbors, compute_ln_norm_distance

from validate import validate

def import_data(test_X_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    return test_X


def predict_target_values(test_X):
    train_X=np.genfromtxt("train_X_knn.csv",delimiter=",",dtype=np.float64,skip_header=1)
    train_Y=np.genfromtxt("train_Y_knn.csv",delimiter=",",dtype=np.float64)
    k=get_best_k_using_validation_set(train_X,train_Y,30,2)
    predicted_Y=classify_points_using_knn(train_X,train_Y,test_X,2,k)
    predicted_Y = np.array(predicted_Y).reshape(test_X.shape[0], 1)
    return predicted_Y

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X=import_data(test_X_file_path)
    pred_Y = predict_target_values(test_X)
    write_to_csv_file(pred_Y, "predicted_test_Y_knn.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_knn.csv") 
