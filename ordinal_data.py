import numpy as np
# import scipy
import pandas
import json

ENCODING = 512
NUM_CLASSES = 3

def load_ordinal_data():
    trainingData = pandas.read_csv("train.csv")

    Xtrain = trainingData.to_numpy()[:, :-1] # ignores labels in last column
    Ytrain = np.atleast_2d(trainingData.to_numpy()[:, -1]).T # grabs labels from last column
    Ytrain = Ytrain.reshape(Ytrain.shape[0], 1) # makes shape a 2d array, easier for later

    Xtrain = np.delete(Xtrain, [0, 1, 8], axis=1)

    num_train = Xtrain.shape[0]
    onehot_train_labels = np.zeros((num_train, NUM_CLASSES)) # 3 classes to predict
    onehot_train_labels[np.arange(num_train), Ytrain[:, 0].astype(int)] = 1 # performs one hot encoding

    onehot_train_coldpresent = np.zeros((num_train, 3))
    onehot_train_coldpresent[Xtrain[:, 6] == 0, 0] = 1
    onehot_train_coldpresent[Xtrain[:, 6] == 1, 1] = 1
    onehot_train_coldpresent[np.isnan(Xtrain[:, 6].astype(float)), 2] = 1

    Xtrain = np.concatenate((Xtrain[:, 0:6].astype(float), onehot_train_coldpresent, Xtrain[:, 6].reshape(num_train, 1).astype(float)), axis=1, dtype=float)

    return Xtrain, onehot_train_labels