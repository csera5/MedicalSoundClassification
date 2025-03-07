import numpy as np
# import scipy
import pandas
import json

ENCODING = 512
NUM_CLASSES = 3

def load_continuous_no_audio_data():
    trainingData = pandas.read_csv("newTrain.csv")

    Xtrain = trainingData.to_numpy()[:, :-1] # ignores labels in last column
    Ytrain = np.atleast_2d(trainingData.to_numpy()[:, -1]).T # grabs labels from last column
    Ytrain = Ytrain.reshape(Ytrain.shape[0], 1) # makes shape a 2d array, easier for later

    num_train = Xtrain.shape[0]
    onehot_train_labels = np.zeros((num_train, NUM_CLASSES)) # 3 classes to predict
    onehot_train_labels[np.arange(num_train), Ytrain[:, 0].astype(int)] = 1 # performs one hot encoding

    Xtrain = np.vstack((Xtrain[:, 1], Xtrain[:, 9])).T

    return Xtrain, onehot_train_labels
