import numpy as np
import scipy
import pandas

ENCODING = 512

def load_data():
    trainingData = pandas.read_csv("train.csv")
    testingData = pandas.read_csv("test.csv")

    Xtrain = trainingData.to_numpy()[:, :-1] # ignores labels in last column
    Xtest = testingData.to_numpy()
    Ytrain = np.atleast_2d(trainingData.to_numpy()[:, -1]).T # grabs labels from last column
    Ytrain = Ytrain.reshape(Ytrain.shape[0] , 1) # makes shape a 2d array, easier for later

    num_train = Xtrain.shape[0]
    num_test = Xtest.shape[0]
    train_coughs = np.zeros((num_train, ENCODING))
    test_coughs = np.zeros((num_test, ENCODING))
    onehot_train_labels = np.zeros((num_train, 3)) # 3 classes to predict
    onehot_train_labels[np.arange(num_train), Ytrain[:, 0].astype(int)] = 1 # performs one hot encoding

    candidateIds = Xtrain[:, 0]
    for i in range(num_train):
        train_coughs[i] = np.load(f"sounds/sounds/{candidateIds[i]}/cough-opera.npy") # loads cough data for each participant

    candidateIds = Xtest[:, 0]
    for i in range(num_test):
        test_coughs[i] = np.load(f"sounds/sounds/{candidateIds[i]}/cough-opera.npy") # loads cough data for test participants

    Xtrain = np.append(train_coughs, Xtrain[:, 1:].astype(float), axis=1) # adds coughs to Xtrain array
    Xtest = np.append(test_coughs, Xtest[:, 1:].astype(float), axis=1) # adds coughs to Ytest array
    return Xtrain, onehot_train_labels, Xtest

load_data()