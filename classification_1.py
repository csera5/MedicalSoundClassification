# Just loading cough.npy, no noise

import numpy as np
import pandas

ENCODING = 512
NUM_CLASSES = 3

def load_data(testing=False):
    if testing:
        trainingData = pandas.read_csv("newTrain.csv").to_numpy()
        testingData = pandas.read_csv("newTest.csv").to_numpy()

        Xtrain = trainingData[:, :-1]
        Ytrain = np.atleast_2d(trainingData[:, -1]).T
        Xtest = testingData[:, :-1]
        Ytest = np.atleast_2d(testingData[:, -1]).T

        onehot_train_labels = np.zeros((Ytrain.shape[0], NUM_CLASSES)) # 3 classes to predict
        onehot_train_labels[np.arange(Ytrain.shape[0]), Ytrain[:, 0].astype(int)] = 1 # performs one hot encoding
        onehot_test_labels = np.zeros((Ytest.shape[0], NUM_CLASSES)) # 3 classes to predict
        onehot_test_labels[np.arange(Ytest.shape[0]), Ytest[:, 0].astype(int)] = 1 # performs one hot encoding
    else:
        trainingData = pandas.read_csv("train.csv").to_numpy()
        testingData = pandas.read_csv("test.csv").to_numpy()
        Xtrain = trainingData[:, :-1]
        Ytrain = np.atleast_2d(trainingData[:, -1]).T
        Xtest = testingData
        print("Xtest", Xtest.shape)
        XtestIDs = testingData[:, 0]

        onehot_train_labels = np.zeros((Ytrain.shape[0], NUM_CLASSES)) # 3 classes to predict
        onehot_train_labels[np.arange(Ytrain.shape[0]), Ytrain[:, 0].astype(int)] = 1 # performs one hot encoding

    newXtrain = np.zeros((0, ENCODING + 8))
    for i in range(Xtrain.shape[0]):
        cough = np.load(f"sounds/sounds/{Xtrain[i, 0]}/cough-opera.npy")
        newXtrain = np.concatenate((newXtrain, np.concatenate((cough, np.atleast_2d(Xtrain[i, 1:8]).astype(float), np.atleast_2d(Xtrain[i, 9]).astype(float)), axis=1)), axis=0, dtype=float)

    newXtest = np.zeros((0, ENCODING + 8))
    for i in range(Xtest.shape[0]):
        cough = np.load(f"sounds/sounds/{Xtest[i, 0]}/cough-opera.npy")
        newXtest = np.concatenate((newXtest, np.concatenate((cough, np.atleast_2d(Xtest[i, 1:8]).astype(float), np.atleast_2d(Xtest[i, 9]).astype(float)), axis=1)), axis=0, dtype=float)

    print(newXtrain.shape)
    print(onehot_train_labels.shape)
    print(newXtest.shape)
    if testing:
        print(onehot_test_labels.shape)
    else:
        print(XtestIDs.shape)

    if testing:
        return newXtrain, onehot_train_labels, newXtest, onehot_test_labels
    else:
        return newXtrain, onehot_train_labels, newXtest, XtestIDs

load_data(testing=True)