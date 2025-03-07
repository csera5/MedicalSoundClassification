# Added vowel data (with noise)

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
        XtestIDs = testingData[:, 0]

        onehot_train_labels = np.zeros((Ytrain.shape[0], NUM_CLASSES)) # 3 classes to predict
        onehot_train_labels[np.arange(Ytrain.shape[0]), Ytrain[:, 0].astype(int)] = 1 # performs one hot encoding

    onehot_train_coldpresent = np.zeros((Xtrain.shape[0], 3))
    onehot_train_coldpresent[Xtrain[:, 8] == 0, 0] = 1
    onehot_train_coldpresent[Xtrain[:, 8] == 1, 1] = 1
    onehot_train_coldpresent[np.isnan(Xtrain[:, 8].astype(float)), 2] = 1

    onehot_test_coldpresent = np.zeros((Xtest.shape[0], 3))
    onehot_test_coldpresent[Xtest[:, 8] == 0, 0] = 1
    onehot_test_coldpresent[Xtest[:, 8] == 1, 1] = 1
    onehot_test_coldpresent[np.isnan(Xtest[:, 8].astype(float)), 2] = 1

    newXtrain = np.zeros((0, ENCODING * 2 + 11))
    new_onehot_train_labels = np.zeros((0, 3))
    for i in range(Xtrain.shape[0]):
        try:
            vowel = np.load(f"sounds/sounds/{Xtrain[i, 0]}/vowel-opera.npy")
            cough = np.load(f"sounds/sounds/{Xtrain[i, 0]}/cough-opera.npy")
            newXtrain = np.append(newXtrain, np.concatenate((cough, vowel, np.atleast_2d(Xtrain[i, 1:8]).astype(float), np.atleast_2d(onehot_train_coldpresent[i]), np.atleast_2d(Xtrain[i, 9]).astype(float)), axis=1), axis=0)
            new_onehot_train_labels = np.append(new_onehot_train_labels, np.atleast_2d(onehot_train_labels[i]), axis=0)
        except FileNotFoundError:
            pass

    averageTrainVowel = np.atleast_2d(newXtrain[:, ENCODING:ENCODING * 2].mean(axis=0))

    newXtest = np.zeros((0, ENCODING * 2 + 11))
    for i in range(Xtest.shape[0]):
        try:
            vowel = np.load(f"sounds/sounds/{Xtest[i, 0]}/vowel-opera.npy")
        except FileNotFoundError:
            vowel = averageTrainVowel
        cough = np.load(f"sounds/sounds/{Xtest[i, 0]}/cough-opera.npy")
        newXtest = np.concatenate((newXtest, np.concatenate((cough, vowel, np.atleast_2d(Xtest[i, 1:8]).astype(float), np.atleast_2d(onehot_test_coldpresent[i]), np.atleast_2d(Xtest[i, 9]).astype(float)), axis=1)), axis=0, dtype=float)

    cough_noise = np.random.default_rng().normal(0, 1e-1, (newXtrain.shape[0], ENCODING))
    vowel_noise = np.random.default_rng().normal(0, 1e-2, (newXtrain.shape[0], ENCODING))
    age_noise = np.random.default_rng().normal(0, 1, (newXtrain.shape[0], 1))
    packYears_noise = np.random.default_rng().normal(0, 5, (newXtrain.shape[0], 1))
    # print("coughs", (newXtrain[:, :ENCODING] + cough_noise).shape)
    # print("age", (np.atleast_2d(newXtrain[:, ENCODING]).T + age_noise).shape)
    # print("block", newXtrain[:, ENCODING + 1:-1].shape)
    # print("packYears", (np.atleast_2d(newXtrain[:, -1]).T + packYears_noise).shape)
    newXtrain = np.concatenate((newXtrain, np.concatenate((newXtrain[:, :ENCODING] + cough_noise, newXtrain[:, ENCODING:ENCODING * 2] + vowel_noise, np.atleast_2d(newXtrain[:, ENCODING * 2]).T + age_noise, newXtrain[:, ENCODING * 2 + 1:-1], np.atleast_2d(newXtrain[:, -1]).T + packYears_noise), axis=1)), axis=0)
    new_onehot_train_labels = np.tile(new_onehot_train_labels, (2, 1))

    print(newXtrain.shape)
    print(new_onehot_train_labels.shape)
    print(newXtest.shape)
    if testing:
        print(onehot_test_labels.shape)
    else:
        print(XtestIDs.shape)

    if testing:
        return newXtrain, new_onehot_train_labels, newXtest, onehot_test_labels
    else:
        return newXtrain, new_onehot_train_labels, newXtest, XtestIDs

load_data(testing=False)