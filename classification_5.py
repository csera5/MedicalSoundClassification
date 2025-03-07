# Added more cough data

import numpy as np
import pandas
import json
from vowel_generation import start

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

    trainIds = Xtrain[:, 0]
    testIds = Xtest[:, 0]
    Xtrain = np.concatenate((Xtrain[:, 1:8].astype(float), onehot_train_coldpresent, np.atleast_2d(Xtrain[:, 9]).T.astype(float)), axis=1)
    Xtest = np.concatenate((Xtest[:, 1:8].astype(float), onehot_test_coldpresent, np.atleast_2d(Xtest[:, 9]).T.astype(float)), axis=1)

    start(Xtrain, trainIds, Xtest, testIds)
    print()

    newXtrain = np.zeros((0, ENCODING * 2 + 11))
    new_onehot_train_labels = np.zeros((0, 3))
    for i in range(Xtrain.shape[0]):
        try:
            vowel = np.load(f"sounds/sounds/{trainIds[i]}/vowel-opera.npy")
        except FileNotFoundError:
            vowel = np.load(f"newSounds/{trainIds[i]}/vowel-opera.npy")

        coughs = np.zeros((0, ENCODING))
        cough = np.zeros((0, ENCODING))
        try:
            with open(f"sounds/sounds/{trainIds[i]}/emb_cough.json") as f:
                coughs = np.array(json.load(f))
        except:
            pass
        cough = np.load(f"sounds/sounds/{trainIds[i]}/cough-opera.npy")
        coughs = np.append(coughs, cough, axis=0)

        # cough = np.load(f"sounds/sounds/{Xtrain[i, 0]}/cough-opera.npy")
        newXtrain = np.append(newXtrain, np.concatenate((coughs, np.tile(vowel, (coughs.shape[0], 1)), np.tile(Xtrain[i], (coughs.shape[0], 1))), axis=1), axis=0)
        new_onehot_train_labels = np.append(new_onehot_train_labels, np.tile(onehot_train_labels[i], (coughs.shape[0], 1)), axis=0)

    newXtest = np.zeros((0, ENCODING * 2 + 11))
    for i in range(Xtest.shape[0]):
        try:
            vowel = np.load(f"sounds/sounds/{testIds[i]}/vowel-opera.npy")
        except FileNotFoundError:
            vowel = np.load(f"newSounds/{testIds[i]}/vowel-opera.npy")
        try:
            with open(f"sounds/sounds/{testIds[i]}/emb_cough.json") as f:
                coughs = np.array(json.load(f))
                newXtest = np.append(newXtest, np.concatenate((coughs.mean(axis=1), np.atleast_2d(vowel), np.atleast_2d(Xtest[i])), axis=1), axis=0)
        except:
            cough = np.load(f"sounds/sounds/{testIds[i]}/cough-opera.npy")
            newXtest = np.append(newXtest, np.concatenate((cough, np.atleast_2d(vowel), np.atleast_2d(Xtest[i])), axis=1), axis=0)

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
        print(testIds.shape)

    if testing:
        return newXtrain, new_onehot_train_labels, newXtest, onehot_test_labels
    else:
        return newXtrain, new_onehot_train_labels, newXtest, testIds
    
if __name__ == "__main__":
    load_data(testing=True)