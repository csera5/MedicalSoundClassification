import numpy as np
# import scipy
import pandas
import json

ENCODING = 512
NUM_CLASSES = 3

def load_data(testing=False):
    trainingData = pandas.read_csv("train.csv")
    testingData = pandas.read_csv("test.csv")
    if testing:
        X = np.random.default_rng().permutation(trainingData.to_numpy(), axis=0)
        cutoff = int(X.shape[0] * 0.8)
        Xtrain = X[:cutoff, :-1]
        Ytrain = np.atleast_2d(X[:cutoff, -1]).T
        Xtest = X[cutoff:, :-1]
        Ytest = np.atleast_2d(X[cutoff:, -1]).T
        onehot_test_labels = np.zeros((Xtest.shape[0], NUM_CLASSES)) # 3 classes to predict
        onehot_test_labels[np.arange(Xtest.shape[0]), Ytest[:, 0].astype(int)] = 1 # performs one hot encoding
    else:
        Xtrain = trainingData.to_numpy()[:, :-1] # ignores labels in last column
        Ytrain = np.atleast_2d(trainingData.to_numpy()[:, -1]).T # grabs labels from last column
        Xtest = testingData.to_numpy()
        XtestIDs = Xtest[:, 0]

    num_train = Xtrain.shape[0]
    num_test = Xtest.shape[0]
    onehot_train_labels = np.zeros((num_train, NUM_CLASSES)) # 3 classes to predict
    onehot_train_labels[np.arange(num_train), Ytrain[:, 0].astype(int)] = 1 # performs one hot encoding

    onehot_train_coldpresent = np.zeros((num_train, 3))
    onehot_train_coldpresent[Xtrain[:, 8] == 0, 0] = 1
    onehot_train_coldpresent[Xtrain[:, 8] == 1, 1] = 1
    onehot_train_coldpresent[np.isnan(Xtrain[:, 8].astype(float)), 2] = 1
    onehot_test_coldpresent = np.zeros((num_test, 3))
    onehot_test_coldpresent[Xtest[:, 8] == 0, 0] = 1
    onehot_test_coldpresent[Xtest[:, 8] == 1, 1] = 1
    onehot_test_coldpresent[np.isnan(Xtest[:, 8].astype(float)), 2] = 1

    candidateIds = Xtrain[:, 0]
    Xtrain = np.concatenate((Xtrain[:, 1:8].astype(float), onehot_train_coldpresent, Xtrain[:, 9].reshape(num_train, 1).astype(float)), axis=1, dtype=float)

    # train_coughs = np.zeros((num_train, ENCODING))
    # test_coughs = np.zeros((num_test, ENCODING))
    # train_vowels = np.zeros((num_train, ENCODING))
    # test_vowels = np.zeros((num_test, ENCODING))
    newXtrain = np.zeros((0, ENCODING * 2 + 11))
    new_onehot_train_labels = np.zeros((0, 3))
    for i in range(num_train):
        # try:
            # train_vowels[i] = np.load(f"sounds/sounds/{candidateIds[i]}/vowel-opera.npy")
            # train_coughs[i] = np.load(f"sounds/sounds/{candidateIds[i]}/cough-opera.npy") # loads cough data for each participant
        # except:
        #     print(candidateIds[i])
        #     np.delete(Xtrain, i, axis=0)
        #     np.delete(train_vowels, i, axis=0)
        #     np.delete(train_coughs, i, axis=0)
        
        # vowels = np.zeros((0, ENCODING))
        vowel = np.zeros((0, ENCODING))
        # try:
        #     with open(f"sounds/sounds/{candidateIds[i]}/emb_vowel.json") as f:
        #         vowels = np.array(json.load(f))
        # except:
        #     pass
        try:
            vowel = np.load(f"sounds/sounds/{candidateIds[i]}/vowel-opera.npy")
        except:
            vowel = np.load(f"newSounds/{candidateIds[i]}/vowel-opera.npy")
        # vowels = np.append(vowels, vowel, axis=0)
        # if vowels.shape[0] == 0:
        #     continue

        coughs = np.zeros((0, ENCODING))
        cough = np.zeros((0, ENCODING))
        try:
            with open(f"sounds/sounds/{candidateIds[i]}/emb_cough.json") as f:
                coughs = np.array(json.load(f))
        except:
            pass
        cough = np.load(f"sounds/sounds/{candidateIds[i]}/cough-opera.npy")
        coughs = np.append(coughs, cough, axis=0)

        newXtrain = np.concatenate((newXtrain, np.concatenate((coughs, np.tile(vowel, (coughs.shape[0], 1)), np.tile(Xtrain[i], (coughs.shape[0], 1))), axis=1)), axis=0)
        new_onehot_train_labels = np.append(new_onehot_train_labels, np.tile(onehot_train_labels[i], (coughs.shape[0], 1)), axis=0)

        # try:
        #     with open(f"sounds/sounds/{candidateIds[i]}/emb_cough.json") as f:
        #         coughs = np.append(np.array(json.load(f)), np.load(f"sounds/sounds/{candidateIds[i]}/cough-opera.npy"), axis=0)
        #         newXtrain = np.concatenate((newXtrain, np.concatenate((coughs, np.tile(Xtrain[i], (coughs.shape[0], 1))), axis=1)), axis=0)
        #         new_onehot_train_labels = np.append(new_onehot_train_labels, np.tile(onehot_train_labels[i], (coughs.shape[0], 1)), axis=0)
        # except:
        #     continue

    candidateIds = Xtest[:, 0]
    Xtest = np.concatenate((Xtest[:, 1:8].astype(float), onehot_test_coldpresent, Xtest[:, 9].reshape(num_test, 1).astype(float)), axis=1, dtype=float)
    newXtest = np.zeros((0, ENCODING * 2 + 11))
    for i in range(num_test):
        # try:
        #     test_vowels[i] = np.load(f"sounds/sounds/{candidateIds[i]}/vowel-opera.npy")
        # except:
        #     print(i, candidateIds[i])
        # test_coughs[i] = np.load(f"sounds/sounds/{candidateIds[i]}/cough-opera.npy") # loads cough data for each participant
        vowel = np.zeros((0, ENCODING))
        try:
            vowel = np.load(f"sounds/sounds/{candidateIds[i]}/vowel-opera.npy")
        except:
            vowel = np.load(f"newSounds/{candidateIds[i]}/vowel-opera.npy")
        try:
            with open(f"sounds/sounds/{candidateIds[i]}/emb_cough.json") as f:
                coughs = np.array(json.load(f))
                newXtest = np.append(newXtest, np.concatenate((coughs.mean(axis=1), np.atleast_2d(vowel), np.atleast_2d(Xtest[i])), axis=1), axis=0)
        except:
            cough = np.load(f"sounds/sounds/{candidateIds[i]}/cough-opera.npy")
            newXtest = np.append(newXtest, np.concatenate((cough, np.atleast_2d(vowel), np.atleast_2d(Xtest[i])), axis=1), axis=0)

    # all_vowels = np.vstack((train_vowels, test_vowels[test_vowels.sum(axis=1) != 0]))
    # Xall = np.vstack((Xtrain, Xtest[test_vowels.sum(axis=1) != 0]))
    # print(train_vowels[np.logical_and(Xtrain[:, 5] == 1, Xtrain[:, 1] < Xtrain[:, 1].mean())].mean(axis=0)[:20])
    # print(all_vowels[np.logical_and(Xall[:, 5] == 1, Xall[:, 1] < Xall[:, 1].mean())].mean(axis=0)[:20])
    # test_vowels[np.logical_and(test_vowels.sum(axis=1) == 0, Xtest[:, 5] == 1)] = all_vowels[np.logical_and(Xall[:, 5] == 1, Xall[:, 1] < Xall[:, 1].mean())].mean(axis=0)
    # test_vowels[np.logical_and(test_vowels.sum(axis=1) == 0, Xtest[:, 5] == 0)] = all_vowels[np.logical_and(Xall[:, 5] == 0, Xall[:, 1] > Xall[:, 1].mean())].mean(axis=0)
    # test_vowels[np.logical_and(test_vowels.sum(axis=1) == 0, Xtest[:, 5] == 1)] = train_vowels[np.logical_and(Xtrain[:, 5] == 1, Xtrain[:, 1] < Xtrain[:, 1].mean())].mean(axis=0)
    # test_vowels[np.logical_and(test_vowels.sum(axis=1) == 0, Xtest[:, 5] == 0)] = train_vowels[np.logical_and(Xtrain[:, 5] == 0, Xtrain[:, 1] > Xtrain[:, 1].mean())].mean(axis=0)

    # Xtrain = np.concatenate((train_coughs, train_vowels, Xtrain[:, 1:8].astype(float), onehot_train_coldpresent, np.atleast_2d(Xtrain[:, 9]).T.astype(float)), axis=1, dtype=float) # adds coughs to Xtrain array
    # Xtest = np.concatenate((test_coughs, test_vowels, Xtest[:, 1:8].astype(float), onehot_test_coldpresent, np.atleast_2d(Xtest[:, 9]).T.astype(float)), axis=1, dtype=float) # adds coughs to Ytest array
    # Xtrain = np.concatenate((train_coughs, Xtrain[:, 1:8].astype(float), onehot_train_coldpresent, np.atleast_2d(Xtrain[:, 9]).T.astype(float)), axis=1, dtype=float) # adds coughs to Xtrain array
    # Xtest = np.concatenate((test_coughs, Xtest[:, 1:8].astype(float), onehot_test_coldpresent, np.atleast_2d(Xtest[:, 9]).T.astype(float)), axis=1, dtype=float) # adds coughs to Ytest array

    cough_noise = np.random.default_rng().normal(0, 1e-1, (newXtrain.shape[0], ENCODING))
    vowel_noise = np.random.default_rng().normal(0, 1e-1, (newXtrain.shape[0], ENCODING))
    age_noise = np.random.default_rng().normal(0, 1, (newXtrain.shape[0], 1))
    packYears_noise = np.random.default_rng().normal(0, 10, (newXtrain.shape[0], 1))
    # Xtrain = np.vstack((Xtrain, np.concatenate((Xtrain[:, :ENCODING] + cough_noise, Xtrain[:, ENCODING:ENCODING * 2] + vowel_noise, np.atleast_2d(Xtrain[:, ENCODING * 2] + age_noise).T, Xtrain[:, (ENCODING * 2) + 1:]), axis=1)))
    newXtrain = np.vstack((newXtrain, np.concatenate((newXtrain[:, :ENCODING] + cough_noise, newXtrain[:, ENCODING:ENCODING * 2] + vowel_noise, np.atleast_2d(newXtrain[:, ENCODING * 2]).T + age_noise, newXtrain[:, ENCODING * 2 + 1:-1], np.atleast_2d(newXtrain[:, -1]).T + packYears_noise), axis=1)))

    new_onehot_train_labels = np.tile(new_onehot_train_labels, (2, 1))
    print()
    print(newXtrain.shape)
    print(newXtest.shape)
    print(new_onehot_train_labels.shape)

    if testing:
        return newXtrain, new_onehot_train_labels, newXtest, onehot_test_labels
    else:
        return newXtrain, new_onehot_train_labels, newXtest, XtestIDs
    
load_data(testing=False)