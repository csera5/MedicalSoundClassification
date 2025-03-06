import numpy as np
# import scipy
import pandas
import json

ENCODING = 512
NUM_CLASSES = 3

def load_continuous_data():
    trainingData = pandas.read_csv("train.csv")

    Xtrain = trainingData.to_numpy()[:, :-1] # ignores labels in last column
    Ytrain = np.atleast_2d(trainingData.to_numpy()[:, -1]).T # grabs labels from last column
    Ytrain = Ytrain.reshape(Ytrain.shape[0], 1) # makes shape a 2d array, easier for later

    num_train = Xtrain.shape[0]
    onehot_train_labels = np.zeros((num_train, NUM_CLASSES)) # 3 classes to predict
    onehot_train_labels[np.arange(num_train), Ytrain[:, 0].astype(int)] = 1 # performs one hot encoding

    onehot_train_coldpresent = np.zeros((num_train, 3))
    onehot_train_coldpresent[Xtrain[:, 8] == 0, 0] = 1
    onehot_train_coldpresent[Xtrain[:, 8] == 1, 1] = 1
    onehot_train_coldpresent[np.isnan(Xtrain[:, 8].astype(float)), 2] = 1

    candidateIds = Xtrain[:, 0]
    Xtrain = np.concatenate((Xtrain[:, 1:8].astype(float), onehot_train_coldpresent, Xtrain[:, 9].reshape(num_train, 1).astype(float)), axis=1, dtype=float)

    train_coughs = np.zeros((num_train, ENCODING))
    train_vowels = np.zeros((num_train, ENCODING))
    newXtrain = np.zeros((0, 523))
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
        try:
            with open(f"sounds/sounds/{candidateIds[i]}/emb_cough.json") as f:
                coughs = np.append(np.array(json.load(f)), np.load(f"sounds/sounds/{candidateIds[i]}/cough-opera.npy"), axis=0)
                newXtrain = np.concatenate((newXtrain, np.concatenate((coughs, np.tile(Xtrain[i], (coughs.shape[0], 1))), axis=1)), axis=0)
                new_onehot_train_labels = np.append(new_onehot_train_labels, np.tile(onehot_train_labels[i], (coughs.shape[0], 1)), axis=0)
        except:
            continue

    cough_noise = np.random.default_rng().normal(0, 1e-1, (newXtrain.shape[0], ENCODING))
    # vowel_noise = np.random.default_rng().normal(0, 1e-2, train_vowels.shape)
    age_noise = np.random.default_rng().normal(0, 1, newXtrain[:, ENCODING].shape)
    # Xtrain = np.vstack((Xtrain, np.concatenate((Xtrain[:, :ENCODING] + cough_noise, Xtrain[:, ENCODING:ENCODING * 2] + vowel_noise, np.atleast_2d(Xtrain[:, ENCODING * 2] + age_noise).T, Xtrain[:, (ENCODING * 2) + 1:]), axis=1)))
    newXtrain = np.vstack((newXtrain, np.concatenate((newXtrain[:, :ENCODING] + cough_noise, np.atleast_2d(newXtrain[:, ENCODING] + age_noise).T, newXtrain[:, ENCODING + 1:]), axis=1)))

    new_onehot_train_labels = np.tile(new_onehot_train_labels, (2, 1))
    print()
    print(newXtrain.shape)
    print(new_onehot_train_labels.shape)

    newXtrain = np.delete(newXtrain, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], axis=1)

    return newXtrain, new_onehot_train_labels
