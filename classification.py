import numpy as np
# import scipy
import pandas

ENCODING = 512
NUM_CLASSES = 3

def load_data():
    trainingData = pandas.read_csv("train.csv")
    testingData = pandas.read_csv("test.csv")

    Xtrain = trainingData.to_numpy()[:, :-1] # ignores labels in last column
    Xtest = testingData.to_numpy()
    XtestIDs = Xtest[:, 0]
    Ytrain = np.atleast_2d(trainingData.to_numpy()[:, -1]).T # grabs labels from last column
    Ytrain = Ytrain.reshape(Ytrain.shape[0], 1) # makes shape a 2d array, easier for later

    num_train = Xtrain.shape[0]
    num_test = Xtest.shape[0]
    train_coughs = np.zeros((num_train, ENCODING))
    test_coughs = np.zeros((num_test, ENCODING))
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
    for i in range(num_train):
        train_coughs[i] = np.load(f"sounds/sounds/{candidateIds[i]}/cough-opera.npy") # loads cough data for each participant

    candidateIds = Xtest[:, 0]
    for i in range(num_test):
        test_coughs[i] = np.load(f"sounds/sounds/{candidateIds[i]}/cough-opera.npy") # loads cough data for test participants

    Xtrain = np.concatenate((train_coughs, Xtrain[:, 1:8].astype(float), onehot_train_coldpresent, np.atleast_2d(Xtrain[:, 9]).T.astype(float)), axis=1, dtype=float) # adds coughs to Xtrain array
    Xtest = np.concatenate((test_coughs, Xtest[:, 1:8].astype(float), onehot_test_coldpresent, np.atleast_2d(Xtest[:, 9]).T.astype(float)), axis=1, dtype=float) # adds coughs to Ytest array

    cough_noise = np.random.default_rng().normal(0, 1e-1, train_coughs.shape)
    age_noise = np.random.default_rng().normal(0, 1, Xtrain[:, 1].shape)
    Xtrain = np.vstack((Xtrain, np.append(np.fliplr(Xtrain[:, :512]), Xtrain[:, 512:], axis=1), np.concatenate((Xtrain[:, :512] + cough_noise, np.atleast_2d(Xtrain[:, 512] + age_noise).T, Xtrain[:, 513:]), axis=1)))
    onehot_train_labels = np.tile(onehot_train_labels, (3, 1))
    print(Xtrain.shape)

    return Xtrain, onehot_train_labels, Xtest, XtestIDs

load_data()