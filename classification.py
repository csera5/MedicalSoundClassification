import numpy as np
import scipy
import pandas

ENCODING = 512

def load_data():
    trainingData = pandas.read_csv("train.csv")
    testingData = pandas.read_csv("test.csv")

    Xtrain = trainingData.to_numpy()[:, :-1] # ignores labels in last column
    Ytest = testingData.to_numpy()
    Xlabel = trainingData.to_numpy()[:, -1] # grabs labels from last column
    Xlabel = Xlabel.reshape(Xlabel.shape[0],1) # makes shape a 2d array, easier for later
    print("Num test candidates: ")
    print(Ytest.shape[0])
    candidateIds = Xtrain[:, 0]
    print("Num train candidates: ")
    print(candidateIds.shape[0])
    num_train = candidateIds.shape[0]
    num_test = Ytest.shape[0]
    train_coughs = np.zeros((num_train, ENCODING))
    test_coughs = np.zeros((num_test, ENCODING))
    vowels = np.zeros((num_train, ENCODING))
    onehot_train_labels = np.zeros((Xtrain.shape[0], 3)) # 3 classes to predict
    onehot_train_labels[np.arange(Xtrain.shape[0]),Xlabel[:,0].astype(int)] = 1 # performs one hot encoding 

    for i in range(num_train):
        # if candidateIds[i] == "d7ed7deb786c3": # ask ellys what this is
        #     continue
        train_coughs[i] = np.load(f"sounds/sounds/{candidateIds[i]}/cough-opera.npy") # loads cough data for each participant
        # vowels[i] = np.load(f"sounds/sounds/{candidateIds[i]}/vowel-opera.npy")
        # print(candidateIds[i])
    print("Shape of train coughs: ")
    print(train_coughs.shape)
    print("Shape of train labels (one-hot encoded): ")
    print(onehot_train_labels.shape) # 3 classes for 3 diseases

    for i in range(num_test):
        test_coughs[i] = np.load(f"sounds/sounds/{candidateIds[i]}/cough-opera.npy") # loads cough data for test participants

    print("Shape of test coughs: ")
    print(test_coughs.shape)
    


    # X = np.append(coughs, np.append(vowels, X[:, 1:], axis=1), axis=1)
    Xtrain = np.append(train_coughs, Xtrain[:, 1:].astype(float), axis=1) # adds coughs to Xtrain array
    print(f"Y test columns: {Ytest[:, 1:].shape}")
    Ytest = np.append(test_coughs, Ytest[:, 1:].astype(float), axis=1) # adds coughs to Ytest array
    print(onehot_train_labels.shape)
    # print(onehot_train_labels[0:10])
    return Xtrain, onehot_train_labels, Ytest

load_data()

