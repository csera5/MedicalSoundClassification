import numpy as np
import pandas

ENCODING = 512
NUM_CLASSES = 3

def load_vowel_data():
    trainingData = pandas.read_csv("train.csv")

    # grab the training data from CSV
    Xtrain = trainingData.to_numpy()[:, :-1] # ignores labels in last column
    total_count = Xtrain.shape[0]

    # grab the XtrainOutput from the CSV and convert to onehot
    XtrainOutput = np.atleast_2d(trainingData.to_numpy()[:, -1]).T # grabs labels from last column
    XtrainOutput = XtrainOutput.reshape(XtrainOutput.shape[0] , 1) # makes shape a 2d array, easier for later
    onehot_train_labels = np.zeros((total_count, NUM_CLASSES)) # 3 classes to predict
    onehot_train_labels[np.arange(total_count), XtrainOutput[:, 0].astype(int)] = 1 # performs one hot encoding

    # append Xtrain and XtrainOutput together
    Xtrain = np.hstack((Xtrain, onehot_train_labels))

    coughs = np.zeros((total_count, ENCODING))
    candidateIds = Xtrain[:, 0]
    for i in range(total_count):
        coughs[i] = np.load(f"sounds/sounds/{candidateIds[i]}/cough-opera.npy") # loads cough data for each participant

    # break out Ytrain and Xtest
    vowels = np.zeros((total_count, ENCODING))
    trainIndices = np.zeros(total_count)
    testIndices = np.zeros(total_count)
    for i in range(total_count):
        try:
            vowels[i] = np.load(f"sounds/sounds/{candidateIds[i]}/vowel-opera.npy") # loads vowel data for each participant
            trainIndices[i] = 1
        except FileNotFoundError:
            testIndices[i] = 1

    testIndices = np.where(testIndices == 1)
    trainIndices = np.where(trainIndices == 1)

    train_vowels = vowels[trainIndices]
    train_coughs = coughs[trainIndices]
    test_coughs = coughs[testIndices]

    Xtest = Xtrain[testIndices]
    Xtrain = Xtrain[trainIndices]
    
    Ytrain = vowels[testIndices]

    Xtrain = np.append(train_coughs, Xtrain[:, 1:].astype(float), axis=1) # adds coughs to Xtrain array, remove candidateIDs
    
    Xtest = np.append(test_coughs, Xtest[:, 1:].astype(float), axis=1) # adds coughs to Xtrain array, remove candidateIDs

    return Xtrain, Ytrain, Xtest

load_vowel_data()