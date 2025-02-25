import numpy as np
import scipy
import pandas

ENCODING = 512

def softmax():
    trainingData = pandas.read_csv("train.csv")
    X = trainingData.to_numpy()[:, :-1]
    candidateIds = X[:, 0]
    print(candidateIds.shape)
    n = candidateIds.shape[0]
    coughs = np.zeros((n, ENCODING))
    vowels = np.zeros((n, ENCODING))
    for i in range(n):
        if candidateIds[i] == "d7ed7deb786c3":
            continue
        coughs[i] = np.load(f"sounds/sounds/{candidateIds[i]}/cough-opera.npy")
        vowels[i] = np.load(f"sounds/sounds/{candidateIds[i]}/vowel-opera.npy")
        # print(candidateIds[i])
    print(coughs.shape)
    X = np.append(coughs, np.append(vowels, X[:, 1:], axis=1), axis=1)
    print(X.shape)

softmax()