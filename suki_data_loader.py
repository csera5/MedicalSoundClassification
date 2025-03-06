import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas 
import numpy as np 
import json 
from sklearn.preprocessing import StandardScaler

NUM_CLASSES = 3
ENCODING = 512
def load_data():
    trainingData = pandas.read_csv("train.csv")
    testingData = pandas.read_csv("test.csv")

    Xtrain = trainingData.to_numpy()[:, :-1]  # ignores labels in last column
    Xtest = testingData.to_numpy()
    XtestIDs = Xtest[:, 0]
    Ytrain = np.atleast_2d(trainingData.to_numpy()[:, -1]).T  # grabs labels from last column
    Ytrain = Ytrain.reshape(Ytrain.shape[0], 1)  # makes shape a 2d array, easier for later

    num_train = Xtrain.shape[0]
    num_test = Xtest.shape[0]
    onehot_train_labels = np.zeros((num_train, NUM_CLASSES))  # 3 classes to predict
    onehot_train_labels[np.arange(num_train), Ytrain[:, 0].astype(int)] = 1  # performs one hot encoding

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

    train_coughs = np.zeros((num_train, ENCODING))
    test_coughs = np.zeros((num_test, ENCODING))
    train_vowels = np.zeros((num_train, ENCODING))
    test_vowels = np.zeros((num_test, ENCODING))
    newXtrain = np.zeros((0, 523))
    new_onehot_train_labels = np.zeros((0, 3))
    for i in range(num_train):
        try:
            with open(f"sounds/sounds/{candidateIds[i]}/emb_cough.json") as f:
                coughs = np.append(np.array(json.load(f)), np.load(f"sounds/sounds/{candidateIds[i]}/cough-opera.npy"), axis=0)
                newXtrain = np.concatenate((newXtrain, np.concatenate((coughs, np.tile(Xtrain[i], (coughs.shape[0], 1))), axis=1)), axis=0)
                new_onehot_train_labels = np.append(new_onehot_train_labels, np.tile(onehot_train_labels[i], (coughs.shape[0], 1)), axis=0)
        except:
            continue

    candidateIds = Xtest[:, 0]
    Xtest = np.concatenate((Xtest[:, 1:8].astype(float), onehot_test_coldpresent, Xtest[:, 9].reshape(num_test, 1).astype(float)), axis=1, dtype=float)
    newXtest = np.zeros((0, 523))
    for i in range(num_test):
        try:
            with open(f"sounds/sounds/{candidateIds[i]}/emb_cough.json") as f:
                coughs = np.array(json.load(f))
                newXtest = np.append(newXtest, np.concatenate((coughs.mean(axis=1), Xtest[i]), axis=1), axis=0)
        except:
            cough = np.load(f"sounds/sounds/{candidateIds[i]}/cough-opera.npy")
            newXtest = np.append(newXtest, np.concatenate((cough, Xtest[i].reshape((1, Xtest.shape[1]))), axis=1), axis=0)

    cough_noise = np.random.default_rng().normal(0, 1e-1, (newXtrain.shape[0], ENCODING))
    age_noise = np.random.default_rng().normal(0, 1, newXtrain[:, ENCODING].shape)
    newXtrain = np.vstack((newXtrain, np.concatenate((newXtrain[:, :ENCODING] + cough_noise, np.atleast_2d(newXtrain[:, ENCODING] + age_noise).T, newXtrain[:, ENCODING + 1:]), axis=1)))

    new_onehot_train_labels = np.tile(new_onehot_train_labels, (2, 1))

    scaler = StandardScaler()
    newXtrain_scaled = scaler.fit_transform(newXtrain)  # Standardizing training data
    newXtest_scaled = scaler.transform(newXtest)  # Standardizing test data

    # PCA
    pca = PCA(n_components=2) 
    Xtrain_pca = pca.fit_transform(newXtrain_scaled[:,])  # Fit and transform training data
    Xtest_pca = pca.transform(newXtest_scaled)  # Use learned components for new dataset
    # Plot
    plt.figure(figsize=(10, 8))

    # Scatter plot of training data
    sns.scatterplot(x=Xtrain_pca[:, 0], y=Xtrain_pca[:, 1], hue=np.argmax(new_onehot_train_labels, axis=1), palette="Set1", s=100, edgecolor="k", alpha=0.7)

    # Title and labels
    plt.title("PCA of Training Data", fontsize=16)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    plt.show()

    print("PCA Applied and Visualization Done")

    return Xtrain_pca, new_onehot_train_labels, Xtest_pca, XtestIDs

# see pretty photo
load_data()
