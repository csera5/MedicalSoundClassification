import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import seaborn as sns
import pandas as pd
from vowel_generation import start

def load_data():
    trainingData = pd.read_csv("train.csv").to_numpy()
    testingData = pd.read_csv("test.csv").to_numpy()
    Xtrain = trainingData[:, :-1]
    Ytrain = np.atleast_2d(trainingData[:, -1]).T
    Xtest = testingData

    onehot_train_labels = np.zeros((Ytrain.shape[0], 3)) # 3 classes to predict
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

    print(Xtrain.shape)
    print(Xtest.shape)
    print(trainIds.shape)
    print(testIds.shape)
    print()
    start(Xtrain, trainIds, Xtest, testIds)

    newXtrain = np.zeros((0, 512 * 2 + 11))
    new_onehot_train_labels = np.zeros((0, 3))
    for i in range(Xtrain.shape[0]):
        try:
            vowel = np.load(f"sounds/sounds/{trainIds[i]}/vowel-opera.npy")
        except FileNotFoundError:
            vowel = np.load(f"newSounds/{trainIds[i]}/vowel-opera.npy")
        cough = np.load(f"sounds/sounds/{trainIds[i]}/cough-opera.npy")
        newXtrain = np.append(newXtrain, np.concatenate((cough, np.atleast_2d(vowel), np.atleast_2d(Xtrain[i])), axis=1), axis=0)
        new_onehot_train_labels = np.append(new_onehot_train_labels, np.atleast_2d(onehot_train_labels[i]), axis=0)

    print(newXtrain.shape)
    print(new_onehot_train_labels.shape)

    return newXtrain, new_onehot_train_labels

def show_continuous_2D_PCA(X_train, Y_train):

    # Standardize the Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply PCA
    pca = PCA(n_components=X_train.shape[1])  # Reduce to N principal components
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Extract first two principal components
    pc1 = X_train_pca[:, 0]
    pc2 = X_train_pca[:, 1]

    # Scatter plot of the first two principal components
    fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111)

    sns.scatterplot(x=pc1, y=pc2, hue=np.argmax(Y_train, axis=1), palette="Set1", s=100)
    # plt.colorbar(label="Class Labels")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Projection of Age and Smoke Count Training Data")
    plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


def show_continuous_PCA(X_train, Y_train, title="PCA Projection of Continuous Training Data"):

    # Standardize the Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply PCA
    pca = PCA(n_components=min(X_train.shape[0], X_train.shape[1]))  # Reduce to N principal components
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Extract first two principal components
    pc1 = X_train_pca[:, 0]
    pc2 = X_train_pca[:, 1]
    pc3 = X_train_pca[:, 2]

    # Scatter plot of the first two principal components
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pc1, pc2, pc3, c=np.argmax(Y_train, axis=1), cmap="Set1", alpha=0.5)
    # plt.colorbar(label="Class Labels")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    ax.set_title(title)
    plt.show()

def show_ordinal_PCA(X_train, Y_train):

    # Normalize the Data
    encoder = OrdinalEncoder()
    X_train_scaled = encoder.fit_transform(X_train)

    # Apply PCA
    pca = PCA(n_components=X_train.shape[1])  # Reduce to N principal components
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Extract first two principal components
    pc1 = X_train_pca[:, 0]
    pc2 = X_train_pca[:, 1]
    pc3 = X_train_pca[:, 2]

    # Scatter plot of the first two principal components
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pc1, pc2, pc3, c=np.argmax(Y_train, axis=1), cmap="Set1")
    # plt.colorbar(label="Class Labels")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    ax.set_title("PCA Projection of Ordinal Training Data")
    plt.show()

allTrainX, allTrainY = load_data()
print("allTrainX", allTrainX.shape)
print("allTrainY", allTrainY.shape)

ordTrainX = allTrainX[:, 2 * 512 + 1:-1]
print("ordTrainX", ordTrainX.shape)
show_ordinal_PCA(ordTrainX, allTrainY)

contTrainX = np.concatenate((allTrainX[:, :512 * 2 + 1], np.atleast_2d(allTrainX[:, -1]).T), axis=1)
print("contTrainX", contTrainX.shape)
show_continuous_PCA(contTrainX, allTrainY)

contTrainX = np.concatenate((np.atleast_2d(allTrainX[:, 512 * 2]).T, np.atleast_2d(allTrainX[:, -1]).T), axis=1)
show_continuous_2D_PCA(contTrainX, allTrainY)

# allTrainX, allTrainY, _, _ = load_sound_data()
allTrainX = pd.read_csv("spectrograph-X.csv").to_numpy().reshape(546, 137)
allTrainY = pd.read_csv("spectrograph-Y.csv").to_numpy().reshape(546, 3)
print(allTrainX.shape, allTrainY.shape)

# df = pd.DataFrame({
#     'allTrainX': allTrainX.reshape(-1),
# })
# df.to_csv('spectrograph-X.csv', index=False)
# print("Spectrogram X Data saved to spectrograph-X.csv")

# df = pd.DataFrame({
#     'allTrainY': allTrainY.reshape(-1),
# })
# df.to_csv('spectrograph-Y.csv', index=False)
# print("Spectrogram Y Data saved to spectrograph-Y.csv")

allTrainX = np.nan_to_num(allTrainX)
allTrainY = np.nan_to_num(allTrainY)

show_continuous_PCA(allTrainX, allTrainY, title="PCA Projection of Spectrograph Training Data")
