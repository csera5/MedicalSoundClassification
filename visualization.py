import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from ellys_data_loader_2 import load_data
from ordinal_data import load_ordinal_data
from continuous_data import load_continuous_data
from continuous_not_audio_data import load_continuous_no_audio_data
# from spectrograph import load_sound_data
import seaborn as sns
import pandas as pd

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
    pca = PCA(n_components=X_train.shape[1])  # Reduce to N principal components
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

allTrainX, allTrainY, _, _ = load_data()

allTrainX, allTrainY = load_ordinal_data()
show_ordinal_PCA(allTrainX, allTrainY)

allTrainX, allTrainY = load_continuous_data()
show_continuous_PCA(allTrainX, allTrainY)

allTrainX, allTrainY = load_continuous_no_audio_data()
show_continuous_2D_PCA(allTrainX, allTrainY)

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

# allTrainX = np.nan_to_num(allTrainX)
# allTrainY = np.nan_to_num(allTrainY)

show_continuous_PCA(allTrainX, allTrainY, title="PCA Projection of Spectrograph Training Data")
