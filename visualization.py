# import os
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz.12.2.1-win64/bin/'
# # cry no work

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SimpleNet(nn.Module):
#     def __init__(self):
#         super(SimpleNet, self).__init__()
#         self.fc1 = nn.Linear(784, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 10)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# dummy_input = torch.randn(1, 784)  # Batch size of 1, 784 input features

# from torchviz import make_dot

# model = SimpleNet()
# output = model(dummy_input)
# dot = make_dot(output, params=dict(model.named_parameters()))

# # Save or display the generated graph
# dot.format = 'png'
# dot.render('simple_net')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from classification import load_data

def show_PCA(X_train, Y_train):

    # Standardize the Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply PCA
    pca = PCA(n_components=X_train.shape[1])  # Reduce to N principal components
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Extract first two principal components
    pc1 = X_train_pca[:, 0]
    pc2 = X_train_pca[:, 1]
    pc3 = X_train_pca[:, 3]

    # Scatter plot of the first two principal components
    plt.figure(figsize=(8, 6))
    plt.scatter(pc1, pc2, c=np.argmax(Y_train, axis=1), cmap='viridis', alpha=0.5)
    plt.colorbar(label="Class Labels")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    # plt.zlabel("Principal Component 3")
    plt.title("PCA Projection of Training Data")
    plt.show()

allTrainX, allTrainY, _, _ = load_data()
allTrainX = np.delete(allTrainX, 519, axis=1) # deleting column 519

show_PCA(allTrainX, allTrainY)