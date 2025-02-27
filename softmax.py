import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from classification import load_data


def softmaxRegression (trainingImages, trainingLabels, testingImages, testingLabels, epsilon, batchSize, alpha):    
    print("Shape of training images: ")
    print(trainingImages.shape)
    print("Shape of training labels: ")
    print(trainingLabels.shape)
    print("Shape of testing images: ")
    print(testingImages.shape)
    w = 0.00001 * np.random.randn(trainingImages.shape[1], trainingLabels.shape[1]) # initialzie w to random numbers
    print("Shape of w: ")
    print(w.shape)
    num_samples = trainingImages.shape[0]
    shuffled_indices = np.random.permutation(num_samples)
    train_image_shuffled = trainingImages[shuffled_indices]
    train_label_shuffled = trainingLabels[shuffled_indices]
    for i in range(15): # num epochs
        count  = 0
        print(i)
        for k in range(0, num_samples, batchSize):
            image_batch = train_image_shuffled[k:k+batchSize]
            label_batch = train_label_shuffled[k:k+batchSize]
            z = np.dot(image_batch, w)
            yhat = np.exp(z)
            yhat = yhat / (np.sum(yhat, axis = 1, keepdims=True))
            grad = (image_batch.T @ (yhat-label_batch))/batchSize
            L2 = (alpha/batchSize) * w
            L2[-1, :] = 0
            grad = grad + L2
            w = w - (epsilon * grad)  # update w

         
    print("Shape of gradient")
    print(grad.shape)
    return w


def visualize_weights(w, title):
    for i in range(w.shape[1]):
        weight_image = w[:-1, i].reshape(28, 28)
        plt.imshow(weight_image, cmap='plasma')
        plt.title(f"{title} of Column {i}")
        plt.colorbar()
        plt.show()


   
if __name__ == "__main__":    
    trainingImages, trainingLabels, testingImages = load_data()
    print(f"Very beginning train {trainingImages.shape}")
    print(f"Very beginning test {testingImages.shape}")
    # Append a constant 1 term to each example to correspond to the bias terms
    training_ones = np.ones((1,trainingImages.shape[0]))

    Xtilde_train = np.vstack((trainingImages.T, training_ones)).T
    print(f"Shape of training images before deleting: {Xtilde_train.shape}")
    Xtilde_train = np.delete(Xtilde_train, 519, axis=1) # deleting column 519
    Xtilde_train = Xtilde_train
    trainingLabels = trainingLabels

    testing_ones = np.ones((1,testingImages.shape[0]))
    Xtilde_test = np.vstack((testingImages.T, testing_ones)).T
    print(f"Shape of testing images before deleting: {Xtilde_test.shape}")
    Xtilde_test = np.delete(Xtilde_test, 519, axis=1)# deleted column 519 

    trainingImages = Xtilde_train
    testingImages = Xtilde_test


    # Train the model
    Wtilde = softmaxRegression(trainingImages, trainingLabels, testingImages, trainingLabels, epsilon=0.1, batchSize=100, alpha=.1)
    print(f"Training images shape:{trainingImages.shape}")
    print(f"Testing images shape:{testingImages.shape}")

    z = np.dot(testingImages, Wtilde)
    yhat = np.exp(z)
    yhat = yhat / (np.sum(yhat, axis = 1, keepdims=True))
    yhat_onehot = np.zeros_like(yhat)
    yhat_onehot[np.arange(len(yhat)), yhat.argmax(1)] = 1
    print(all(yhat_onehot[:,0] == 1))
