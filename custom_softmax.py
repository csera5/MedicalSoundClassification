import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from classification_5 import load_data

TESTING = False

def softmaxRegression (trainingImages, trainingLabels, testingImages, testingLabels, epsilon, batchSize, alpha):    
    w = 0.0001 * np.random.randn(trainingImages.shape[1], trainingLabels.shape[1]) # initialzie w to random numbers
    print("Shape of w: ")
    print(w.shape)
    num_samples = trainingImages.shape[0]
    shuffled_indices = np.random.permutation(num_samples)
    train_image_shuffled = trainingImages[shuffled_indices]
    train_label_shuffled = trainingLabels[shuffled_indices]
    for i in range(250): # num epochs
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
    Xtrain, Ytrain, Xtest, Ytest = load_data(testing=TESTING)
    print(f"Xtrain: {Xtrain.shape}")
    print(f"Xtest: {Xtest.shape}")

    # Append a constant 1 term to each example to correspond to the bias terms
    training_ones = np.ones((1, Xtrain.shape[0]))
    Xtilde_train = np.vstack((Xtrain.T, training_ones)).T
    testing_ones = np.ones((1, Xtest.shape[0]))
    Xtilde_test = np.vstack((Xtest.T, testing_ones)).T

    Xtrain = Xtilde_train
    Xtest = Xtilde_test
    print(f"Training images shape: {Xtrain.shape}")
    print(f"Testing images shape: {Xtest.shape}")

    # Train the model
    Wtilde = softmaxRegression(Xtrain, Ytrain, Xtest, Ytrain, epsilon=0.001, batchSize=64, alpha=.2)

    z = np.dot(Xtest, Wtilde)
    z_train = np.dot(Xtrain, Wtilde)

    yhat = np.exp(z)
    yhat_train = np.exp(z_train)

    yhat = yhat / (np.sum(yhat, axis = 1, keepdims=True))
    yhat_onehot = np.zeros_like(yhat)
    yhat_onehot[np.arange(len(yhat)), yhat.argmax(1)] = 1

    yhat_train = yhat_train / (np.sum(yhat_train, axis = 1, keepdims=True))
    yhat_train_onehot = np.zeros_like(yhat_train)
    yhat_train_onehot[np.arange(len(yhat_train)), yhat_train.argmax(1)] = 1

    accuracy = np.sum(np.all(Ytrain == yhat_train_onehot, axis = 1)) / len(Ytrain)
    print(f"Training: {accuracy}")

    if TESTING:
        accuracy = np.sum(np.all(Ytest == yhat_onehot, axis = 1)) / len(Ytest)
        print(f"Testing: {accuracy}")
        
        yhat_baseline = np.zeros_like(Ytest)
        yhat_baseline[:, Ytest.sum(axis=0).argmax(axis=0)] = 1
        accuracy = np.sum(np.all(Ytest == yhat_baseline, axis=1)) / len(Ytest)
        print(f"Baseline: {accuracy}")
    else:
        disease = yhat_onehot.argmax(axis=1)
        df = pd.DataFrame({'candidateID': Ytest, 'disease': disease})
        df.to_csv('submission.csv', index = False) # write to csv file