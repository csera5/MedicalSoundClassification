import numpy as np
import matplotlib.pyplot as plt

########################################################################################################################
# PROBLEM 1
########################################################################################################################
# Given a vector x of (scalar) inputs and associated vector y of the target labels, and given
# degree d of the polynomial, train a polynomial regression model and return the optimal weight vector.
def trainPolynomialRegressor(x, y, d):
    X = np.vander(x, d + 1, True).T
    return np.linalg.solve(X @ X.T, X @ y)

########################################################################################################################
# PROBLEM 2
########################################################################################################################
# Given training and testing data, learning rate epsilon, batch size, and regularization strength alpha,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix Wtilde (785x10).
# Then return Wtilde.
def softmaxRegression(trainingImages, trainingLabels, epsilon, batchSize, alpha):
    return stochasticGradDescent(trainingImages, trainingLabels, epsilon, batchSize, alpha)

def stochasticGradDescent(Xtilde, Y, epsilon, ntilde, alpha):
    Wtilde = 1e-5 * np.random.randn(Xtilde.shape[0], Y.shape[1])
    numEpochs = 3
    n = Y.shape[0]
    newIndices = np.random.permutation(n)
    for _ in range(numEpochs):
        for r in range(int(np.ceil(n / ntilde))):
            indices = newIndices[ntilde * r : ntilde * (r + 1)]
            Wtilde -= epsilon * gradfCE_reg(Xtilde[:, indices], Wtilde, Y[indices], 0)
            print(f"{ntilde * r}-{ntilde * (r + 1)}:\t{fCE_reg(Xtilde, Wtilde, Y, alpha)}")
    return Wtilde

def fPC(Xtilde, Wtilde, Y):
    Yhat = softmax(Xtilde, Wtilde)
    return np.count_nonzero(Y.argmax(axis=1) == Yhat.argmax(axis=1)) / Y.shape[0]

def fCE_reg(Xtilde, Wtilde, Y, alpha):
    Yhat = softmax(Xtilde, Wtilde)
    n = Y.shape[0]
    return (-1 * (Y * np.log(Yhat)).sum() / n) + ((alpha / (2 * n)) * (Wtilde[:-1] * Wtilde[:-1]).sum())

def gradfCE_reg(Xtilde, Wtilde, Y, alpha):
    Yhat = softmax(Xtilde, Wtilde)
    n = Y.shape[0]
    newWtilde = (Xtilde @ (Yhat - Y)) / n
    L2 = (alpha / n) * Wtilde[:-1]
    return np.vstack((newWtilde[:-1] + L2, newWtilde[-1]))

def softmax(Xtilde, Wtilde):
    Z = Xtilde.T @ Wtilde
    return np.exp(Z) / np.exp(Z).sum(axis=1)[:, np.newaxis]

if __name__ == "__main__":
    # Test case 1: Quadratic function
    x = np.linspace(-1, 1, 10)
    y = x**2 - 2*x + 1  # y = x^2 - 2x + 1
    w = trainPolynomialRegressor(x, y, 2)
    expected_w = np.array([1, -2, 1])
    assert np.allclose(w, expected_w, atol=1e-5), f"Test case 1 failed: {w} != {expected_w}"

    # Test case 2: Linear function
    x = np.linspace(-1, 1, 10)
    y = 3*x + 5  # y = 3x + 5
    w = trainPolynomialRegressor(x, y, 1)
    expected_w = np.array([5, 3])
    assert np.allclose(w, expected_w, atol=1e-5), f"Test case 2 failed: {w} != {expected_w}"
    
    # Test case 3: Cubic function
    x = np.linspace(-2, 2, 10)
    y = 2*x**3 - 4*x**2 + 3*x - 7  # y = 2x^3 - 4x^2 + 3x - 7
    w = trainPolynomialRegressor(x, y, 3)
    expected_w = np.array([-7, 3, -4, 2])
    assert np.allclose(w, expected_w, atol=1e-5), f"Test case 3 failed: {w} != {expected_w}"
    
    print("All test cases passed!")
    

    # Load data
    trainingImages = np.load("fashion_mnist_train_images.npy") / 255.0  # Normalizing by 255 helps accelerate training
    trainingLabels = np.load("fashion_mnist_train_labels.npy")
    testingImages = np.load("fashion_mnist_test_images.npy") / 255.0  # Normalizing by 255 helps accelerate training
    testingLabels = np.load("fashion_mnist_test_labels.npy")

    # Append a constant 1 term to each example to correspond to the bias terms
    trainingImages = np.vstack((trainingImages.T, np.ones((1, trainingImages.shape[0]))))
    testingImages = np.vstack((testingImages.T, np.ones((1, testingImages.shape[0]))))

    # Change from 0-9 labels to "one-hot" binary vector labels. For instance, 
    # if the label of some example is 3, then its y should be [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ]
    newTrainingLabels = np.zeros((trainingLabels.shape[0], 10))
    newTrainingLabels[np.arange(trainingLabels.shape[0]), trainingLabels] = 1
    newTestingLabels = np.zeros((testingLabels.shape[0], 10))
    newTestingLabels[np.arange(testingLabels.shape[0]), testingLabels] = 1

    # Train the model
    Wtilde = softmaxRegression(trainingImages, newTrainingLabels, epsilon=0.1, batchSize=100, alpha=0.1)
    print(f"fPC:\t{fPC(testingImages, Wtilde, newTestingLabels)}")
    
    # Visualize the vectors
    for i in range(Wtilde.shape[1]):
        plt.imshow(Wtilde[:-1, i].reshape(28, 28), cmap="gray")
        plt.show()