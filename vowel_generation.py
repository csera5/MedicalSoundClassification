import numpy as np
import pandas
import json
import os

ENCODING = 512
NUM_CLASSES = 3

def load_vowel_data(trainingData, trainIds, testingData, testIds, type="npy"):
    # grab the training data from CSV
    allX = np.append(trainingData, testingData, axis=0) # ignores labels in last column
    total_count = allX.shape[0]

    # grab the XtrainOutput from the CSV and convert to onehot
    # XtrainOutput = np.atleast_2d(trainingData.to_numpy()[:, -1]).T # grabs labels from last column
    # XtrainOutput = XtrainOutput.reshape(XtrainOutput.shape[0] , 1) # makes shape a 2d array, easier for later
    # onehot_train_labels = np.zeros((total_count, NUM_CLASSES)) # 3 classes to predict
    # onehot_train_labels[np.arange(total_count), XtrainOutput[:, 0].astype(int)] = 1 # performs one hot encoding

    # append allX and XtrainOutput together
    # allX = np.hstack((allX, onehot_train_labels))

    coughs = np.zeros((total_count, ENCODING))
    candidateIds = np.append(trainIds, testIds, axis=0)
    for i in range(total_count):
        coughs[i] = np.load(f"sounds/sounds/{candidateIds[i]}/cough-opera.npy") # loads cough data for each participant

    # print(allX.shape)

    # break out Ytrain and Xtest
    vowels = np.zeros((total_count, ENCODING))
    trainIndices = np.zeros(total_count)
    testIndices = np.zeros(total_count)
    for i in range(total_count):
        try:
            if type == "npy":
                vowels[i] = np.load(f"sounds/sounds/{candidateIds[i]}/vowel-opera.npy") # loads vowel data for each participant
            else:
                with open(f"sounds/sounds/{candidateIds[i]}/emb_vowel.json") as f:
                    temp = json.load(f)
                vowels[i] = np.array(temp)
            trainIndices[i] = 1
        except FileNotFoundError:
            testIndices[i] = 1

    testIndices = np.where(testIndices == 1)
    trainIndices = np.where(trainIndices == 1)

    train_coughs = coughs[trainIndices]
    test_coughs = coughs[testIndices]

    Xtest = allX[testIndices]
    Xtrain = allX[trainIndices]
    Ytrain = vowels[trainIndices]

    newXtrain = np.zeros((0, ENCODING + 11))
    newYtrain = np.zeros((0, ENCODING))
    for i in range(Xtrain.shape[0]):
        coughs = np.zeros((0, ENCODING))
        cough = np.zeros((0, ENCODING))
        try:
            with open(f"sounds/sounds/{candidateIds[i]}/emb_cough.json") as f:
                coughs = np.array(json.load(f))
        except:
            pass
        cough = np.load(f"sounds/sounds/{candidateIds[i]}/cough-opera.npy")
        coughs = np.append(coughs, cough, axis=0)

        newXtrain = np.concatenate((newXtrain, np.concatenate((coughs, np.tile(Xtrain[i], (coughs.shape[0], 1))), axis=1)), axis=0)
        newYtrain = np.append(newYtrain, np.tile(Ytrain[i], (coughs.shape[0], 1)), axis=0)

    Xtest = np.concatenate((test_coughs, Xtest), axis=1) # adds coughs to Xtest array, remove candidateIDs

    cough_noise = np.random.default_rng().normal(0, 1e-1, (newXtrain.shape[0], ENCODING))
    age_noise = np.random.default_rng().normal(0, 1, (newXtrain.shape[0], 1))
    packYears_noise = np.random.default_rng().normal(0, 5, (newXtrain.shape[0], 1))
    newXtrain = np.vstack((newXtrain, np.concatenate((newXtrain[:, :ENCODING] + cough_noise, newXtrain[:, ENCODING].reshape((newXtrain.shape[0], 1)) + age_noise, newXtrain[:, ENCODING + 1:-1], np.atleast_2d(newXtrain[:, -1]).T + packYears_noise), axis=1)))
    newYtrain = np.tile(newYtrain, (2, 1))

    return newXtrain, newYtrain, Xtest, candidateIds[testIndices]

def relu(z):
    return np.maximum(0, z)

def fMSE(yhat, y):
    n = yhat.shape[1]
    return np.sum(np.sum((y - yhat) ** 2, axis=0) / (2 * n)) / n

def forward_prop (x, y, W1, b1, W2, b2):
    z = np.dot(W1, x) + b1.reshape(-1, 1)  # 20xbatchSize
    h = relu(z) # 20xbatchSize
    yhat = np.dot(W2, h) + b2.reshape(-1, 1)  # 1xbatchSize
    loss = fMSE(yhat, y)
    return loss, x, z, h, yhat

def relu_prime(z):
    return np.where(z > 0, 1, 0)

def back_prop (X, y, W1, b1, W2, b2, alpha=0):
    n = X.shape[1]
    _, _, z, h, yhat = forward_prop(X, y, W1, b1, W2, b2)
    g = (np.dot((yhat-y).T, W2) * relu_prime(z.T)).T # 20xbatchSize
    gradW1 = np.dot(g, X.T) / n + alpha / n * W1 # 20x2304
    gradW2 = np.dot((yhat-y), h.T) / n + alpha / n * W2 # 1x20
    gradb1 = np.sum(g, axis=1) / n # 20 x batchSize
    gradb2 = (np.sum(yhat-y, axis=1)) / n # 1xbatchSize
    return gradW1, gradb1, gradW2, gradb2

def train (trainX, trainY, W1, b1, W2, b2, epsilon = 1e-2, batchSize = 64, numEpochs = 1000, alpha=0):
    M = trainX.shape[1] # number of examples
    indexes = np.random.permutation(M)
    for e in range(numEpochs):
        if e % 20 == 0: print("Progress:", str(round(float(e)/numEpochs*100, 2)) + "%")
        for i in range(0, M, batchSize):
            end = i + batchSize if ((i+batchSize) < M) else M-1
            batchX = trainX[:, indexes[i:end]] # 2304 x batchSize
            batchY = trainY[:, indexes[i:end]] # 1 x batchSize
            gradW1, gradb1, gradW2, gradb2 = back_prop(batchX, batchY, W1, b1, W2, b2, alpha)
            curr_epsilon = epsilon if epsilon > 0 else (1e-4 if e < 20 else (1e-5 if e < 100 else 1e-6))
            curr_epsilon = epsilon
            W1 -= curr_epsilon * gradW1 
            b1 -= curr_epsilon * gradb1
            W2 -= curr_epsilon * gradW2
            b2 -= curr_epsilon * gradb2

    return W1, b1, W2, b2

def vowelNN(trainX, trainY, epsilon=1e-4, batchSize=32, numEpochs=100, numHidden=20, alpha=0):
    num_input = trainX.shape[0]
    num_output = trainY.shape[0]
    # Initialize weights to reasonable random values
    W1 = 2*(np.random.random(size=(numHidden, num_input))/num_input**0.5) - 1./num_input**0.5
    b1 = 0.01 * np.ones(numHidden)
    W2 = 2*(np.random.random(size=(num_output, numHidden))/numHidden**0.5) - 1./numHidden**0.5
    b2 = np.mean(trainY)

    W1, b1, W2, b2 = train(trainX, trainY, W1, b1, W2, b2, epsilon=epsilon, batchSize=batchSize, numEpochs=numEpochs, alpha=alpha)
    loss, _, _, _, _ = forward_prop(trainX, trainY, W1, b1, W2, b2)
    print("Final Training Loss:", loss)

    return W1, b1, W2, b2

def format_vowel_data(train_test_split=0.75, type="npy"):

    allX, allY, actualTest = load_vowel_data(type)
    print(allX.shape, allY.shape, actualTest.shape)

    cutoff = int(allX.shape[0] * train_test_split)

    Xtrain = allX[0:cutoff, :]
    Ytrain = allY[0:cutoff, :]
    Xtest = allX[cutoff+1:, :]
    Ytest = allY[cutoff+1:, :]

    print(f"Training data shape: X={Xtrain.shape}, Y={Ytrain.shape}")
    print(f"Testing data shape: X={Xtest.shape}, Y={Ytest.shape}")
    print(f"Actual Testing data shape:{actualTest.shape}")

    return Xtrain, Ytrain, Xtest, Ytest, actualTest

def generate_vowels(X, W1, b1, W2, b2):
    z = np.dot(W1, X.T) + b1.reshape(-1, 1)  # 20xbatchSize
    h = relu(z) # 20xbatchSize
    yhat = np.dot(W2, h) + b2.reshape(-1, 1)  # 1xbatchSize
    return yhat.T

def start(trainingData, trainIds, testingData, testIds):
    file_type = "json"

    Xtrain, Ytrain, actualTest, newIds = load_vowel_data(trainingData, trainIds, testingData, testIds, type=file_type)

    print(Xtrain.shape)
    print(Ytrain.shape)
    print(actualTest.shape)

    if file_type == "npy":
        W1, b1, W2, b2 = vowelNN(Xtrain.T, Ytrain.T, epsilon=0.00001, batchSize=16, numEpochs=200, numHidden=20)
    else:
        W1, b1, W2, b2 = vowelNN(Xtrain.T, Ytrain.T, epsilon=0.00001, batchSize=32, numEpochs=100, numHidden=30, alpha=0.0001)

    vowels = generate_vowels(actualTest, W1, b1, W2, b2)
    print(vowels.shape)

    try:
        os.mkdir("./newSounds")
        for i in range(len(newIds)):
            os.mkdir(f"./newSounds/{newIds[i]}")
    except:
        pass
    for i in range(len(newIds)):
        np.save(f"./newSounds/{newIds[i]}/vowel-opera.npy", vowels[i])