8abb94e5 (taccat3           2025-03-05 22:16:30 -0500   1) import numpy as np
8abb94e5 (taccat3           2025-03-05 22:16:30 -0500   2) import pandas
8abb94e5 (taccat3           2025-03-05 22:16:30 -0500   3) import json
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500   4) from vowel_generation import start
8abb94e5 (taccat3           2025-03-05 22:16:30 -0500   5) 
8abb94e5 (taccat3           2025-03-05 22:16:30 -0500   6) ENCODING = 512
8abb94e5 (taccat3           2025-03-05 22:16:30 -0500   7) NUM_CLASSES = 3
8abb94e5 (taccat3           2025-03-05 22:16:30 -0500   8) 
00000000 (Not Committed Yet 2025-03-07 12:59:17 -0500   9) def load_continuous_data(testing=False):
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  10)     if testing:
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  11)         trainingData = pandas.read_csv("newTrain.csv").to_numpy()
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  12)         testingData = pandas.read_csv("newTest.csv").to_numpy()
8abb94e5 (taccat3           2025-03-05 22:16:30 -0500  13) 
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  14)         Xtrain = trainingData[:, :-1]
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  15)         Ytrain = np.atleast_2d(trainingData[:, -1]).T
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  16)         Xtest = testingData[:, :-1]
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  17)         Ytest = np.atleast_2d(testingData[:, -1]).T
8abb94e5 (taccat3           2025-03-05 22:16:30 -0500  18) 
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  19)         onehot_train_labels = np.zeros((Ytrain.shape[0], NUM_CLASSES)) # 3 classes to predict
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  20)         onehot_train_labels[np.arange(Ytrain.shape[0]), Ytrain[:, 0].astype(int)] = 1 # performs one hot encoding
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  21)         onehot_test_labels = np.zeros((Ytest.shape[0], NUM_CLASSES)) # 3 classes to predict
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  22)         onehot_test_labels[np.arange(Ytest.shape[0]), Ytest[:, 0].astype(int)] = 1 # performs one hot encoding
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  23)     else:
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  24)         trainingData = pandas.read_csv("train.csv").to_numpy()
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  25)         testingData = pandas.read_csv("test.csv").to_numpy()
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  26)         Xtrain = trainingData[:, :-1]
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  27)         Ytrain = np.atleast_2d(trainingData[:, -1]).T
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  28)         Xtest = testingData
8abb94e5 (taccat3           2025-03-05 22:16:30 -0500  29) 
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  30)         onehot_train_labels = np.zeros((Ytrain.shape[0], NUM_CLASSES)) # 3 classes to predict
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  31)         onehot_train_labels[np.arange(Ytrain.shape[0]), Ytrain[:, 0].astype(int)] = 1 # performs one hot encoding
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  32) 
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  33)     onehot_train_coldpresent = np.zeros((Xtrain.shape[0], 3))
8abb94e5 (taccat3           2025-03-05 22:16:30 -0500  34)     onehot_train_coldpresent[Xtrain[:, 8] == 0, 0] = 1
8abb94e5 (taccat3           2025-03-05 22:16:30 -0500  35)     onehot_train_coldpresent[Xtrain[:, 8] == 1, 1] = 1
8abb94e5 (taccat3           2025-03-05 22:16:30 -0500  36)     onehot_train_coldpresent[np.isnan(Xtrain[:, 8].astype(float)), 2] = 1
8abb94e5 (taccat3           2025-03-05 22:16:30 -0500  37) 
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  38)     onehot_test_coldpresent = np.zeros((Xtest.shape[0], 3))
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  39)     onehot_test_coldpresent[Xtest[:, 8] == 0, 0] = 1
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  40)     onehot_test_coldpresent[Xtest[:, 8] == 1, 1] = 1
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  41)     onehot_test_coldpresent[np.isnan(Xtest[:, 8].astype(float)), 2] = 1
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  42) 
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  43)     trainIds = Xtrain[:, 0]
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  44)     testIds = Xtest[:, 0]
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  45)     
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  46)     # Remove columns 1-8 (ordinal data), keeping other columns
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  47)     Xtrain = np.concatenate((Xtrain[:, 9:], onehot_train_coldpresent, np.atleast_2d(Xtrain[:, 9]).T.astype(float)), axis=1)
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  48)     Xtest = np.concatenate((Xtest[:, 9:], onehot_test_coldpresent, np.atleast_2d(Xtest[:, 9]).T.astype(float)), axis=1)
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  49) 
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  50)     start(Xtrain, trainIds, Xtest, testIds)
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  51)     print()
8abb94e5 (taccat3           2025-03-05 22:16:30 -0500  52) 
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  53)     newXtrain = np.zeros((0, ENCODING * 2 + 11))
8abb94e5 (taccat3           2025-03-05 22:16:30 -0500  54)     new_onehot_train_labels = np.zeros((0, 3))
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  55)     for i in range(Xtrain.shape[0]):
8abb94e5 (taccat3           2025-03-05 22:16:30 -0500  56)         try:
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  57)             vowel = np.load(f"sounds/sounds/{trainIds[i]}/vowel-opera.npy")
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  58)         except FileNotFoundError:
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  59)             vowel = np.load(f"newSounds/{trainIds[i]}/vowel-opera.npy")
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  60) 
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  61)         coughs = np.zeros((0, ENCODING))
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  62)         cough = np.zeros((0, ENCODING))
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  63)         try:
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  64)             with open(f"sounds/sounds/{trainIds[i]}/emb_cough.json") as f:
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  65)                 coughs = np.array(json.load(f))
8abb94e5 (taccat3           2025-03-05 22:16:30 -0500  66)         except:
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  67)             pass
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  68)         cough = np.load(f"sounds/sounds/{trainIds[i]}/cough-opera.npy")
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  69)         coughs = np.append(coughs, cough, axis=0)
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  70)         Xtrain_i = np.tile(Xtrain[i], (coughs.shape[0], 1))
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  71)         Xtrain_i = Xtrain_i[:, :coughs.shape[1]]
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  72)                             
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  73)         newXtrain = np.concatenate((newXtrain, np.concatenate((coughs, Xtrain_i), axis=1)), axis=0)
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  74)         new_onehot_train_labels = np.append(new_onehot_train_labels, np.tile(onehot_train_labels[i], (coughs.shape[0], 1)), axis=0)
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  75) 
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  76)     newXtest = np.zeros((0, ENCODING * 2 + 11))
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  77)     for i in range(Xtest.shape[0]):
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  78)         try:
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  79)             vowel = np.load(f"sounds/sounds/{testIds[i]}/vowel-opera.npy")
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  80)         except FileNotFoundError:
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  81)             vowel = np.load(f"newSounds/{testIds[i]}/vowel-opera.npy")
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  82)         try:
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  83)             with open(f"sounds/sounds/{testIds[i]}/emb_cough.json") as f:
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  84)                 coughs = np.array(json.load(f))
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  85)                 newXtest = np.append(newXtest, np.concatenate((coughs.mean(axis=1), np.atleast_2d(vowel), np.atleast_2d(Xtest[i])), axis=1), axis=0)
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  86)         except:
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  87)             cough = np.load(f"sounds/sounds/{testIds[i]}/cough-opera.npy")
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  88)             newXtest = np.append(newXtest, np.concatenate((cough, np.atleast_2d(vowel), np.atleast_2d(Xtest[i])), axis=1), axis=0)
8abb94e5 (taccat3           2025-03-05 22:16:30 -0500  89) 
8abb94e5 (taccat3           2025-03-05 22:16:30 -0500  90)     cough_noise = np.random.default_rng().normal(0, 1e-1, (newXtrain.shape[0], ENCODING))
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  91)     vowel_noise = np.random.default_rng().normal(0, 1e-2, (newXtrain.shape[0], ENCODING))
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  92)     age_noise = np.random.default_rng().normal(0, 1, (newXtrain.shape[0], 1))
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  93)     packYears_noise = np.random.default_rng().normal(0, 5, (newXtrain.shape[0], 1))
8abb94e5 (taccat3           2025-03-05 22:16:30 -0500  94) 
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  95)     newXtrain = np.concatenate((newXtrain, np.concatenate((newXtrain[:, :ENCODING] + cough_noise, newXtrain[:, ENCODING:ENCODING * 2] + vowel_noise, np.atleast_2d(newXtrain[:, ENCODING * 2]).T + age_noise, newXtrain[:, ENCODING * 2 + 1:-1], np.atleast_2d(newXtrain[:, -1]).T + packYears_noise), axis=1)), axis=0)
8abb94e5 (taccat3           2025-03-05 22:16:30 -0500  96)     new_onehot_train_labels = np.tile(new_onehot_train_labels, (2, 1))
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500  97) 
8abb94e5 (taccat3           2025-03-05 22:16:30 -0500  98)     print(newXtrain.shape)
8abb94e5 (taccat3           2025-03-05 22:16:30 -0500  99)     print(new_onehot_train_labels.shape)
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500 100)     print(newXtest.shape)
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500 101)     if testing:
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500 102)         print(onehot_test_labels.shape)
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500 103)     else:
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500 104)         print(testIds.shape)
8abb94e5 (taccat3           2025-03-05 22:16:30 -0500 105) 
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500 106)     if testing:
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500 107)         return newXtrain, new_onehot_train_labels, newXtest, onehot_test_labels
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500 108)     else:
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500 109)         return newXtrain, new_onehot_train_labels, newXtest, testIds
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500 110)     
467ab31e (Sukriti Kushwaha  2025-03-07 17:47:27 -0500 111) if __name__ == "__main__":
00000000 (Not Committed Yet 2025-03-07 12:59:17 -0500 112)     load_continuous_data(testing=False)
