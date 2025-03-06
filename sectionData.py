import numpy as np
import pandas

trainingData = pandas.read_csv("train.csv").to_numpy()
randTrainingData = np.random.default_rng().permutation(trainingData)
cutoff = int(trainingData.shape[0] * 0.8)
newTrainingData = randTrainingData[:cutoff]
newTestingData = randTrainingData[cutoff:]
header = "candidateID,age,gender,tbContactHistory,wheezingHistory,phlegmCough,familyAsthmaHistory,feverHistory,coldPresent,packYears,disease"
np.savetxt("newTrain.csv", newTrainingData, delimiter=",", fmt="%s", header=header, comments="")
np.savetxt("newTest.csv", newTestingData, delimiter=",", fmt="%s", header=header, comments="")