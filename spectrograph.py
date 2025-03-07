import numpy as np
import librosa
import librosa.display
import pandas as pd
import os
import matplotlib.pyplot as plt

ENCODING = 128  #features in melspec
NUM_CLASSES = 3 
N_MELS = 128  #mel bands in spectro
HOP_LENGTH = 512  #samples between frames 

#makes .wav file into melspec
def extract_mel_spectrogram(wav_path, n_mels=N_MELS, hop_length=HOP_LENGTH, visualize=False):
    try:
        y, sr = librosa.load(wav_path, sr=None) 
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  #decibals 

        if visualize:
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Mel Spectrogram for {os.path.basename(wav_path)}')
            plt.show()

        return mel_spec_db.mean(axis=1)  
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None


def load_sound_data(testing=False, visualize_samples=3):
    if testing:
        trainingData = pd.read_csv("newTrain.csv").to_numpy()
        testingData = pd.read_csv("newTest.csv").to_numpy()

        Xtrain = trainingData[:, :-1]
        Ytrain = np.atleast_2d(trainingData[:, -1]).T
        Xtest = testingData[:, :-1]
        Ytest = np.atleast_2d(testingData[:, -1]).T

        onehot_train_labels = np.zeros((Ytrain.shape[0], NUM_CLASSES)) # 3 classes to predict
        onehot_train_labels[np.arange(Ytrain.shape[0]), Ytrain[:, 0].astype(int)] = 1 # performs one hot encoding
        onehot_test_labels = np.zeros((Ytest.shape[0], NUM_CLASSES)) # 3 classes to predict
        onehot_test_labels[np.arange(Ytest.shape[0]), Ytest[:, 0].astype(int)] = 1 # performs one hot encoding
    else:
        trainingData = pd.read_csv("train.csv").to_numpy()
        testingData = pd.read_csv("test.csv").to_numpy()
        Xtrain = trainingData[:, :-1]
        Ytrain = np.atleast_2d(trainingData[:, -1]).T
        Xtest = testingData

        onehot_train_labels = np.zeros((Ytrain.shape[0], NUM_CLASSES)) # 3 classes to predict
        onehot_train_labels[np.arange(Ytrain.shape[0]), Ytrain[:, 0].astype(int)] = 1 # performs one hot encoding

    XtestIDs = Xtest[:, 0]

    num_train = Xtrain.shape[0]
    num_test = Xtest.shape[0]

    candidateIds = Xtrain[:, 0]
    Xtrain = Xtrain[:, 1:].astype(float)
    Xtest = Xtest[:, 1:].astype(float)

    newXtrain = []
    new_onehot_train_labels = []
    sample_count = 0

    print("Processing training data...")
    #get all sounds for each training sample
    for i in range(num_train):
        candidate_id = candidateIds[i]
        wav_path = f"sounds/sounds/{candidate_id}/cough.wav"
        if os.path.exists(wav_path):
            mel_spec = extract_mel_spectrogram(wav_path, visualize=False)
            if mel_spec is not None:
                newXtrain.append(np.concatenate((mel_spec, Xtrain[i])))
                new_onehot_train_labels.append(onehot_train_labels[i])
                sample_count += 1 #using to make visualization code 

    newXtest = []
    print("Processing testing data...")
    sample_count = 0
    for i in range(num_test):
        candidate_id = XtestIDs[i]
        wav_path = f"sounds/sounds/{candidate_id}/cough.wav"
        if os.path.exists(wav_path):
            mel_spec = extract_mel_spectrogram(wav_path, visualize=(sample_count < visualize_samples))
            if mel_spec is not None:
                newXtest.append(np.concatenate((mel_spec, Xtest[i])))
                sample_count += 1

    #melspec data as array 
    newXtrain = np.array(newXtrain)
    new_onehot_train_labels = np.array(new_onehot_train_labels)
    newXtest = np.array(newXtest)

    print(f"Train Data Shape: {newXtrain.shape}")
    print(f"Train Labels Shape: {new_onehot_train_labels.shape}")
    print(f"Test Data Shape: {newXtest.shape}")
    print("Test", Xtest.shape)

    if testing:
        return newXtrain, new_onehot_train_labels, newXtest, Ytest
    else:
        return newXtrain, new_onehot_train_labels, newXtest, XtestIDs

if __name__ == "__main__":
    load_sound_data(testing=False)