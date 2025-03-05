import numpy as np
import librosa
import librosa.display
import pandas as pd
import os
import matplotlib.pyplot as plt

ENCODING = 128  
NUM_CLASSES = 3
N_MELS = 128  
HOP_LENGTH = 512  

def extract_mel_spectrogram(wav_path, n_mels=N_MELS, hop_length=HOP_LENGTH, visualize=False):
    try:
        y, sr = librosa.load(wav_path, sr=None) 
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  

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


def load_sound_data(visualize_samples=3):
    trainingData = pd.read_csv("train.csv")
    testingData = pd.read_csv("test.csv")

    Xtrain = trainingData.to_numpy()[:, :-1]  # Ignores labels in last column
    Xtest = testingData.to_numpy()
    XtestIDs = Xtest[:, 0]
    Ytrain = np.atleast_2d(trainingData.to_numpy()[:, -1]).T  # Labels
    Ytrain = Ytrain.reshape(Ytrain.shape[0], 1)

    num_train = Xtrain.shape[0]
    num_test = Xtest.shape[0]
    onehot_train_labels = np.zeros((num_train, NUM_CLASSES))
    onehot_train_labels[np.arange(num_train), Ytrain[:, 0].astype(int)] = 1  # One-hot encoding

    candidateIds = Xtrain[:, 0]
    Xtrain = Xtrain[:, 1:].astype(float)
    Xtest = Xtest[:, 1:].astype(float)

    newXtrain = []
    new_onehot_train_labels = []
    sample_count = 0  

    print("Processing training data...")
    for i in range(num_train):
        candidate_id = candidateIds[i]
        wav_path = f"sounds/sounds/{candidate_id}/cough.wav"
        if os.path.exists(wav_path):
            mel_spec = extract_mel_spectrogram(wav_path, visualize=False)
            if mel_spec is not None:
                newXtrain.append(np.concatenate((mel_spec, Xtrain[i])))
                new_onehot_train_labels.append(onehot_train_labels[i])
                sample_count += 1

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

    newXtrain = np.array(newXtrain)
    new_onehot_train_labels = np.array(new_onehot_train_labels)
    newXtest = np.array(newXtest)

    print(f"Train Data Shape: {newXtrain.shape}")
    print(f"Train Labels Shape: {new_onehot_train_labels.shape}")
    print(f"Test Data Shape: {newXtest.shape}")

    return newXtrain, new_onehot_train_labels, newXtest, XtestIDs

load_sound_data()
