from preprocessing import *
import numpy as np
import scipy
import librosa
from scipy.io.wavfile import write
import json
import pandas

ENCODING = 512

# print("Loading Audio...")
# original_audio, original_sample_rate = librosa.load(f"sounds/sounds/0a6908bb417fa/cough.wav", sr=None, mono=False)
# print("Loaded audio:", original_audio.shape, "sample rate:", original_sample_rate)
# sample_rate, audio = ensure_sample_rate(original_sample_rate, original_audio)
# print("Ensured sample rate:", audio.shape, "sample rate:", sample_rate)
# cough_segments, cough_times = segment_cough_sound(audio, sample_rate)
# print("Cough Segments:", len(cough_segments))
# print("cough times:", cough_times)

# coughs_joined = np.array([])
# for cough in cough_segments[0:4]:
#     coughs_joined = np.concatenate((coughs_joined, cough))

# print("truncated cough shape:", coughs_joined.shape)

# audio = adjust_audio(audio)
# print("Adjusted audio:", audio.shape, "sample rate:", sample_rate)

# cough1_start = int(cough_times[0][0]*sample_rate)
# cough4_end = int(cough_times[3][1]*sample_rate)

# data = np.random.uniform(-1, 1, sample_rate) # 1 second worth of random samples between -1 and 1
# data = audio[cough1_start:cough4_end]
# print(data.shape)
# scaled = np.int16(data / np.max(np.abs(data)) * 32767)
# write('test.wav', sample_rate, scaled)
# print("Wrote Test Wav file")

# print("\nNUMPY FILE\n")

np_audio = np.load(f"sounds/sounds/0a6908bb417fa/vowel-opera.npy").T
print("NUMPY", np_audio)

# cough_segments, cough_times = segment_cough_sound(np_audio, sample_rate)
# print("Cough Segments:", len(cough_segments))#, cough_segments)
# print("cough times:", cough_times)

with open(f"sounds/sounds/0a6908bb417fa/emb_vowel.json") as f:
    json_cough = json.load(f)

json_cough = np.array(json_cough)

print("vowel:", json_cough.shape, json_cough[0][0])

# print("shape:", (json_cough[0][0] + json_cough[1][0] + json_cough[2][0] + json_cough[3][0]) /4)

# data = json_cough.reshape(json_cough.shape[1]*4,)
# scaled = np.int16(data / np.max(np.abs(data)) * 32767)
# write('test.wav', int(sample_rate/4), scaled)
# print("Wrote Test Wav file")

# trainingData = pandas.read_csv("train.csv")

# # grab the training data from CSV
# Xtrain = trainingData.to_numpy()[:, :-1] # ignores labels in last column
# total_count = Xtrain.shape[0]

# coughs = np.zeros((total_count, ENCODING))
# vowels = np.zeros((total_count, ENCODING))
# candidateIds = Xtrain[:, 0]
# ones = 0
# twos = 0
# threes = 0
# fours = 0
# not_found = 0
# for i in range(total_count):
#     try:
#         with open(f"sounds/sounds/{candidateIds[i]}/emb_vowel.json") as f:
#             coughs = json.load(f)
#         coughs = np.array(coughs)
#         if coughs.shape[0] == 1:
#             ones += 1
#         elif coughs.shape[0] == 2:
#             twos +=1
#         elif coughs.shape[0] == 3:
#             threes += 1
#         elif coughs.shape[0] == 4:
#             fours += 1
#         else:
#             print(coughs.shape)
#     except FileNotFoundError:
#         not_found += 1
#         print(candidateIds[i])
# print("cough count", not_found, ones, twos, threes, fours)
