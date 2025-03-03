import numpy as np
import scipy
import librosa

'''PREPROCESSING STARTER CODE '''
def pad_audio(y, target_length):
    """Pads the audio with zeros to match the target length."""
    if len(y) < target_length:
        padding = target_length - len(y)
        y = np.pad(y, (0, padding), 'constant')
    return y

def truncate_audio(y, target_length):
    """Truncates the audio to match the target length."""
    if len(y) > target_length:
        y = y[:target_length]
    return y

def adjust_audio(y, target_length=32000):
    """Adjusts the audio to match the target length by padding or truncating."""
    if len(y) < target_length:
        return pad_audio(y, target_length)
    else:
        return truncate_audio(y, target_length)

def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) /
                                  original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform




''' Isolation of cough sounds '''
def segment_cough_sound(signal, sr, cough_threshold=0.05, min_cough_duration=0.1, padding=0.05):

    hop_length = int(min_cough_duration*sr)
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)

    energy = librosa.feature.rms(y=signal, hop_length=hop_length)[0]

    # Normalize the energy values
    energy_min, energy_max = np.min(energy), np.max(energy)
    if energy_max > energy_min:
        normalized_energy = (energy - energy_min) / (energy_max - energy_min)
    else:
        normalized_energy = np.zeros_like(energy)

    # normalized_energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))

    # Set the energy threshold for event detection
    cough_threshold = np.max(normalized_energy) * cough_threshold
    min_cough_samples = round(sr * min_cough_duration)

    # Find the cough segments
    cough_segments = []
    cough_times = []
    event_start = None

    for i, value in enumerate(normalized_energy):
        if value >= cough_threshold:
            if event_start is None:
                event_start = i*hop_length
        else:
            if event_start is not None:
                cough_duration = i*hop_length - event_start
                if cough_duration >= min_cough_samples:
                    event_end = i*hop_length + int(padding * sr)
                    event_start -= int(padding * sr)
                    event_start = max(event_start, 0)
                    cough_segments.append(signal[event_start: event_end+1])
                    cough_times.append((event_start / sr, event_end / sr))
                event_start = None

    return cough_segments, cough_times
