import warnings
import librosa
from .preprocessing import Samples
import numpy as np


def pitch(samples: Samples) -> float:
    freqs = librosa.yin(samples.data, 65, 2000, sr=samples.rate)
    return np.mean(freqs)


def spectral_rolloff(samples: Samples) -> float:
    roll_off = librosa.feature.spectral_rolloff(samples.data, samples.rate)
    return np.mean(roll_off)
