import librosa
import numpy as np


def load(path: str) -> np.ndarray:
    (samples, _) = librosa.load(path, sr=441000, mono=True)
    return samples
