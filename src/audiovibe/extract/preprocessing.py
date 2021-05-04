from dataclasses import dataclass
import warnings
import librosa
import numpy as np


@dataclass
class Samples():
    data: np.ndarray
    rate: int

    @staticmethod
    def from_path(path: str) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PySoundFile")
            (data, rate) = librosa.load(path, sr=44100, mono=True)
        return Samples(data, rate)
