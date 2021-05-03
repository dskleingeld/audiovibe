import aubio
import librosa
import numpy


def pitch(data: np.ndarray) -> float:
    return aubio.pitch()
