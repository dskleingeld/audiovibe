import warnings
import librosa
from .preprocessing import Samples
import numpy as np
import scipy


def pitch(samples: Samples) -> float:
    freqs = librosa.yin(samples.data, 65, 2000, sr=samples.rate)
    return np.mean(freqs)


def spectral_rolloff(samples: Samples) -> float:
    roll_off = librosa.feature.spectral_rolloff(samples.data, samples.rate)
    return np.mean(roll_off)


def mel_freq_c_coeff(samples: Samples) -> float:
    sequence = librosa.feature.mfcc(samples.data, samples.rate)
    return np.mean(sequence)


def tempo(samples: Samples) -> float:
    tempo = librosa.beat.tempo(samples.data, samples.rate)
    return tempo


def rms_eng(samples: Samples) -> float:
    energies = librosa.feature.rms(samples.data, samples.rate)
    return np.mean(energies)


def spect_centr(samples: Samples) -> float:
    freqs = librosa.feature.spectral_centroid(samples.data, samples.rate)
    return np.mean(freqs)


def beat_spec(samples: Samples) -> float:
    # TODO https://www.fxpal.com/publications/the-beat-spectrum-a-new-approach-to-rhythm-analysis.pdf
    raise NotImplementedError
    return 0


def zero_crossing(samples: Samples) -> float:
    rates = librosa.feature.zero_crossing_rate(samples.data)
    return np.mean(rates)


def short_fft(samples: Samples) -> float:
    spectrum = librosa.stft(samples.data, center=False)
    return np.mean(np.abs(spectrum))


def kurtosis(samples: Samples) -> float:
    k = scipy.stats.kurtosis(samples.data)
    return np.mean(k)
