from dataclasses import dataclass
import numpy as np
import extract


@dataclass
class Features:
    pitch: float
    spectral_rolloff: float
    mel_freq: float
    cepstral_coeff: float
    tempo: float
    rms_eng: float
    spect_centr: float
    beat_spec: float
    zero_crossing: float
    short_fft: float
    kurtosis: float

    @staticmethod
    def from_samples(samples: np.ndarray):
        return Features(
            pitch=extract.pitch(samples),
            spectral_rolloff=0,
            mel_freq=0,
            cepstral_coeff=0,
            tempo=0,
            rms_eng=0,
            spect_centr=0,
            beat_spec=0,
            zero_crossing=0,
            short_fft=0,
            kurtosis=0,
        )
