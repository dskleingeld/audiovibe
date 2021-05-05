from typing import List
from dataclasses import dataclass
from .preprocessing import Samples
from . import analyze
from loguru import logger


@dataclass
class Features:
    pitch: float  # fundamental fequency over entire sample song
    spectral_rolloff: float
    mel_freq_c_coeff: float
    tempo: float
    rms_eng: float
    spect_centr: float
    beat_spec: float
    zero_crossing: float
    short_fft: float
    kurtosis: float

    @staticmethod
    def from_samples(samples: Samples):
        return Features(
            pitch=analyze.pitch(samples),
            spectral_rolloff=analyze.spectral_rolloff(samples),
            mel_freq_c_coeff=analyze.mel_freq_c_coeff(samples),
            tempo=analyze.tempo(samples),
            rms_eng=analyze.rms_eng(samples),
            spect_centr=analyze.spect_centr(samples),
            # beat_spec=analyze.beat_spec(samples),
            beat_spec=0,
            zero_crossing=analyze.zero_crossing(samples),
            short_fft=analyze.short_fft(samples),
            kurtosis=analyze.kurtosis(samples),
        )

    @staticmethod
    def empty(placeholder=0):
        return Features(
            pitch=placeholder,
            spectral_rolloff=placeholder,
            mel_freq_c_coeff=placeholder,
            tempo=placeholder,
            rms_eng=placeholder,
            spect_centr=placeholder,
            beat_spec=placeholder,
            zero_crossing=placeholder,
            short_fft=placeholder,
            kurtosis=placeholder,
        )

    def as_list(self) -> List[float]:
        return [self.pitch,
                self.spectral_rolloff,
                self.mel_freq_c_coeff,
                self.tempo,
                self.rms_eng,
                self.spect_centr,
                self.beat_spec,
                self.zero_crossing,
                self.short_fft,
                self.kurtosis]
