from typing import List
from dataclasses import dataclass
from .preprocessing import Samples
from . import analyze


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
    def empty():
        return Features(
            pitch=0,
            spectral_rolloff=0,
            mel_freq_c_coeff=0,
            tempo=0,
            rms_eng=0,
            spect_centr=0,
            beat_spec=0,
            zero_crossing=0,
            short_fft=0,
            kurtosis=0,
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
