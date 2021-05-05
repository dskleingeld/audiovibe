from audiovibe.extract.preprocessing import Samples
from audiovibe.extract.features import Features
import pytest
import numpy as np
from math import isclose
from audiovibe.extract.analyze import (
    pitch, spectral_rolloff, mel_freq_c_coeff, tempo,
    rms_eng, spect_centr, beat_spec, zero_crossing, short_fft, kurtosis
)


@pytest.fixture(scope="session")
def samples():
    PATH = "data/clips_45sec/1.mp3"
    return Samples.from_path(PATH)


def test_pitch(samples):
    f = pitch(samples)
    assert isinstance(f, float) or isinstance(f, np.float32)
    assert isclose(f, 142.85, abs_tol=0.01)


def test_spectral_rolloff(samples):
    f = spectral_rolloff(samples)
    assert isinstance(f, float) or isinstance(f, np.float32)
    assert isclose(f, 8086.47, abs_tol=0.01)


def test_mel_freq_c_coeff(samples):
    f = mel_freq_c_coeff(samples)
    assert isinstance(f, float) or isinstance(f, np.float32)
    assert isclose(f, 1.34, abs_tol=0.01)


def test_tempo(samples):
    f = tempo(samples)
    assert isinstance(f, float) or isinstance(f, np.float32)
    assert isclose(f, 114.84, abs_tol=0.01)


def test_rms_eng(samples):
    f = rms_eng(samples)
    assert isinstance(f, float) or isinstance(f, np.float32)
    assert isclose(f, 0.181, abs_tol=0.01)


def test_spect_centr(samples):
    f = spect_centr(samples)
    assert isinstance(f, float) or isinstance(f, np.float32)
    assert isclose(f, 3763.49, abs_tol=0.01)


# def test_beat_spec(samples):
#     f = beat_spec(samples)
    # assert isinstance(f, float)
#     assert isclose(f, 0, abs_tol=0.01)


def test_zero_crossing(samples):
    f = zero_crossing(samples)
    assert isinstance(f, float) or isinstance(f, np.float32)
    assert isclose(f, 0.091, abs_tol=0.01)


def test_short_fft(samples):
    f = short_fft(samples)
    assert isinstance(f, float) or isinstance(f, np.float32)
    assert isclose(f, 1.69, abs_tol=0.01)


def test_kurtosis(samples):
    f = kurtosis(samples)
    assert isinstance(f, float) or isinstance(f, np.float32)
    assert isclose(f, 0.654, abs_tol=0.01)

# if __name__ == "__main__":

#     s = samples()
#     test_tempo(s)
