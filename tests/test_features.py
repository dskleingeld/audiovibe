import pytest
from math import isclose
from audiovibe.extract.analyze import pitch, spectral_rolloff
from audiovibe.extract.features import Features
from audiovibe.extract.preprocessing import Samples


@pytest.fixture(scope="session")
def samples():
    PATH = "data/clips_45sec/1.mp3"
    return Samples.from_path(PATH)


def test_pitch(samples):
    f = pitch(samples)
    assert isclose(f, 819.47, abs_tol=0.01)


def test_spectral_rolloff(samples):
    f = spectral_rolloff(samples)
    assert isclose(f, 7582.28, abs_tol=0.01)


def test_mel_freq(samples):

# if __name__ == "__main__":
#     test_pitch(samples)
