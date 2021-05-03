# import audiovibe.extract.preprocessing
import audiovibe


PATH = "data/clips_45sec/1.mp3"


def test_pitch():
    samples = audiovibe.extract.preprocessing.load(PATH)
    audiovibe.extract.features.Features.from_samples(samples)


if __name__ == "__main__":
    test_pitch()
