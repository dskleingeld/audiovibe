import os
from dataclasses import dataclass
from typing import Optional
from ..extract.features import Features
from .emotion import Emotion

from loguru import logger
import pickledb


@dataclass
class Row:
    features: Features
    emotion: Optional[Emotion]

    @staticmethod
    def without_emotion(features: Features):
        return Row(features, None)

    @staticmethod
    def deserializable(data: str):
        fields = data.split(",")
        floats = [float(e) for e in fields]
        features = Features(*floats[:10])

        emotion = None
        if len(floats) > 10:
            emotion = Emotion(floats[10])
        return Row(features, emotion)

    def serializable(self) -> str:

        data = [self.features.pitch,
                self.features.spectral_rolloff,
                self.features.mel_freq_c_coeff,
                self.features.tempo,
                self.features.rms_eng,
                self.features.spect_centr,
                self.features.beat_spec,
                self.features.zero_crossing,
                self.features.short_fft,
                self.features.kurtosis]
        if self.emotion is not None:
            data.append(self.emotion.test)

        data = [str(e) for e in data]
        return ",".join(data)


@dataclass
class Database:
    db: pickledb.PickleDB

    @staticmethod
    def from_path(path: str):
        if not os.path.isfile(path):
            logger.info(f"creating new database in: {path}")

        db = pickledb.load(path, True)
        print(type(db))
        return Database(db)

    def add(self, key: str, value: Row):
        assert isinstance(value, Row)
        data = value.serializable()
        self.db.set(key, data)

    def get(self, key: str) -> Row:
        data = self.db.get(key)
        return Row.deserializable(data)
