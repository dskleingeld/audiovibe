import os
from dataclasses import dataclass
from typing import Optional, List
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
        print(data)
        fields = data.split(",")
        floats = [float(e) for e in fields]
        features = Features(*floats[:10])

        emotion = None
        if len(floats) > 10:
            emotion = Emotion(floats[10])
        return Row(features, emotion)

    def serializable(self) -> str:
        data = self.features.as_list()
        if self.emotion is not None:
            data.extend(self.emotion.as_list())

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

    def insert(self, key: str, value: Row):
        assert isinstance(value, Row)
        data = value.serializable()
        self.db.set(key, data)

    def get(self, key: str) -> Row:
        data = self.db.get(key)
        assert data is not False, f"key: \'{key}\' not in database"
        return Row.deserializable(data)

    def contains(self, key: str) -> bool:
        value = self.db.get(key)
        return value is not False

    def set_emotion(self, key: str, emotion: Emotion):
        assert isinstance(emotion, Emotion)
        row = self.get(key)
        row.emotion = emotion
        self.insert(key, row)

    def get_with_emotion(self) -> List[Row]:
        list = []
        for key in self.db.getall():
            row = self.db.get(key)
            if row.emotion is None:
                continue
            list.append(row)
        return list
