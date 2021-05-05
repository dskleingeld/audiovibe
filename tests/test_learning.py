import pytest
import numpy as np

from audiovibe.think.learn import prepare_data
from audiovibe.index.database import Database, Row
from audiovibe.index.emotion import Emotion
from audiovibe.extract.features import Features


@pytest.fixture(scope="session")
def test_database(tmp_path_factory) -> Database:
    db = Database.from_path(str(tmp_path_factory))
    for i in range(1, 5):
        features = Features.empty(placeholder=i)
        emotion = Emotion.empty(placeholder=i/10)
        db.insert(str(i), Row(features, emotion))
    return db


def test_data_split(test_database):
    db = test_database
    data = db.get_with_emotion()
    train_in, train_out, val_in, val_out = prepare_data(data)

    assert isinstance(train_in, np.ndarray)
    assert train_in.shape[0] > 2
    assert train_in.shape[1] == len(Features.empty().as_list())

    assert isinstance(train_out, np.ndarray)
    assert train_out.shape[0] > 2
    assert train_out.shape[1] == len(Emotion.empty().as_list())
