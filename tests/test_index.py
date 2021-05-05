from audiovibe.extract.features import Features
from audiovibe.index.database import Database, Row


def test_add(tmp_path):
    f_in = Features.empty()
    db = Database.from_path(tmp_path/"db")
    row = Row.without_emotion(f_in)
    db.insert("test_row", row)
    del db

    db = Database.from_path(tmp_path/"db")
    row = db.get("test_row")
    assert f_in == row.features


def test_exists(tmp_path):
    f_in = Features.empty()
    db = Database.from_path(tmp_path/"db")
    assert not db.contains("test_row")

    row = Row.without_emotion(f_in)
    db.insert("test_row", row)
    assert db.contains("test_row")
