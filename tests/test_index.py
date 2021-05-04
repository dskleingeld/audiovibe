from audiovibe.extract.features import Features
from audiovibe.index.database import Database, Row


def test_add():
    f_in = Features.for_testing()
    db = Database.from_path("test.db")
    row = Row.without_emotion(f_in)
    db.add("test_row", row)
    del db

    db = Database.from_path("test.db")
    row = db.get("test_row")
    assert f_in == row.features
