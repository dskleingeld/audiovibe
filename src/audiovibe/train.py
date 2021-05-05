from audiovibe.index.database import Database
from audiovibe.index.dir import index, add_emotion_from_csv
from audiovibe.think.learn import train

if __name__ == "__main__":
    db = Database.from_path("audiovibe.db")
    index(db, "data/clips_45sec")
    add_emotion_from_csv(db, "data/annotations/static_annotations.csv")

    train(db)
