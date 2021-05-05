import os
import numpy as np
from glob import glob
from tqdm import tqdm  # progress bar
from ..extract.preprocessing import Samples
from ..extract.features import Features
from .database import Database, Row
from .emotion import Emotion


def index(db: Database, dir: str):
    """ dir without ending /"""

    extensions = [".mp3", ".wav"]
    patterns = [dir+"/*"+ext for ext in extensions]
    paths = []
    for p in patterns:
        paths.extend(glob(p))

    for path in tqdm(paths):
        name = os.path.basename(path)
        print(f"name: {name}")
        if db.contains(name):
            continue

        samples = Samples.from_path(path)
        features = Features.from_samples(samples)
        row = Row(features, None)
        db.insert(name, row)


def add_emotion_from_csv(db: Database, csv: str):
    """expects csv with columns: song_id,mean_arousal,
       std_arousal,mean_valence,std_valence"""

    data = np.loadtxt(csv, delimiter=",", skiprows=1)
    for song in data:
        name = str(int(song[0]))+".mp3"
        arousal = song[1]
        valence = song[3]

        emotion = Emotion(arousal, valence)
        db.set_emotion(name, emotion)
