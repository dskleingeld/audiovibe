import os
import numpy as np
from glob import glob
from typing import Tuple
from tqdm.contrib.concurrent import process_map
from functools import partial
from multiprocessing import Lock

from ..extract.preprocessing import Samples
from ..extract.features import Features
from .database import Database, Row
from .emotion import Emotion


def analyze(path: str) -> Tuple[str, Row]:
    samples = Samples.from_path(path)
    features = Features.from_samples(samples)
    row = Row(features, None)
    name = os.path.basename(path)
    return name, row


def index(db: Database, dir: str, max_workers=None):
    """ index all files in dir, pass dir as string without ending /
        reading and analyzing files is done in parallel, pass
        max workers to limit number of workers used (default cpu_count() + 4)
    """

    extensions = [".mp3", ".wav", ".m4a"]
    patterns = [dir+"/*"+ext for ext in extensions]
    paths = []
    for p in patterns:
        paths.extend(glob(p))
    paths = [p for p in paths if not db.contains(os.path.basename(p))]
    for name, row in process_map(analyze, paths, max_workers=max_workers):
        print("writing to db")
        db.insert(name, row)


# better dataset: https://cvml.unige.ch/databases/DEAM/
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
