from contextlib import redirect_stdout
from tensorflow import keras
from tensorflow.keras import layers
from glob import glob
from matplotlib import pyplot as plt
from typing import List
from loguru import logger
import numpy as np
import random

from ..extract.features import Features
from ..index.database import Database, Row


def build_model():
    features = Features.empty().as_list()
    model = keras.Sequential([
        keras.Input(shape=len(features)),
        layers.BatchNormalization(),
        layers.Dense(10, activation="relu"),
        layers.Dense(10, activation="relu"),
        layers.Dense(10, activation="relu"),
        layers.Dense(5, activation="relu"),
        layers.Dense(2, activation="linear"),
    ])
    model.compile(
        loss=keras.losses.MeanAbsoluteError(),
        optimizer=keras.optimizers.RMSprop(),
        metrics=[keras.metrics.MeanAbsoluteError()],
    )
    return model


def prepare_data(data: List[Row]):
    TRAIN_FRACTION = 0.8
    needed = int(len(data)*TRAIN_FRACTION)
    train_idxs = random.sample(range(0, len(data)), needed)
    train_idxs.sort(reverse=True)

    data_train = [data.pop(idx) for idx in train_idxs]

    train_features = [r.features.as_list() for r in data_train]
    train_features = np.array(train_features)
    train_emotions = [r.emotion.as_list() for r in data_train]
    train_emotions = np.array(train_emotions)

    val_features = [r.features.as_list() for r in data]
    val_features = np.array(val_features)
    val_emotions = [r.emotion.as_list() for r in data]
    val_emotions = np.array(val_emotions)

    return train_features, train_emotions, val_features, val_emotions


def plot(history, path: str):
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.ylim(0, 10)
    plt.savefig(path)


def next_file_numb(path: str) -> int:
    files = glob(path+"/loss*.png")
    if len(files) == 0:
        return 0

    def extract_numb(p): return int(p[len(path+"/loss"):-4])
    numbs = map(extract_numb, files)
    return max(numbs) + 1


def store_model_info(model, hist):
    dir = "models_tried"
    numb = next_file_numb(dir)
    plot(hist.history, path=dir+f"/loss{numb}.png")

    with open(dir+f"/model{numb}.txt", "w") as f:
        with redirect_stdout(f):
            model.summary()


def train(db: Database):
    data = db.get_with_emotion()
    train_in, train_out, val_in, val_out = prepare_data(data)

    model = build_model()
    logger.info(f"training on: {len(train_in)} samples")
    hist = model.fit(train_in, train_out, epochs=80, verbose=0,
                     validation_data=(val_in, val_out))

    store_model_info(model, hist)
    logger.info(f"best loss reached: {hist.history['val_loss'][-1]}")
