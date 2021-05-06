import numpy as np
from tensorflow import keras
from ..extract.features import Features
from ..index.emotion import Emotion


def load_model():
    BEST_MODEL = "final_model.tf"
    return keras.models.load_model(BEST_MODEL, compile=True)


def predict(model, features: Features) -> Emotion:
    input = np.array(features.as_list())
    input = np.array([input])
    out = model.predict(input)[0]

    return Emotion(out[0], out[1])
