from dataclasses import dataclass
import numpy as np
import math


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


@dataclass
class Emotion:
    arousal: float
    valence: float

    def as_list(self):
        return [self.arousal, self.valence]

    @staticmethod
    def empty(placeholder=0):
        return Emotion(placeholder, placeholder)

    def as_word(self) -> str:
        # TODO
        x = self.valence - 0.5
        y = self.arousal - 0.5
        angle = math.atan2(y, x)/math.pi*180  # in degees

        ANGLE_TO_IDX = [30, 60, 120, 150, 210, 250, 290, 330]
        IDX_TO_EMOTION = [
            "excited",
            "happy",
            "content",
            "calm",
            "depressed",
            "sad",
            "afraid",
            "angry",
        ]

        idx = find_nearest(ANGLE_TO_IDX, angle)
        return IDX_TO_EMOTION[idx]
