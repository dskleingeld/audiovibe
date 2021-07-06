from dataclasses import dataclass
import numpy as np
import math


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


@dataclass(repr=False)
class Emotion:
    arousal: float
    valence: float

    def as_list(self):
        return [self.arousal, self.valence]

    @staticmethod
    def empty(placeholder=0):
        return Emotion(placeholder, placeholder)

    def angle(self) -> float:
        x = self.valence - 0.5
        y = self.arousal - 0.5
        angle_rel_x_axis = math.atan2(y, x)/math.pi*180
        return angle_rel_x_axis

    def weight(self) -> float:
        return abs(self.valence) + abs(self.arousal)

    def as_word(self) -> str:
        ANGLE = [15, 40, 65, 115, 140, 160, ]
        ANGLE_TO_IDX = ANGLE
        ANGLE_TO_IDX.extend([-a for a in ANGLE])
        IDX_TO_EMOTION = [
            "Happy",
            "Delighted",
            "Excited",
            "Tense",
            "Angry",
            "Frustrated",
            "Content",
            "Relaxed",
            "Calm",
            "Tired",
            "Bored",
            "Depressed",
        ]

        idx = find_nearest(ANGLE_TO_IDX, self.angle())
        return IDX_TO_EMOTION[idx]

    def __str__(self):
        return (f"{self.__class__.__name__}("
                + f"{self.as_word()}, angle: {self.angle()}, "
                + f"arousal: {self.arousal-0.5}, valence: {self.valence-0.5}, "
                + f"weight: {self.weight()})")

    def __repr__(self):
        return self.__str__()
