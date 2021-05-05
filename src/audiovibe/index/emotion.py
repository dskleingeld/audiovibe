from dataclasses import dataclass


@dataclass
class Emotion:
    arousal: float
    valence: float

    def as_list(self):
        return [self.arousal, self.valence]

    @staticmethod
    def empty(placeholder=0):
        return Emotion(placeholder, placeholder)
