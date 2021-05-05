from dataclasses import dataclass


@dataclass
class Emotion:
    arousal: float
    valence: float

    def as_list(self):
        return [self.arousal, self.valence]
