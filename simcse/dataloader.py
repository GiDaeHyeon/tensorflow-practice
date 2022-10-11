"""Triplet DataLoader"""
from pathlib import Path
from keras.utils import Sequence


class SimCSEDataLoader(Sequence):
    def __init__(self, datapath: str) -> None:
        super().__init__()
        self.data = Path(datapath)


