from .storage import BasicStorageIterator, STAGE, BasicDataDirectory
from typing import Optional, Generator, Generator
import mne
import numpy as np
import pandas as pd
from collections import namedtuple


EOI = 103 # Id of the events of interest


Preprocessed = namedtuple('Preprocessed', 'epochs session_info coordinates clusters')


class BasicPreprocessor(object):
    def __init__(self, eoi: int, resample: Optional[int | float] = None):
        self.eoi = eoi
        self.resample = resample

    def __call__(self, data: BasicDataDirectory) -> Generator[Preprocessed, None, None]:

        events = data.events
        sesinfo = data.sesinfo
        selected_trial_nums = events[events[:, 0] == self.eoi][:, 1]
        sesinfo.drop(np.setxor1d(selected_trial_nums, sesinfo['Trial_n'].to_numpy() - 1), inplace=True)
        trials_sel = np.where(events[:, 0] == self.eoi)[0]
        mask = np.logical_and(sesinfo.Missed == 0, sesinfo.Seed == 1)
        trials_sel = trials_sel[mask]
        sesinfo = sesinfo[mask]
        coords = sesinfo[['SpatialCoordinates_1', 'SpatialCoordinates_2']].to_numpy() - 50

        clusters = np.array(list(map(
            self.define_cluster,
            coords
        )))
        epochs = data.epochs[trials_sel] if self.resample is None else data.epochs[trials_sel].resample(self.resample)

        return Preprocessed(
            epochs,
            sesinfo,
            coords.astype(float),
            clusters.astype(float)
        )

    @staticmethod
    def define_quarter(bool_pair: tuple[bool, bool]) -> int:
        bool_pair = tuple(bool_pair) if not isinstance(bool_pair, tuple) else bool_pair
        posx, posy = bool_pair

        if posx and posy:
            return 1
        elif posx:
            return 0
        elif posy:
            return 2
        else:
            return 3

    @staticmethod
    def define_cluster(coords: tuple[int, int]) -> int:

        dist = np.sqrt(coords[0]**2 + coords[1]**2)

        if dist <= 40:
            return 0
        else:
            return BasicPreprocessor.define_quarter(coords >= 0) + 1
