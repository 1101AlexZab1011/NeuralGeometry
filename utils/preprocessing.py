from .storage import BasicStorageManager, STAGE
from typing import Optional, Generator, Generator
import mne
import numpy as np
import pandas as pd


EOI = 103 # Id of the events of interest


class BasicPreprocessor(object):
    def __init__(self, storage_manager: BasicStorageManager):
        self.storage_manager = storage_manager

    def __call__(self, stage: Optional[STAGE] = STAGE.PRETEST) -> Generator[tuple[str, mne.Epochs, pd.DataFrame], None, None]:

        for subject_name in self.storage_manager:
            data = list(filter(
                lambda datapath: datapath.stage == stage,
                self.storage_manager.data_paths
            ))[0]
            events = data.events
            sesinfo = data.sesinfo
            selected_trial_nums = events[events[:, 0] == EOI][:, 1]
            sesinfo.drop(np.setxor1d(selected_trial_nums, sesinfo['Trial_n'].to_numpy() - 1), inplace=True)
            trials_sel = np.where(events[:, 0] == 103)[0]
            mask = np.logical_and(sesinfo.Missed == 0, sesinfo.Seed == 1)
            trials_sel = trials_sel[mask]
            sesinfo = sesinfo[mask]
            coords = sesinfo[['SpatialCoordinates_1', 'SpatialCoordinates_2']].to_numpy() - 50

            clusters = np.array(list(map(
                self.define_cluster,
                coords
            )))

            yield subject_name, data.epochs[trials_sel].resample(200), sesinfo, coords.astype(float), clusters.astype(float)

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
