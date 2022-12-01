import os
from enum import Enum
import logging
import pandas as pd
import numpy as np
import mne


STAGE = Enum('STAGE', ['POSTTEST', 'PRETEST', 'TRAINING'])


class BasicDataDirectory:
    def __init__(self, path: os.PathLike):
        self.path = os.fspath(path)
        self.name = os.path.basename(path)

        match self.name:
            case 'PostTest':
                self.stage = STAGE.POSTTEST
            case 'PreTest':
                self.stage = STAGE.PRETEST
            case 'Training':
                self.stage = STAGE.TRAINING
            case '_':
                raise OSError(f'Unexpected directory: {self.path}')

        self.epochs_path = os.path.join(self.path, f'{self.name}_epochs_sel.fif')
        self.events_path = os.path.join(self.path, f'{self.name}_EventFile_trls_sel.npy')
        sesinfo_file_name = f'{os.path.basename(os.path.dirname(self.path)).replace("u", "")}_{self.name.lower()}'
        self.sesinfo_path = os.path.join(self.path, f'{sesinfo_file_name[:3] + sesinfo_file_name[4:]}.csv')

    def __str__(self) -> str:
        return f'BasicDataDirectory at {self.path}'

    @property
    def epochs(self):
        logging.info(f'Reading epochs from {self.epochs_path}')
        return mne.read_epochs(self.epochs_path)

    @property
    def events(self):
        logging.info(f'Reading events from {self.events_path}')
        return np.load(self.events_path)

    @property
    def sesinfo(self):
        logging.info(f'Reading sesinfo from {self.sesinfo_path}')
        return pd.read_csv(self.sesinfo_path)


class BasicStorageManager:
    def __init__(self, subjects_dir: str):
        self.subjects_dir = subjects_dir
        logging.info(f'Initialize storage management for {self.subjects_dir}')
        self.subject_dirs = os.listdir(subjects_dir)
        self.subject_path = None
        self.data_paths = None
        self.__current_subject_index = 0

    def __str__(self) -> str:
        return f'BasicStorageManager for {self.subjects_dir}'

    def select_subject(self, subject_name: str):
        logging.info(f'Select subject {subject_name}')
        self.subject_path = os.path.join(self.subjects_dir, subject_name)
        self.data_paths = [
            BasicDataDirectory(path)
            for file_name in os.listdir(self.subject_path) if os.path.isdir(path := os.path.join(self.subject_path, file_name))
        ]

    def __iter__(self):
        self.__current_subject_index = 0
        return self

    def __next__(self):
        if self.__current_subject_index < len(self.subject_dirs):
            self.select_subject(self.subject_dirs[self.__current_subject_index])
            subject_name = self.subject_dirs[self.__current_subject_index]
            self.__current_subject_index += 1
            return subject_name
        else:
            raise StopIteration
