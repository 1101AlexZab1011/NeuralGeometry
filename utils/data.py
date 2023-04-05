from utils.storage import DLStorageIterator
from torch.utils.data import Dataset, ConcatDataset
from deepmeg.data.datasets import EpochsDataset
import os


def get_combined_dataset(iterator: DLStorageIterator, *subjects: str) -> Dataset:

    all_data = list()
    for sbj in subjects:
        iterator.select_subject(sbj)

        if not os.path.exists(iterator.dataset_path):
            raise OSError(f'Dataset for subject "{sbj}" does not exist')

        all_data.append(EpochsDataset.load(iterator.dataset_path))

    return ConcatDataset(all_data)