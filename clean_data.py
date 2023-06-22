#! ./venv/bin/python

import matplotlib as mpl
import argparse
import os
from utils.storage import DLStorageIterator, STAGE
from utils.preprocessing import BasicPreprocessor
import numpy as np
import pandas as pd
from time import perf_counter
import re
import logging
from utils import balance
from deepmeg.models.interpretable import LFCNN, HilbertNet
from deepmeg.experimental.models import SPIRIT, FourierSPIRIT, CanonicalSPIRIT, LFCNNW
from deepmeg.interpreters import LFCNNInterpreter
from deepmeg.experimental.interpreters import SPIRITInterpreter, LFCNNWInterpreter
from deepmeg.data.datasets import EpochsDataset
from deepmeg.preprocessing.transforms import one_hot_encoder, zscore
from deepmeg.training.callbacks import PrintingCallback, EarlyStopping, L2Reg, Callback
from deepmeg.training.trainers import Trainer
from deepmeg.utils.params import Predictions, save, LFCNNParameters
from deepmeg.experimental.params import SPIRITParameters
import torch
from torch.utils.data import DataLoader
import torchmetrics
from utils import PenalizedEarlyStopping
import shutil


if __name__ == '__main__':
    mpl.use('agg')
    parser = argparse.ArgumentParser(
        description='The script for applying the neural network "LF-CNN" to the '
        'epoched data from gradiometers related to events for classification'
    )
    parser.add_argument('-es', '--exclude-subjects', type=int, nargs='+',
                        default=[], help='IDs of subjects to exclude')
    parser.add_argument('-from', type=int,
                        default=None, help='ID of a subject to start from')
    parser.add_argument('-to', type=int,
                        default=None, help='ID of a subject to end with')
    parser.add_argument('-sd', '--subjects-dir', type=str,
                        default=os.path.join(os.getcwd(), 'DATA'),
                        help='Path to the subjects directory')
    parser.add_argument('--names', nargs='+', type=str, default=None,
                        help='Name of a task')
    parser.add_argument('--postfix', type=str,
                        default='', help='String to append to a task name')
    parser.add_argument('--prefix', type=str,
                        default='', help='String to set in the start of a task name')
    parser.add_argument('--project-name', type=str,
                        default='mem_arch_epochs', help='Name of a project')
    parser.add_argument('--clean-params', action='store_true', help='To remove parameters')

    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


    excluded_subjects, \
        from_, \
        to, \
        subjects_dir, \
        classification_names,\
        classification_postfix,\
        classification_prefix, \
        project_name,\
        remove_params = vars(parser.parse_args()).values()

    for classification_name in classification_names:

        classification_name_formatted = "_".join(list(filter(
            lambda s: s not in (None, ""),
            [
                classification_prefix,
                classification_name,
                classification_postfix
            ]
        )))
        logging.basicConfig(
            format='%(asctime)s %(levelname)s:\t%(message)s',
            filename=f'./logs/{classification_name_formatted}.log',
            encoding='utf-8',
            level=logging.DEBUG
        )
        logging.info(f'Clean data for: {classification_name_formatted}')

        iterator = DLStorageIterator(subjects_dir, name=classification_name_formatted)
        for subject_name in iterator:
            logging.debug(f'Processing subject: {subject_name}')
            subject_num = int(re.findall(r'\d+', subject_name)[0])

            if (subject_num in excluded_subjects) or\
                (from_ and subject_num < from_) or\
                (to and subject_num > to):
                logging.debug(f'Skipping subject {subject_name}')
                continue

            if remove_params:
                logging.debug(f'Remove all results')
                shutil.rmtree(iterator.subject_results_path)
            else:
                logging.debug(f'Remove dataset')
                shutil.rmtree(iterator.dataset_content_path)
