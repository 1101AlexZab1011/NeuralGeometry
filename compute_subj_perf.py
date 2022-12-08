#! ./venv/bin/python

import matplotlib as mpl
import argparse
import os
from utils.storage import DLStorageIterator, STAGE
from utils.preprocessing import BasicPreprocessor, Preprocessed
import numpy as np
import mneflow as mf
from utils.models import SimpleNet
import tensorflow as tf
import pandas as pd
from time import perf_counter
from sklearn.metrics import accuracy_score
from deepmeg.params import save_parameters, compute_temporal_parameters, compute_waveforms, \
    Predictions, WaveForms, TemporalParameters, SpatialParameters, ComponentsOrder, get_order
import re
import logging


if __name__ == '__main__':
    mpl.use('agg')
    parser = argparse.ArgumentParser(
        description='The script for applying the neural network "SimpleNet" to the '
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

    excluded_subjects, \
        from_, \
        to, \
        subjects_dir = vars(parser.parse_args()).values()

    iterator = DLStorageIterator(subjects_dir, name='Default Name')
    subject_names, pretest_acc, traininig_acc, posttest_acc = list(), list(), list(), list()
    for subject_name in iterator:
        subject_num = int(re.findall(r'\d+', subject_name)[0])

        if (subject_num in excluded_subjects) or\
            (from_ and subject_num < from_) or\
            (to and subject_num > to):
            logging.debug(f'Skipping subject {subject_name}')
            continue

        pretest = iterator.get_data(STAGE.PRETEST).sesinfo
        pretest = pretest[pretest.Missed == 0]
        # training = iterator.get_data(STAGE.TRAINING).sesinfo
        # training = pretest[training.Missed == 0]
        posttest = iterator.get_data(STAGE.POSTTEST).sesinfo
        posttest = posttest[posttest.Missed == 0]
        subject_names.append(subject_name)
        pretest_acc.append(pretest.Feedback.sum()/pretest.Feedback.shape[0])
        # traininig_acc.append(training.Feedback.sum()/training.Feedback.shape[0])
        posttest_acc.append(posttest.Feedback.sum()/posttest.Feedback.shape[0])
    pd.DataFrame(
        list(zip(pretest_acc, traininig_acc, posttest_acc)),
        columns=['PRETEST', 'TRAINING', 'POSTTEST'],
        index=subject_names
    ).to_csv(
        os.path.join(
            iterator.history_path,
            'subjects_perf.csv'
        )
    )
