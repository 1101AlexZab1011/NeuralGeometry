#! ./venv/bin/python

from deepmeg.preds import PredictionsParser
from deepmeg.params import Predictions
from deepmeg import read_pkl
import matplotlib as mpl
import argparse
import os
from utils.storage import DLStorageIterator
import pandas as pd
import re
import numpy as np


if __name__ == '__main__':
    mpl.use('agg')
    parser = argparse.ArgumentParser(
        description='The script for visualising spatial and spectral patterns learned by a LFCNN-like network'
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
    parser.add_argument('--name', type=str, default='Default_name',
                        help='Name of a task')
    parser.add_argument('--postfix', type=str,
                        default='', help='String to append to a task name')
    parser.add_argument('--prefix', type=str,
                        default='', help='String to set in the start of a task name')

    excluded_subjects, \
        from_, \
        to, \
        subjects_dir,\
        classification_name,\
        classification_postfix,\
        classification_prefix = vars(parser.parse_args()).values()

    classification_name_formatted = "_".join(list(filter(
        lambda s: s not in (None, ""),
        [
            classification_prefix,
            classification_name,
            classification_postfix
        ]
    )))

    iterator = DLStorageIterator(subjects_dir, classification_name_formatted)
    all_sumdf = list()
    all_confdf = list()
    for subject_name in iterator:
        subject_num = int(re.findall(r'\d+', subject_name)[0])

        if (subject_num in excluded_subjects) or\
            (from_ and subject_num < from_) or\
            (to and subject_num > to):
            continue

        predictions = read_pkl(os.path.join(iterator.predictions_path, 'y_pred.pkl'))

        sumdf = pd.DataFrame()
        confdf = pd.DataFrame()
        pp = PredictionsParser(predictions.y_true, predictions.y_p)
        sumdf = pd.concat([
                sumdf,
                pp.summary(),
            ],
            axis=1
        )
        confdf = pd.concat([
                confdf,
                pp.confusion,
                pd.DataFrame(list(confdf.index), index=confdf.index),
            ],
            axis=1
        )
        sumdf.to_csv(os.path.join(iterator.predictions_path, f'summary.csv'))
        confdf.to_csv(os.path.join(iterator.predictions_path, f'confusion.csv'))
        all_sumdf.append(sumdf)
        all_confdf.append(confdf)

    pd.DataFrame(
        data=np.array([df.to_numpy() for df in all_sumdf]).mean(0),
        columns=all_sumdf[0].columns,
        index=all_sumdf[0].index
    )\
        .to_csv(os.path.join(iterator.history_path, f'{classification_name_formatted}_summary.csv'))
    pd.DataFrame(
        data=np.array([df.to_numpy() for df in all_confdf]).mean(0),
        columns=all_confdf[0].columns,
        index=all_confdf[0].index
    )\
        .to_csv(os.path.join(iterator.history_path, f'{classification_name_formatted}_confusion.csv'))
