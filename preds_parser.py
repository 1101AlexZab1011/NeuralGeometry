#! ./venv/bin/python

from deepmeg.preds import PredictionsParser
from deepmeg.params import Predictions
from deepmeg import read_pkl
import matplotlib as mpl
import argparse
import os
from utils.storage import DLStorageIterator
import pandas as pd


if __name__ == '__main__':
    mpl.use('agg')
    parser = argparse.ArgumentParser(
        description='The script for visualising spatial and spectral patterns learned by a LFCNN-like network'
    )
    parser.add_argument('-sd', '--subjects-dir', type=str,
        default=os.path.join(os.getcwd(), 'DATA'),
        help='Path to the subjects directory')
    parser.add_argument('--name', type=str, default='Default_name',
                        help='Name of a task')
    parser.add_argument('--postfix', type=str,
                        default='', help='String to append to a task name')
    parser.add_argument('--prefix', type=str,
                        default='', help='String to set in the start of a task name')

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
    for subject in iterator:

        predictions = read_pkl(os.path.join(iterator.predictions_path, 'y_pred.pkl'))
        sumdf = pd.DataFrame()
        confdf = pd.DataFrame()
        pp = PredictionsParser(predictions.y_true, predictions.y_p)
        sumdf = pd.concat([
                sumdf,
                pp.summary(),
                pd.DataFrame([None for _ in range(pp.summary().shape[0])], index=pp.summary().index),
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