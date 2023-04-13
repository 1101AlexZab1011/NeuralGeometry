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
from utils import TempConvAveClipping
from deepmeg.training.trainers import Trainer
from deepmeg.utils.params import Predictions, save, LFCNNParameters
from deepmeg.experimental.params import SPIRITParameters
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchmetrics
from utils import PenalizedEarlyStopping
from deepmeg.utils.convtools import conviter
from deepmeg.utils import check_path
from deepmeg.utils.viz import plot_metrics
from training_pipeline import moving_average
from deepmeg.utils import check_path
from deepmeg.utils.viz import plot_metrics
from utils.data import get_combined_dataset
from utils.models import get_model_by_name


if __name__ == '__main__':
    mpl.use('agg')
    parser = argparse.ArgumentParser(
        description='The script for applying the neural network "LF-CNN" to the '
        'epoched data from gradiometers related to events for classification'
    )
    parser.add_argument('-es', '--exclude-subjects', type=int, nargs='+',
                        default=[], help='IDs of subjects to exclude')
    parser.add_argument('-gs', '--generalize-subjects', type=int, nargs='+',
                        default=[], help='IDs of subjects to generalize but do not include to training sets')
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
    parser.add_argument('--project-name', type=str,
                        default='mem_arch_epochs', help='Name of a project')
    parser.add_argument('--no-params', action='store_true', help='Do not compute parameters')
    parser.add_argument('-k', '--kind', type=str, help='Data to generalize: spatial (sp) or conceptual (con)', default='spcon')
    parser.add_argument('-cf', '--crop-from', type=float, help='Crop epoch from time', default=0.)
    parser.add_argument('-ct', '--crop-to', type=float, help='Crop epoch to time', default=None)
    parser.add_argument('-bf', '--bl-from', type=float, help='Baseline epoch from time', default=None)
    parser.add_argument('-bt', '--bl-to', type=float, help='Baseline epoch to time', default=0.)
    parser.add_argument('-m', '--model', type=str, help='Model to use', default='lfcnn')
    parser.add_argument('-d', '--device', type=str, help='Device to use', default='cuda')


    excluded_subjects, \
        generalized_subjects,\
        from_, \
        to, \
        subjects_dir, \
        classification_name,\
        classification_postfix,\
        classification_prefix, \
        project_name, \
        no_params, \
        kind,\
        crop_from, crop_to,\
        bl_from, bl_to,\
        model_name,\
        device = vars(parser.parse_args()).values()

    win = 20
    n_latent = 8
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
    logging.info(f'Current classification: {classification_name_formatted}')

    iterator = DLStorageIterator(subjects_dir, name=classification_name_formatted)
    info = None
    training_subjects, testing_subjects = list(), list()

    for subject_name in iterator:
        subject_num = int(re.findall(r'\d+', subject_name)[0])
        if not os.path.exists(iterator.dataset_path) or info is None:
            logging.debug(f'Processing subject: {subject_name}')

            if (subject_num in excluded_subjects) or\
                (from_ and subject_num < from_) or\
                (to and subject_num > to):
                logging.debug(f'Skipping subject {subject_name}')
                continue

            # sp_preprocessor = BasicPreprocessor(103, 200)
            # con_preprocessor = BasicPreprocessor(103, 200, 2)
            sp_preprocessor = BasicPreprocessor(105, 200)
            con_preprocessor = BasicPreprocessor(105, 200, 2)
            preprcessed = list()
            if 'sp' in kind:
                preprcessed.append(sp_preprocessor(iterator.get_data(STAGE.TRAINING)))
            if 'con' in kind:
                preprcessed.append(con_preprocessor(iterator.get_data(STAGE.TRAINING)))
            if not preprcessed:
                raise ValueError(f'No data selected. Your config is: {kind = }')

            info = preprcessed[0].epochs.pick_types(meg='grad').info if info is None else info
            X = np.concatenate([
                data.
                epochs.
                pick_types(meg='grad').
                apply_baseline((bl_from, bl_to)).
                crop(crop_from, crop_to).
                get_data()
                for data in preprcessed
            ])
            feedbacks = [data.session_info.Feedback.to_numpy() for data in preprcessed]
            feedbacks_np = moving_average(np.concatenate(feedbacks), win)
            min_, max_ = feedbacks_np.min(), feedbacks_np.max()
            minmax = lambda x, min_, max_: (x - min_)/(max_ - min_)
            acc = [
                np.expand_dims(
                    minmax(
                        moving_average(
                            data.session_info.Feedback, win
                        ),
                        min_, max_
                    ),
                    1
                )
                for data in preprcessed
            ]
            Y = np.concatenate(acc)

            dataset = EpochsDataset((X, Y), transform=zscore, savepath=iterator.dataset_content_path)
            dataset.save(iterator.dataset_path)
            logging.debug(f'Dataset for subject {subject_name} has been made ({len(dataset)} samples)')

            if subject_num in generalized_subjects:
                testing_subjects.append(subject_name)
                logging.debug(f'Subject {subject_name} is added to testing group')
            else:
                training_subjects.append(subject_name)
                logging.debug(f'Subject {subject_name} is added to training group')
        else:
            logging.debug(f'Dataset for subject {subject_name} is already exists')

            if subject_num in generalized_subjects:
                testing_subjects.append(subject_name)
                logging.debug(f'Subject {subject_name} is added to testing group')
            else:
                training_subjects.append(subject_name)
                logging.debug(f'Subject {subject_name} is added to training group')

    perf_table_path = os.path.join(
        iterator.history_path,
        f'{classification_name_formatted}.csv'
    )

    for selected_subject in training_subjects:
        logging.debug(f'Selected subject: {selected_subject}')
        current_training = list(set(training_subjects) - {selected_subject})
        current_testing = testing_subjects + [selected_subject]
        logging.debug(f'\ntraining group:\n{current_training}\ntesting group:\n{current_testing}')
        logging.debug('Preparing training set')
        dataset = get_combined_dataset(iterator, *current_training)
        train, test = torch.utils.data.random_split(dataset, [.7, .3])
        X, Y = next(iter(DataLoader(train, 2)))

        sbj_test_datasets = list()
        logging.debug('Preparing testing sets')
        for sbj in current_testing:
            iterator.select_subject(sbj)
            sbj_test_datasets.append(EpochsDataset.load(iterator.dataset_path))

        model, _, _ = get_model_by_name(model_name, X, Y)
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.0001)
        loss = torch.nn.MSELoss()
        metric = ('mae', torch.nn.L1Loss())
        model.compile(
            optimizer,
            loss,
            metric,
            callbacks=[
                PrintingCallback(),
                TempConvAveClipping(),
                EarlyStopping(monitor='loss_val', patience=15, restore_best_weights=True),
                L2Reg(
                    [
                        'unmixing_layer.weight', 'temp_conv.weight',
                    ], lambdas=.01
                )
            ],
            device=device
        )

        t1 = perf_counter()
        history = model.fit(train, n_epochs=150, batch_size=200, val_batch_size=60)
        runtime = perf_counter() - t1
        figs = plot_metrics(history)
        for fig, name in zip(
            figs,
            set(map(lambda name: name.split('_')[0], history.keys()))
        ):
            fig.savefig(os.path.join(iterator.pics_path, f'{name}.png'), dpi=300)

        train_result = model.evaluate(train)
        result = model.evaluate(test)
        train_loss_, train_acc_ = train_result.values()
        test_loss_, test_acc_ = result.values()

        subject_accs = list()
        for subject_name, subject_data in zip(current_testing, sbj_test_datasets):
            result = model.evaluate(subject_data)
            subject_loss, subject_acc = result.values()
            subject_accs.append(subject_acc)

        processed_df = pd.Series(
            [
                len(train),
                len(test),
                train_acc_,
                test_acc_,
                runtime,
                *subject_accs
            ],
            index=[
                'train_set',
                'test_set',
                'train_mae',
                'test_mae',
                'runtime',
                *[f'{sbj}_mae' for sbj in current_testing]
            ],
            name=selected_subject
        ).to_frame().T

        if os.path.exists(perf_table_path):
            pd.concat([pd.read_csv(perf_table_path, index_col=0, header=0), processed_df], axis=0)\
                .to_csv(perf_table_path)
        else:
            processed_df.to_csv(perf_table_path)
    logging.info('All subjects are processed')
