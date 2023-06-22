#! ./venv/bin/python

import matplotlib as mpl
import argparse
import os
from utils.models import get_model_by_name
from utils.storage import DLStorageIterator, STAGE
from utils.preprocessing import BasicPreprocessor
import numpy as np
import pandas as pd
from time import perf_counter
import re
import logging
from utils import accuracy, balance
from deepmeg.models.interpretable import LFCNN, HilbertNet
from deepmeg.experimental.models import SPIRIT, FourierSPIRIT, CanonicalSPIRIT, LFCNNW
from deepmeg.interpreters import LFCNNInterpreter
from deepmeg.experimental.interpreters import SPIRITInterpreter, LFCNNWInterpreter
from deepmeg.data.datasets import EpochsDataset
from deepmeg.preprocessing.transforms import one_hot_encoder, zscore
from deepmeg.training.callbacks import PrintingCallback, EarlyStopping, L2Reg, Callback, VisualizingCallback
from deepmeg.training.trainers import Trainer
from deepmeg.utils.params import Predictions, save, LFCNNParameters
from deepmeg.experimental.params import SPIRITParameters
import torch
from torch.utils.data import DataLoader
import torchmetrics
from utils import PenalizedEarlyStopping, TempConvAveClipping, IndependanceConstraint


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
    parser.add_argument('--name', type=str, default='Default_name',
                        help='Name of a task')
    parser.add_argument('--postfix', type=str,
                        default='', help='String to append to a task name')
    parser.add_argument('--prefix', type=str,
                        default='', help='String to set in the start of a task name')
    parser.add_argument('--project-name', type=str,
                        default='mem_arch_epochs', help='Name of a project')
    parser.add_argument('--no-params', action='store_true', help='Do not compute parameters')
    parser.add_argument('--not-save-params', action='store_true', help='Do not save parameters')
    parser.add_argument('--balance', action='store_true', help='Balance classes')
    parser.add_argument('-t', '--target', type=str, help='Target to predict (must be a column from sesinfo csv file)')
    parser.add_argument('-k', '--kind', type=str, help='Spatial (sp) or conceptual (con) or both "spccon"', default='spcon')
    parser.add_argument('-st', '--stage', type=str, help='Training (train), PreTest (pre) or PostTest (post) or both "prepost"', default='prepost')
    parser.add_argument('-cf', '--crop-from', type=float, help='Crop epoch from time', default=0.)
    parser.add_argument('-ct', '--crop-to', type=float, help='Crop epoch to time', default=None)
    parser.add_argument('-bf', '--bl-from', type=float, help='Baseline epoch from time', default=None)
    parser.add_argument('-bt', '--bl-to', type=float, help='Baseline epoch to time', default=0.)
    parser.add_argument('-m', '--model', type=str, help='Model to use', default='lfcnn')
    parser.add_argument('-d', '--device', type=str, help='Device to use', default='cuda')
    parser.add_argument('-l', '--lock', type=str, help='Clue lock (clue), feedback lock (feedback) or stimulus lock (stim)', default='stim')

    excluded_subjects, \
        from_, \
        to, \
        subjects_dir, \
        classification_name,\
        classification_postfix,\
        classification_prefix, \
        project_name, \
        no_params, \
        not_save_params, \
        balance_classes, \
        target_col_name,\
        kind,\
        stage,\
        crop_from, crop_to,\
        bl_from, bl_to,\
        model_name,\
        device,\
        lock = vars(parser.parse_args()).values()

    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

    n_latent = 8

    match lock:
        case 'clue':
            lock = 102
        case 'stim':
            lock = 103
        case 'feedback':
            lock = 105
        case _:
            raise ValueError(f'Invalid lock: {lock}')

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
    for subject_name in iterator:
        logging.debug(f'Processing subject: {subject_name}')
        subject_num = int(re.findall(r'\d+', subject_name)[0])

        if (subject_num in excluded_subjects) or\
            (from_ and subject_num < from_) or\
            (to and subject_num > to):
            logging.debug(f'Skipping subject {subject_name}')
            continue

        sp_preprocessor = BasicPreprocessor(lock, 200)
        con_preprocessor = BasicPreprocessor(lock, 200, 2)
        preprcessed = list()
        if 'sp' in kind:
            if 'pre' in stage:
                preprcessed.append(sp_preprocessor(iterator.get_data(STAGE.PRETEST)))
            if 'post' in stage:
                preprcessed.append(sp_preprocessor(iterator.get_data(STAGE.POSTTEST)))
            if 'train' in stage:
                preprcessed.append(sp_preprocessor(iterator.get_data(STAGE.TRAINING)))
        if 'con' in kind:
            if 'pre' in stage:
                preprcessed.append(con_preprocessor(iterator.get_data(STAGE.PRETEST)))
            if 'post' in stage:
                preprcessed.append(con_preprocessor(iterator.get_data(STAGE.POSTTEST)))
            if 'train' in stage:
                preprcessed.append(con_preprocessor(iterator.get_data(STAGE.TRAINING)))
        if not preprcessed:
            raise ValueError(f'No data selected. Your config is: {kind = }, {stage = }')

        info = preprcessed[0].epochs.pick_types(meg='grad').info
        if bl_from:
            X = np.concatenate([
                data.
                epochs.
                pick_types(meg='grad').
                apply_baseline((bl_from, bl_to)).
                crop(crop_from, crop_to).
                get_data()
                for data in preprcessed
                ])
        else:
            X = np.concatenate([
                data.
                epochs.
                pick_types(meg='grad').
                crop(crop_from, crop_to).
                get_data()
                for data in preprcessed
                ])
        Y = np.concatenate([data.session_info[target_col_name].to_numpy() for data in preprcessed])

        if balance_classes:
            X, Y = balance(X, Y)

        n_classes, classes_samples = np.unique(Y, return_counts=True)
        n_classes = len(n_classes)
        classes_samples = classes_samples.tolist()
        Y = one_hot_encoder(Y)
        dataset = EpochsDataset((X, Y), transform=zscore, savepath=iterator.dataset_content_path)
        dataset.save(iterator.dataset_path)
        train, test = torch.utils.data.random_split(dataset, [.7, .3])

        model, interpretation, parametrizer = get_model_by_name(model_name, X, Y)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0001)
        loss = torch.nn.BCEWithLogitsLoss()
        # metric = torchmetrics.functional.classification.binary_accuracy
        metric = ('acc', accuracy)
        model.compile(
            optimizer,
            loss,
            metric,
            callbacks=[
                # PrintingCallback(),
                VisualizingCallback(
                    width = 60,
                    height = 10,
                    n_epochs=150,
                    loss_colors=['magenta', 'yellow'],
                    metric_colors=['green', 'blue'],
                    metric_label='Accuracy',
                    loss_label='BCELoss',
                    metric_names=['acc_train', 'acc_val'],
                    loss_names=['loss_train', 'loss_val'],
                ),
                # IndependanceConstraint(8),
                TempConvAveClipping(),
                # EarlyStopping(monitor='loss_val', patience=15, restore_best_weights=True),
                PenalizedEarlyStopping(monitor='loss_val', measure='acc_val', patience=15, restore_best_weights=True),
                L2Reg(
                    [
                        'unmixing_layer.weight', 'temp_conv.weight',
                    ], lambdas=.01
                )
            ],
            device=device
        )

        t1 = perf_counter()
        history = model.fit(train, n_epochs=150, batch_size=100, val_batch_size=60)
        runtime = perf_counter() - t1

        x_train, y_true_train = next(iter(DataLoader(train, len(train))))
        y_pred_train = torch.squeeze(model(x_train)).detach().numpy()
        x_test, y_true_test = next(iter(DataLoader(test, len(test))))
        y_pred_test = torch.squeeze(model(x_test)).detach().numpy()

        save(
            Predictions(
                y_pred_test,
                y_true_test
            ),
            iterator.predictions_path
        )

        train_result = model.evaluate(train)
        result = model.evaluate(test)

        train_loss_, train_acc_ = train_result.values()
        test_loss_, test_acc_ = result.values()
        logging.info(f'{subject_name}\nClassification results:\n \tRUNTIME: {runtime}\n\tTRAIN_ACC: {train_acc_}\n\tTEST_ACC: {test_acc_}')

        if not no_params:
            logging.debug('Computing parameters')
            interpreter = interpretation(model, test, info)

            if not not_save_params:
                params = parametrizer(interpreter)
                params.save(iterator.parameters_path)

            for i in range(n_latent):
                fig = interpreter.plot_branch(i)
                fig.savefig(os.path.join(iterator.pics_path, f'Branch_{i}.png'), dpi=300)

        perf_table_path = os.path.join(
            iterator.history_path,
            f'{classification_name_formatted}.csv'
        )
        processed_df = pd.Series(
            [
                n_classes,
                *classes_samples,
                sum(classes_samples),
                len(test),
                train_acc_,
                train_loss_,
                test_acc_,
                test_loss_,
                runtime
            ],
            index=[
                'n_classes',
                *[str(i) for i in range(len(classes_samples))],
                'total',
                'test_set',
                'train_acc',
                'train_loss',
                'test_acc',
                'test_loss',
                'runtime'
            ],
            name=subject_name
        ).to_frame().T

        if os.path.exists(perf_table_path):
            pd.concat([pd.read_csv(perf_table_path, index_col=0, header=0), processed_df], axis=0)\
                .to_csv(perf_table_path)
        else:
            processed_df.to_csv(perf_table_path)

        logging.info(f'Processing of subject {subject_name} is done')
    logging.info('All subjects are processed')
