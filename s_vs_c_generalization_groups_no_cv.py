#! ./venv/bin/python

import matplotlib as mpl
import argparse
import os
from utils.data import get_combined_dataset
from utils.models import get_model_by_name
from utils.storage import DLStorageIterator, STAGE
from utils.preprocessing import BasicPreprocessor
import numpy as np
import pandas as pd
from time import perf_counter
import re
import logging
from utils import TempConvAveClipping, accuracy, balance
from deepmeg.data.datasets import EpochsDataset
from deepmeg.preprocessing.transforms import one_hot_encoder, zscore, one_hot_decoder
from deepmeg.training.callbacks import PrintingCallback, EarlyStopping, L2Reg, VisualizingCallback
import torch
from torch.utils.data import DataLoader
from deepmeg.utils.params import Predictions, save
import torchmetrics
from utils import PenalizedEarlyStopping
from deepmeg.utils.viz import plot_metrics


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
    parser.add_argument('--not-save-params', action='store_true', help='Do not save parameters')
    parser.add_argument('--balance', action='store_true', help='Balance classes')
    parser.add_argument('-st', '--stage', type=str, help='PreTest (pre) or PostTest (post) or both "prepost"', default='prepost')
    parser.add_argument('-cf', '--crop-from', type=float, help='Crop epoch from time', default=0.)
    parser.add_argument('-ct', '--crop-to', type=float, help='Crop epoch to time', default=None)
    parser.add_argument('-bf', '--bl-from', type=float, help='Baseline epoch from time', default=None)
    parser.add_argument('-bt', '--bl-to', type=float, help='Baseline epoch to time', default=0.)
    parser.add_argument('-m', '--model', type=str, help='Model to use', default='lfcnn')
    parser.add_argument('-d', '--device', type=str, help='Device to use', default='cuda')
    parser.add_argument('-l', '--lock', type=str, help='Clue lock (clue), feedback lock (feedback) or stimulus lock (stim)', default='stim')


    excluded_subjects, \
        generalized_subjects, \
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
    all_data = list()
    info = None
    training_subjects, testing_subjects = list(), list()
    n_classes = None
    all_classes_samples = None
    for subject_name in iterator:
        if not os.path.exists(iterator.dataset_path) or info is None:
            logging.debug(f'Processing subject: {subject_name}')
            subject_num = int(re.findall(r'\d+', subject_name)[0])

            if (subject_num in excluded_subjects) or\
                (from_ and subject_num < from_) or\
                (to and subject_num > to):
                logging.debug(f'Skipping subject {subject_name}')
                continue

            sp_preprocessor = BasicPreprocessor(lock, 200)
            con_preprocessor = BasicPreprocessor(lock, 200, 2)
            preprocessed = list()
            labels = list()
            for preprocessor, label_gen in zip([sp_preprocessor, con_preprocessor], [np.zeros, np.ones]):
                if 'pre' in stage:
                    data_pre = preprocessor(iterator.get_data(STAGE.PRETEST))
                    preprocessed.append(data_pre)
                    labels.append(label_gen((len(data_pre.clusters),)))
                if 'post' in stage:
                    data_post = preprocessor(iterator.get_data(STAGE.POSTTEST))
                    preprocessed.append(data_post)
                    labels.append(label_gen((len(data_post.clusters),)))
                if 'train' in stage:
                    data_train = preprocessor(iterator.get_data(STAGE.TRAINING))
                    preprocessed.append(data_train)
                    labels.append(label_gen((len(data_train.clusters),)))
            if not preprocessed:
                raise ValueError(f'No data selected. Your config is: {stage = }')

            X = np.concatenate([
                data.
                epochs.
                pick_types(meg='grad').
                apply_baseline((bl_from, bl_to)).
                crop(crop_from, crop_to).
                get_data()
                for data in preprocessed
            ])
            sesinfo = pd.concat([data.session_info for data in preprocessed], axis=0)
            sesinfo.to_csv(os.path.join(iterator.subject_results_path, 'session_info.csv'))
            Y = np.concatenate(labels)
            logging.debug('Collecting dataset information...')
            info = preprocessed[0].epochs.pick_types(meg='grad').info

            if balance_classes:
                X, Y = balance(X, Y)

            logging.debug(f'Final number of samples: {len(X)}')

            Y = one_hot_encoder(Y)

            if n_classes is None and all_classes_samples is None:
                n_classes, classes_samples = np.unique(Y, return_counts=True)
                n_classes = len(n_classes)
                classes_samples = classes_samples.tolist()
                all_classes_samples = classes_samples
            elif n_classes is not None and all_classes_samples is not None:
                _, classes_samples = np.unique(Y, return_counts=True)
                classes_samples = classes_samples.tolist()
                all_classes_samples = [x + y for x, y in zip(all_classes_samples, classes_samples)]

            dataset = EpochsDataset((X, Y), transform=zscore, savepath=iterator.dataset_content_path)
            dataset.save(iterator.dataset_path)
            logging.debug(f'Dataset for subject {subject_name} has been made ({len(dataset)} samples)')

            if subject_num in generalized_subjects:
                testing_subjects.append(subject_name)
                logging.debug(f'Subject {subject_name} is added to testing group')
            else:
                training_subjects.append(subject_name)
                logging.debug(f'Subject {subject_name} is added to training group')

            all_data.append(dataset)
        else:
            logging.debug(f'Dataset for subject {subject_name} is already exists')
            subject_num = int(re.findall(r'\d+', subject_name)[0])

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

    current_training = training_subjects
    current_testing = testing_subjects
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

    model, interpretation, parametrizer = get_model_by_name(model_name, X, Y)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.0001)
    loss = torch.nn.BCEWithLogitsLoss()
    metric = accuracy
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
                metric_names=['accuracy_train', 'accuracy_val'],
                loss_names=['loss_train', 'loss_val'],
            ),
            TempConvAveClipping(),
            # EarlyStopping(monitor='loss_val', patience=15, restore_best_weights=True),
            PenalizedEarlyStopping(monitor='loss_val', measure='accuracy_val', patience=15, restore_best_weights=True),
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

        iterator.select_subject(subject_name)

        x_test, y_true_test = next(iter(DataLoader(subject_data, len(subject_data))))
        y_pred_test = torch.squeeze(model(x_test)).detach().numpy()
        save(
            Predictions(
                y_pred_test,
                y_true_test
            ),
            iterator.predictions_path
        )
        if not no_params:
            logging.debug(f'Computing parameters for {subject_name}')
            interpreter = interpretation(model, subject_data, info)

            if not not_save_params:
                params = parametrizer(interpreter)
                params.save(iterator.parameters_path)

            for i in range(n_latent):
                fig = interpreter.plot_branch(i)
                fig.savefig(os.path.join(iterator.pics_path, f'Branch_{i}.png'), dpi=300)

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
            'train_acc',
            'test_acc',
            'runtime',
            *[f'{sbj}_acc' for sbj in current_testing]
        ],
        name=f'{current_training[0]} - {current_training[-1]} / {current_testing[0]} - {current_testing[-1]}'
    ).to_frame().T

    if os.path.exists(perf_table_path):
        pd.concat([pd.read_csv(perf_table_path, index_col=0, header=0), processed_df], axis=0)\
            .to_csv(perf_table_path)
    else:
        processed_df.to_csv(perf_table_path)
logging.info('All subjects are processed')
