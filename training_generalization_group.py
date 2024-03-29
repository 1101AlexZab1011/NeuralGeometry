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
from torch.utils.data import DataLoader
import torchmetrics
from utils import PenalizedEarlyStopping
from deepmeg.utils.convtools import conviter
from deepmeg.utils import check_path
from deepmeg.utils.viz import plot_metrics
from torch.utils.data import DataLoader, ConcatDataset
from training_pipeline import moving_average
from deepmeg.utils import check_path
from deepmeg.utils.viz import plot_metrics


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
    parser.add_argument('-k', '--kind', type=str, help='Data to generalize: spatial (sp) or conceptual (con)', default='sp')
    parser.add_argument('-cf', '--crop-from', type=float, help='Crop epoch from time', default=0.)
    parser.add_argument('-ct', '--crop-to', type=float, help='Crop epoch to time', default=None)
    parser.add_argument('-bf', '--bl-from', type=float, help='Baseline epoch from time', default=None)
    parser.add_argument('-bt', '--bl-to', type=float, help='Baseline epoch to time', default=0.)
    parser.add_argument('-m', '--model', type=str, help='Model to use', default='lfcnn')
    parser.add_argument('-d', '--device', type=str, help='Device to use', default='cuda')


    excluded_subjects, \
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
    all_train_data, all_test_data = list(), list()
    info = None
    for subject_name in iterator:
        if not os.path.exists(iterator.dataset_path[:-3] + '_train.pt') or info is None:
            logging.debug(f'Processing subject: {subject_name}')
            subject_num = int(re.findall(r'\d+', subject_name)[0])

            if (subject_num in excluded_subjects) or\
                (from_ and subject_num < from_) or\
                (to and subject_num > to):
                logging.debug(f'Skipping subject {subject_name}')
                continue

            # sp_preprocessor = BasicPreprocessor(103, 200)
            # con_preprocessor = BasicPreprocessor(103, 200, 2)
            sp_preprocessor = BasicPreprocessor(105, 200)
            con_preprocessor = BasicPreprocessor(105, 200, 2)
            preprcessed_training, preprcessed_testing = list(), list()
            if 'sp' in kind and 'con' in kind:
                raise ValueError('Training data can not be both conteptual and spatial')
            if 'sp' in kind:
                preprcessed_training.append(sp_preprocessor(iterator.get_data(STAGE.TRAINING)))
                preprcessed_testing.append(con_preprocessor(iterator.get_data(STAGE.TRAINING)))
            if 'con' in kind:
                preprcessed_training.append(con_preprocessor(iterator.get_data(STAGE.TRAINING)))
                preprcessed_testing.append(sp_preprocessor(iterator.get_data(STAGE.TRAINING)))
            if not preprcessed_training:
                raise ValueError(f'No data selected. Your config is: {kind = }')

            info = preprcessed_training[0].epochs.pick_types(meg='grad').info if info is None else info
            X_train = np.concatenate([
                data.
                epochs.
                pick_types(meg='grad').
                apply_baseline((bl_from, bl_to)).
                crop(crop_from, crop_to).
                get_data()
                for data in preprcessed_training
            ])
            X_test = np.concatenate([
                data.
                epochs.
                pick_types(meg='grad').
                apply_baseline((bl_from, bl_to)).
                crop(crop_from, crop_to).
                get_data()
                for data in preprcessed_testing
            ])
            feedbacks = [data.session_info.Feedback.to_numpy() for data in preprcessed_training]
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
                for data in preprcessed_training
            ]
            Y_train = np.concatenate(acc)

            feedbacks = [data.session_info.Feedback.to_numpy() for data in preprcessed_testing]
            feedbacks_np = moving_average(np.concatenate(feedbacks), win)
            min_, max_ = feedbacks_np.min(), feedbacks_np.max()
            del feedbacks_np
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
                for data in preprcessed_testing
            ]
            Y_test = np.concatenate(acc)

            dataset_train = EpochsDataset((X_train, Y_train), transform=zscore, savepath=iterator.dataset_content_path + '_train')
            dataset_train.save(iterator.dataset_path[:-3] + '_train.pt')
            dataset_test = EpochsDataset((X_test, Y_test), transform=zscore, savepath=iterator.dataset_content_path + '_test')
            dataset_test.save(iterator.dataset_path[:-3] + '_test.pt')
            all_train_data.append(dataset_train)
            all_test_data.append(dataset_test)
        else:
            logging.debug(f'Reading dataset of subject: {subject_name}')
            all_train_data.append(EpochsDataset.load(iterator.dataset_path[:-3] + '_train.pt'))
            all_test_data.append(EpochsDataset.load(iterator.dataset_path[:-3] + '_test.pt'))

    iterator.select_subject('group')
    img_path = os.path.join(iterator.subject_results_path, 'Pictures')
    check_path(img_path)
    train_dataset = ConcatDataset(all_train_data)
    test_dataset = ConcatDataset(all_test_data)
    logging.info(f'{len(train_dataset)} samples in generalized dataset and {len(test_dataset)} samples in total to generalize')
    train, test = torch.utils.data.random_split(train_dataset, [.7, .3])
    _, gtest = torch.utils.data.random_split(test_dataset, [.7, .3])
    X, Y = next(iter(DataLoader(train, 2)))

    match model_name:
        case 'lfcnn':
            model = LFCNN(
                n_channels=X.shape[1],
                n_latent=n_latent,
                n_times=X.shape[-1],
                filter_size=50,
                pool_factor=10,
                n_outputs=Y.shape[1]
            )
            interpretation = LFCNNInterpreter
            parametrizer = LFCNNParameters
        case 'lfcnnw':
                model = LFCNNW(
                    n_channels=X.shape[1],
                    n_latent=n_latent,
                    n_times=X.shape[-1],
                    filter_size=50,
                    pool_factor=10,
                    n_outputs=Y.shape[1]
                )
                interpretation = LFCNNWInterpreter
                parametrizer = SPIRITParameters
        case 'hilbert':
            model = HilbertNet(
                n_channels=X.shape[1],
                n_latent=n_latent,
                n_times=X.shape[-1],
                filter_size=50,
                pool_factor=10,
                n_outputs=Y.shape[1]
            )
            interpretation = LFCNNInterpreter
            parametrizer = LFCNNParameters
        case 'spirit':
            model = SPIRIT(
                n_channels=X.shape[1],
                n_latent=n_latent,
                n_times=X.shape[-1],
                window_size=20,
                latent_dim=10,
                filter_size=50,
                pool_factor=10,
                n_outputs=Y.shape[1]
            )
            interpretation = SPIRITInterpreter
            parametrizer = SPIRITParameters
        case 'fourier':
            model = FourierSPIRIT(
                n_channels=X.shape[1],
                n_latent=n_latent,
                n_times=X.shape[-1],
                window_size=20,
                latent_dim=10,
                filter_size=50,
                pool_factor=10,
                n_outputs=Y.shape[1]
            )
            interpretation = SPIRITInterpreter
            parametrizer = SPIRITParameters
        case 'canonical':
            model = CanonicalSPIRIT(
                n_channels=X.shape[1],
                n_latent=n_latent,
                n_times=X.shape[-1],
                window_size=20,
                latent_dim=10,
                filter_size=50,
                pool_factor=10,
                n_outputs=Y.shape[1]
            )
            interpretation = SPIRITInterpreter
            parametrizer = SPIRITParameters
        case _:
            raise ValueError(f'Invalid model name: {model_name}')

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
        fig.savefig(os.path.join(img_path, f'{name}.png'), dpi=300)

    x_train, y_true_train = next(iter(DataLoader(train, len(train))))
    y_pred_train = torch.squeeze(model(x_train)).detach().numpy()
    x_test, y_true_test = next(iter(DataLoader(test, len(test))))
    y_pred_test = torch.squeeze(model(x_test)).detach().numpy()
    x_gtest, y_true_gtest = next(iter(DataLoader(gtest, len(gtest))))
    y_pred_gtest = torch.squeeze(model(x_gtest)).detach().numpy()

    save(
        Predictions(
            y_pred_test,
            y_true_test
        ),
        iterator.predictions_path
    )
    save(
        Predictions(
            y_pred_gtest,
            y_true_gtest
        ),
        iterator.predictions_path[:-4] + '_generalized.pkl'
    )

    train_result = model.evaluate(train)
    result = model.evaluate(test)
    gresult = model.evaluate(gtest)


    train_loss_, train_acc_ = train_result.values()
    test_loss_, test_acc_ = result.values()
    gtest_loss_, gtest_acc_ = gresult.values()
    logging.info(f'{subject_name}\nClassification results:\n \tRUNTIME: {runtime}\n\tTRAIN_ACC: {train_acc_}\n\tTEST_ACC: {test_acc_}\n\tGENERALIZATION_ACC: {gtest_acc_}')

    if not no_params:
        logging.debug('Computing parameters')
        interpreter = interpretation(model, test, info)
        params = parametrizer(interpreter)
        params.save(iterator.parameters_path)

        for i in range(n_latent):
            fig = interpreter.plot_branch(i)
            fig.savefig(os.path.join(img_path, f'Branch_{i}.png'), dpi=300)

        interpreter = interpretation(model, gtest, info)
        params = parametrizer(interpreter)
        params.save(iterator.parameters_path[:-4] + '_generalized.pkl')

        for i in range(n_latent):
            fig = interpreter.plot_branch(i)
            fig.savefig(os.path.join(img_path, f'Branch_{i}_g.png'), dpi=300)

    perf_table_path = os.path.join(
        iterator.history_path,
        f'{classification_name_formatted}.csv'
    )
    processed_df = pd.Series(
        [
            len(train),
            len(test),
            train_acc_,
            train_loss_,
            test_acc_,
            test_loss_,
            gtest_acc_,
            gtest_loss_,
            runtime
        ],
        index=[
            'train_set',
            'test_set',
            'train_mae',
            'train_loss',
            'test_mae',
            'test_loss',
            'generalization_mae',
            'generalization_loss',
            'runtime'
        ],
        name='group'
    ).to_frame().T

    if os.path.exists(perf_table_path):
        pd.concat([pd.read_csv(perf_table_path, index_col=0, header=0), processed_df], axis=0)\
            .to_csv(perf_table_path)
    else:
        processed_df.to_csv(perf_table_path)

logging.info('All subjects are processed')
