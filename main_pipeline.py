#! ./venv/bin/python

import matplotlib as mpl
import argparse
import os
from utils.storage import DLStorageIterator, STAGE
from utils.preprocessing import BasicPreprocessor, Preprocessed
import numpy as np
import mneflow as mf
from utils.models import SimpleNet, SimpleNetA
import tensorflow as tf
import pandas as pd
from time import perf_counter
from deepmeg.params import save_parameters, compute_temporal_parameters, compute_waveforms, \
    Predictions, WaveForms, TemporalParameters, SpatialParameters, ComponentsOrder, get_order, \
    compute_compression_parameters, CompressionParameters
import re
import logging
from utils import balance


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
    parser.add_argument('--name', type=str, default='Default_name',
                        help='Name of a task')
    parser.add_argument('--postfix', type=str,
                        default='', help='String to append to a task name')
    parser.add_argument('--prefix', type=str,
                        default='', help='String to set in the start of a task name')
    parser.add_argument('--project-name', type=str,
                        default='mem_arch_epochs', help='Name of a project')
    parser.add_argument('--no-params', action='store_true', help='Do not compute parameters')
    parser.add_argument('--balance', action='store_true', help='Balance classes')
    parser.add_argument('-t', '--target', type=str, help='Target to predict (must be a column from sesinfo csv file)')
    parser.add_argument('-k', '--kind', type=str, help='Spatial (sp) or conceptual (con) or both "spccon"', default='spcon')
    parser.add_argument('-st', '--stage', type=str, help='PreTest (pre) or PostTest (post) or both "prepost"', default='prepost')
    parser.add_argument('-cf', '--crop-from', type=float, help='Crop epoch from time', default=None)
    parser.add_argument('-ct', '--crop-to', type=float, help='Crop epoch to time', default=None)
    parser.add_argument('-m', '--model', type=str, help='Model to use', default='simplenet')


    excluded_subjects, \
        from_, \
        to, \
        subjects_dir, \
        classification_name,\
        classification_postfix,\
        classification_prefix, \
        project_name, \
        no_params, \
        balance_classes, \
        target_col_name,\
        kind,\
        stage,\
        crop_from, crop_to,\
        model_name = vars(parser.parse_args()).values()

    import_opt = dict(
        savepath=None,  # path where TFR files will be saved
        out_name=project_name,  # name of TFRecords files
        fs=200,
        input_type='trials',
        target_type='int',
        picks={'meg':'grad'},
        scale=True,  # apply baseline_scaling
        crop_baseline=True,  # remove baseline interval after scaling
        decimate=None,
        scale_interval=(0, 40),  # indices in time axis corresponding to baseline interval
        n_folds=5,  # validation set size set to 20% of all data
        overwrite=True,
        segment=False,
        test_set='holdout'
    )
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
        filename=f'./logs/{classification_name}.log',
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

        sp_preprocessor = BasicPreprocessor(103, 200)
        con_preprocessor = BasicPreprocessor(103, 200, 2)
        preprcessed = list()
        if 'sp' in kind:
            if 'pre' in stage:
                preprcessed.append(sp_preprocessor(iterator.get_data(STAGE.PRETEST)))
            if 'post' in stage:
                preprcessed.append(sp_preprocessor(iterator.get_data(STAGE.POSTTEST)))
        if 'con' in kind:
            if 'pre' in stage:
                preprcessed.append(con_preprocessor(iterator.get_data(STAGE.PRETEST)))
            if 'post' in stage:
                preprcessed.append(con_preprocessor(iterator.get_data(STAGE.POSTTEST)))
        if not preprcessed:
            raise ValueError(f'No data selected. Your config is: {kind = }, {stage = }')

        X = np.concatenate([data.epochs.pick_types(meg='grad').crop(crop_from, crop_to).get_data() for data in preprcessed])
        Y = np.concatenate([data.session_info[target_col_name].to_numpy() for data in preprcessed])

        if balance_classes:
            X, Y = balance(X, Y)

        n_classes, classes_samples = np.unique(Y, return_counts=True)
        n_classes = len(n_classes)
        classes_samples = classes_samples.tolist()
        import_opt['savepath'] = iterator.network_out_path + '/'
        meta = mf.produce_tfrecords((X, Y), **import_opt)
        dataset = mf.Dataset(meta, train_batch=100)
        lf_params = dict(
            n_latent=8, #number of latent factors
            filter_length=50, #convolutional filter length in time samples
            nonlin = tf.nn.relu,
            padding = 'SAME',
            pooling = 5,#pooling factor
            stride = 5, #stride parameter for pooling layer
            pool_type='max',
            model_path = import_opt['savepath'],
            dropout = .4,
            l1_scope = ["weights"],
            l1=3e-1
        )

        match model_name:
            case 'simplenet':
                model = SimpleNet(dataset, lf_params)
            case 'lfcnn':
                model = mf.models.LFCNN(dataset, lf_params)
            case 'simplenetA':
                model = SimpleNetA(dataset, lf_params)

        model.build()
        t1 = perf_counter()
        model.train(n_epochs=25, eval_step=100, early_stopping=3)
        runtime = perf_counter() - t1
        y_true_train, y_pred_train = model.predict(meta['train_paths'])
        y_true_test, y_pred_test = model.predict(meta['test_paths'])
        save_parameters(
            Predictions(
                y_pred_test,
                y_true_test
            ),
            os.path.join(iterator.predictions_path, 'y_pred.pkl'),
            'predictions'
        )
        train_loss_, train_acc_ = model.evaluate(meta['train_paths'])
        test_loss_, test_acc_ = model.evaluate(meta['test_paths'])
        logging.info(f'{subject_name}\nClassification results:\n \tRUNTIME: {runtime}\n\tTRAIN_ACC: {train_acc_}\n\tTEST_ACC: {test_acc_}')
        if not no_params:
            logging.debug('Computing parameters')
            model.compute_patterns(meta['test_paths'])
            nt = model.dataset.h_params['n_t']
            time_courses = np.squeeze(model.lat_tcs.reshape([model.specs['n_latent'], -1, nt]))
            time_courses_filt = np.squeeze(model.lat_tcs_filt.reshape([model.specs['n_latent'], -1, nt]))
            times = (1 / float(model.dataset.h_params['fs'])) *\
                np.arange(model.dataset.h_params['n_t'])
            patterns = model.patterns.copy()
            model.compute_patterns(meta['test_paths'], output='filters', relevances=False)
            filters = model.patterns.copy()
            franges, finputs, foutputs, fresponces, fpatterns = compute_temporal_parameters(model)
            induced, induced_filt, times, time_courses = compute_waveforms(model)

            save_parameters(
                model.branch_relevance_loss,
                os.path.join(iterator.parameters_path, 'branch_loss.pkl'),
                'Branches relevance'
            )

            save_parameters(
                WaveForms(time_courses.mean(1), time_courses_filt.mean(1), induced, induced_filt, times, time_courses),
                os.path.join(iterator.parameters_path, 'waveforms.pkl'),
                'WaveForms'
            )

            save_parameters(
                SpatialParameters(patterns, filters),
                os.path.join(iterator.parameters_path, 'spatial.pkl'),
                'spatial'
            )
            save_parameters(
                TemporalParameters(franges, finputs, foutputs, fresponces, fpatterns),
                os.path.join(iterator.parameters_path, 'temporal.pkl'),
                'temporal'
            )
            if model_name == 'simplenetA':
                temp_relevance_loss, eigencentrality_, time_courses_env, compression_weights = compute_compression_parameters(model)
                save_parameters(
                    CompressionParameters(
                        temp_relevance_loss,
                        eigencentrality_,
                        time_courses_env,
                        compression_weights
                    ),
                    os.path.join(iterator.parameters_path, 'compression.pkl'),
                    'compression'
                )

        perf_table_path = os.path.join(
            iterator.history_path,
            f'{classification_name_formatted}.csv'
        )
        processed_df = pd.Series(
            [
                n_classes,
                *classes_samples,
                sum(classes_samples),
                np.array(meta['test_fold'][0]).shape[0],
                train_acc_,
                train_loss_,
                test_acc_,
                test_loss_,
                model.v_metric,
                model.v_loss,
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
                'val_acc',
                'val_loss',
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
