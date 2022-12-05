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
from deepmeg.params import save_parameters, compute_temporal_parameters, compute_waveforms, \
    Predictions, WaveForms, TemporalParameters, SpatialParameters, ComponentsOrder, get_order
import re
import logging


if __name__ == '__main__':
    mpl.use('agg')
    logging.basicConfig(
        format='%(asctime)s %(levelname)s:\t%(message)s',
        filename='./logs/main.log',
        encoding='utf-8',
        level=logging.DEBUG
    )
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

    excluded_subjects, \
        from_, \
        to, \
        subjects_dir, \
        classification_name,\
        classification_postfix,\
        classification_prefix, \
        project_name, \
        no_params = vars(parser.parse_args()).values()

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
    logging.info(f'Current classification: {classification_name}')

    iterator = DLStorageIterator(subjects_dir, name=classification_name_formatted)
    for subject_name in iterator:
        logging.debug(f'Processing subject: {subject_name}')
        subject_num = int(re.findall(r'\d+', subject_name)[0])

        if (subject_num in excluded_subjects) or\
            (from_ and subject_num < from_) or\
            (to and subject_num > to):
            logging.debug(f'Skipping subject {subject_name}')
            continue

        preprocessor = BasicPreprocessor(103, 200)
        data_pre: Preprocessed = preprocessor(iterator.get_data(STAGE.PRETEST))
        data_post: Preprocessed  = preprocessor(iterator.get_data(STAGE.POSTTEST))
        X = np.concatenate([
            data_pre.epochs.pick_types(meg='grad').get_data(),
            data_post.epochs.pick_types(meg='grad').get_data()
        ])
        Y = np.concatenate([data_pre.clusters, data_post.clusters])
        n_classes, classes_samples = np.unique(Y, return_counts=True)
        n_classes = len(n_classes)
        classes_samples = classes_samples.tolist()
        import_opt['savepath'] = iterator.network_out_path + '/'
        meta = mf.produce_tfrecords((X, Y), **import_opt)
        dataset = mf.Dataset(meta, train_batch=100)
        lf_params = dict(
            n_latent=16, #number of latent factors
            filter_length=50, #convolutional filter length in time samples
            nonlin = tf.nn.elu,
            padding = 'SAME',
            pooling = 5,#pooling factor
            stride = 5, #stride parameter for pooling layer
            pool_type='max',
            model_path = import_opt['savepath'],
            dropout = .5,
            l1_scope = ["weights"],
            l1=3e-1
        )

        model = SimpleNet(dataset, lf_params)
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
            times = (1 / float(model.dataset.h_params['fs'])) *\
                np.arange(model.dataset.h_params['n_t'])
            patterns = model.patterns.copy()
            model.compute_patterns(meta['test_paths'], output='filters')
            filters = model.patterns.copy()
            franges, finputs, foutputs, fresponces, fpatterns = compute_temporal_parameters(model)
            induced, times, time_courses = compute_waveforms(model)

            save_parameters(
                model.branch_relevance_loss,
                os.path.join(iterator.parameters_path, 'branch_loss.pkl'),
                'Branches relevance'
            )

            save_parameters(
                WaveForms(time_courses.mean(1), induced, times, time_courses),
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
            save_parameters(
                ComponentsOrder(
                    get_order(*model._sorting('l2')),
                    get_order(*model._sorting('compwise_loss')),
                    get_order(*model._sorting('weight')),
                    get_order(*model._sorting('output_corr')),
                    get_order(*model._sorting('weight_corr')),
                ),
                os.path.join(iterator.parameters_path, 'sorting.pkl'),
                'sorting'
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
