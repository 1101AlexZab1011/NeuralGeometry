#! ./venv/bin/python

from deepmeg.viz import plot_spatial_weights
from deepmeg.params import WaveForms, SpatialParameters, TemporalParameters
from deepmeg import read_pkl, info_pick_channels
import matplotlib as mpl
import argparse
import os
from utils.storage import DLStorageIterator
import mne
import matplotlib.pyplot as plt


if __name__ == '__main__':
    mpl.use('agg')
    parser = argparse.ArgumentParser(
        description='The script for visualising spatial and spectral patterns learned by a LFCNN-like network'
    )
    parser.add_argument('-sd', '--subjects-dir', type=str,
        default=os.path.join(os.getcwd(), 'DATA'),
        help='Path to the subjects directory')
    parser.add_argument('-s', '--subject-name', type=str,
        default=None,
        help='ID if the subject to visualize')
    parser.add_argument('--name', type=str, default='Default_name',
                        help='Name of a task')
    parser.add_argument('--postfix', type=str,
                        default='', help='String to append to a task name')
    parser.add_argument('--prefix', type=str,
                        default='', help='String to set in the start of a task name')
    parser.add_argument('--sort', type=str,
                        default='branch_loss', help='A way to sort components. Can be "sumabs", "sum", "abssum" or "branch_loss"')
    parser.add_argument('--filt_induced', action='store_true',
                        help='Show induced after filtering or before (default is before)')

    subjects_dir,\
        subject_name,\
        classification_name,\
        classification_postfix,\
        classification_prefix,\
        sort_,\
        filt_induced = vars(parser.parse_args()).values()

    classification_name_formatted = "_".join(list(filter(
        lambda s: s not in (None, ""),
        [
            classification_prefix,
            classification_name,
            classification_postfix
        ]
    )))

    iterator = DLStorageIterator(subjects_dir, classification_name_formatted)
    iterator.select_subject(subject_name)

    sp = read_pkl(os.path.join(iterator.parameters_path, 'spatial.pkl'))
    tp = read_pkl(os.path.join(iterator.parameters_path, 'temporal.pkl'))
    wf = read_pkl(os.path.join(iterator.parameters_path, 'waveforms.pkl'))
    if sort_ == 'branch_loss':
        order = read_pkl(os.path.join(iterator.parameters_path, 'branch_loss.pkl'))
        order = order - order.min()
    elif sort_ in ['sum', 'sumabs', 'abssum']:
        order = sort_
    else:
        raise NotImplementedError(f'This kind of sorting is not implemented: "{sort_}"')

    epochs =  mne.read_epochs(iterator.data_paths[-1].epochs_path).pick_types(meg='grad')

    plot_spatial_weights(
        sp,
        tp,
        wf,
        epochs.info,
        summarize=order,
        logscale=False,
        temp_params=['input', 'output', 'response', 'pattern'],
        shift_induced_times = -epochs.times[0],
        filtered_induced=filt_induced
    )
    plt.show()
