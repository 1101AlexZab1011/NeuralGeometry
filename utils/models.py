from mneflow.layers import LFTConv, DeMixing, Dense
from tensorflow.keras.layers import Dropout
import tensorflow as tf
import matplotlib.pyplot as plt
import mneflow as mf
import numpy as np
import scipy.signal as sl
from mne import channels, evoked, create_info
import mne
import scipy as sp
from typing import Optional

def eigencentrality(matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the eigencentrality of a matrix.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to calculate the eigencentrality of.

    Returns
    -------
    np.ndarray
        The eigencentrality of the matrix.
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    eigencentrality = eigenvectors[:,0]

    return eigencentrality


class SimpleNet(mf.models.LFCNN):
    def __init__(self, Dataset, specs=None):
        if specs is None:
            specs=dict()
        super().__init__(Dataset, specs)

    def build_graph(self):
        self.dmx = DeMixing(size=self.specs['n_latent'], nonlin=tf.identity,
                            axis=3, specs=self.specs)
        self.dmx_out = self.dmx(self.inputs)

        self.tconv = LFTConv(
            size=self.specs['n_latent'],
            nonlin=tf.identity,
            filter_length=self.specs['filter_length'],
            padding=self.specs['padding'],
            specs=self.specs
        )
        self.tconv_out = self.tconv(self.dmx_out)

        self.pool = lambda X: X[:, :, ::self.specs['pooling'], :]

        self.pooled = self.pool(self.tconv_out)

        dropout = Dropout(
            self.specs['dropout'],
            noise_shape=None
        )(self.pooled)

        self.fin_fc = Dense(size=self.out_dim, nonlin=tf.identity,
                            specs=self.specs)

        y_pred = self.fin_fc(dropout)

        return y_pred

    def compute_patterns(self, data_path=None, *, output='patterns'):

        if not data_path:
            print("Computing patterns: No path specified, using validation dataset (Default)")
            ds = self.dataset.val
        elif isinstance(data_path, str) or isinstance(data_path, (list, tuple)):
            ds = self.dataset._build_dataset(
                data_path,
                split=False,
                test_batch=None,
                repeat=True
            )
        elif isinstance(data_path, mf.data.Dataset):
            if hasattr(data_path, 'test'):
                ds = data_path.test
            else:
                ds = data_path.val
        elif isinstance(data_path, tf.data.Dataset):
            ds = data_path
        else:
            raise AttributeError('Specify dataset or data path.')

        X, y = [row for row in ds.take(1)][0]

        self.out_w_flat = self.fin_fc.w.numpy()
        self.out_weights = np.reshape(
            self.out_w_flat,
            [-1, self.dmx.size, self.out_dim]
        )
        self.out_biases = self.fin_fc.b.numpy()
        self.feature_relevances = self.componentwise_loss(X, y)
        self.branchwise_loss(X, y)

        # compute temporal convolution layer outputs for vis_dics
        tc_out = self.pool(self.tconv(self.dmx(X)).numpy())

        # compute data covariance
        X = X - tf.reduce_mean(X, axis=-2, keepdims=True)
        X = tf.transpose(X, [3, 0, 1, 2])
        X = tf.reshape(X, [X.shape[0], -1])
        self.dcov = tf.matmul(X, tf.transpose(X))

        # get spatial extraction fiter weights
        demx = self.dmx.w.numpy()

        kern = np.squeeze(self.tconv.filters.numpy()).T

        X = X.numpy().T
        if 'patterns' in output:
            if 'old' in output:
                self.patterns = np.dot(self.dcov, demx)
            else:
                patterns = []
                X_filt = np.zeros_like(X)
                for i_comp in range(kern.shape[0]):
                    for i_ch in range(X.shape[1]):
                        x = X[:, i_ch]
                        X_filt[:, i_ch] = np.convolve(x, kern[i_comp, :], mode="same")
                    patterns.append(np.cov(X_filt.T) @ demx[:, i_comp])
                self.patterns = np.array(patterns).T
                # self.lat_tcs_filt = np.dot(demx.T, X_filt.T)
        else:
            self.patterns = demx

        # self.lat_tcs = np.dot(demx.T, X.T)
        lat_tcs = self.dmx(X)
        lat_tcs_filt = np.squeeze(self.tconv(lat_tcs).numpy())
        self.lat_tcs = np.transpose(np.squeeze(lat_tcs.numpy()), (0, 2, 1))
        self.lat_tcs_filt = np.transpose(lat_tcs_filt, (0, 2, 1))

        del X

        #  Temporal conv stuff
        self.filters = kern.T
        self.tc_out = np.squeeze(tc_out)
        self.corr_to_output = self.get_output_correlations(y)

    def plot_patterns(
        self, sensor_layout=None, sorting='l2', percentile=90,
        scale=False, class_names=None, info=None
    ):
        order, ts = self._sorting(sorting)
        self.uorder = order.ravel()
        l_u = len(self.uorder)
        if info:
            info.__setstate__(dict(_unlocked=True))
            info['sfreq'] = 1.
            self.fake_evoked = evoked.EvokedArray(self.patterns, info, tmin=0)
            if l_u > 1:
                self.fake_evoked.data[:, :l_u] = self.fake_evoked.data[:, self.uorder]
            elif l_u == 1:
                self.fake_evoked.data[:, l_u] = self.fake_evoked.data[:, self.uorder[0]]
            self.fake_evoked.crop(tmax=float(l_u))
            if scale:
                _std = self.fake_evoked.data[:, :l_u].std(0)
                self.fake_evoked.data[:, :l_u] /= _std
        elif sensor_layout:
            lo = channels.read_layout(sensor_layout)
            info = create_info(lo.names, 1., sensor_layout.split('-')[-1])
            orig_xy = np.mean(lo.pos[:, :2], 0)
            for i, ch in enumerate(lo.names):
                if info['chs'][i]['ch_name'] == ch:
                    info['chs'][i]['loc'][:2] = (lo.pos[i, :2] - orig_xy)/3.
                    #info['chs'][i]['loc'][4:] = 0
                else:
                    print("Channel name mismatch. info: {} vs lo: {}".format(
                        info['chs'][i]['ch_name'], ch))

            self.fake_evoked = evoked.EvokedArray(self.patterns, info)

            if l_u > 1:
                self.fake_evoked.data[:, :l_u] = self.fake_evoked.data[:, self.uorder]
            elif l_u == 1:
                self.fake_evoked.data[:, l_u] = self.fake_evoked.data[:, self.uorder[0]]
            self.fake_evoked.crop(tmax=float(l_u))
            if scale:
                _std = self.fake_evoked.data[:, :l_u].std(0)
                self.fake_evoked.data[:, :l_u] /= _std
        else:
            raise ValueError("Specify sensor layout")


        if np.any(self.uorder):
            nfilt = max(self.out_dim, 8)
            nrows = max(1, l_u//nfilt)
            ncols = min(nfilt, l_u)
            if class_names:
                comp_names = class_names
            else:
                comp_names = ["Class #{}".format(jj+1) for jj in range(ncols)]
            f, ax = plt.subplots(nrows, ncols, sharey=True)
            plt.tight_layout()
            f.set_size_inches([16, 3])
            ax = np.atleast_2d(ax)

            for ii in range(nrows):
                fake_times = np.arange(ii * ncols,  (ii + 1) * ncols, 1.)
                vmax = np.percentile(self.fake_evoked.data[:, :l_u], 95)
                self.fake_evoked.plot_topomap(
                    times=fake_times,
                    axes=ax[ii],
                    colorbar=False,
                    vmax=vmax,
                    scalings=1,
                    time_format="Class #%g",
                    title='Patterns ('+str(sorting)+')',
                    outlines='head',
                )

    def branchwise_loss(self, X, y):
        self_weights_original = self.km.get_weights().copy()
        base_loss, _ = self.km.evaluate(X, y, verbose=0)

        losses = []
        for i in range(self.specs["n_latent"]):
            self_weights = self_weights_original.copy()
            spatial_weights = self_weights[0].copy()
            # spatial_biases = self_weights[1].copy()
            # temporal_biases = self_weights[3].copy()
            spatial_weights[:, i] = 0
            # spatial_biases[i] = 0
            # temporal_biases[i] = 0
            self_weights[0] = spatial_weights
            # self_weights[1] = spatial_biases
            # self_weights[3] = temporal_biases
            self.km.set_weights(self_weights)
            losses.append(self.km.evaluate(X, y, verbose=0)[0])
        self.km.set_weights(self_weights_original)
        self.branch_relevance_loss = base_loss - np.array(losses)

    def plot_branch(
        self,
        branch_num: int,
        info: mne.Info,
        params: Optional[list[str]] = ['input', 'output', 'response']
    ):
        info.__setstate__(dict(_unlocked=True))
        info['sfreq'] = 1.
        data = self.patterns[:, np.argsort(self.branch_relevance_loss)]
        relevances = self.branch_relevance_loss - self.branch_relevance_loss.min()
        relevance = sorted([np.round(rel/relevances.sum(), 2) for rel in relevances], reverse=True)[branch_num]
        self.fake_evoked = evoked.EvokedArray(data, info, tmin=0)
        fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)
        fig.tight_layout()

        self.fs = self.dataset.h_params['fs']

        out_filter = self.filters[:, branch_num]
        _, psd = sl.welch(self.lat_tcs[branch_num], fs=self.fs, nperseg=self.fs * 2)
        w, h = (lambda w, h: (w, h))(*sl.freqz(out_filter, 1, worN=self.fs))
        frange = w / np.pi * self.fs / 2

        for param in params:
            if param == 'input':
                finput = psd[:-1]
                ax2.plot(frange, (finput - finput.mean())/finput.std() + finput.mean()/finput.std(), color='tab:blue')
            elif param == 'output':
                foutput = np.real(finput * h * np.conj(h))
                ax2.plot(frange, (foutput - foutput.mean())/foutput.std() + foutput.mean()/foutput.std(), color='tab:orange')
            elif param == 'response':
                fresponce = np.abs(h)
                ax2.plot(frange, (fresponce - fresponce.mean())/fresponce.std() + fresponce.mean()/fresponce.std(), color='tab:green')
            elif param == 'pattern':
                fpattern = finput * np.abs(h)
                ax2.plot((fpattern - fpattern.mean())/fpattern.std() + fpattern.mean()/fpattern.std(), color='tab:pink')

        ax2.legend([param.capitalize() for param in params])
        ax2.set_xlim(0, 100)

        fig.suptitle(f'Branch {branch_num}', y=0.95, x=0.2, fontsize=30)
        fig.set_size_inches(10, 5)
        self.fake_evoked.plot_topomap(
            times=branch_num,
            axes=ax1,
            colorbar=False,
            scalings=1,
            time_format="",
            outlines='head',
        )

        return fig


class SimpleNetA(SimpleNet):
    def __init__(self, Dataset, specs=None):
        if specs is None:
            specs=dict()
        super().__init__(Dataset, specs)

    def build_graph(self):
        self.dmx = DeMixing(size=self.specs['n_latent'], nonlin=tf.identity,
                            axis=3, specs=self.specs)
        self.dmx_out = self.dmx(self.inputs)

        self.tconv = LFTConv(
            size=self.specs['n_latent'],
            nonlin=tf.identity,
            filter_length=self.specs['filter_length'],
            padding=self.specs['padding'],
            specs=self.specs
        )
        self.tconv_out = self.tconv(self.dmx_out)

        n_times = self.tconv_out.shape[-2]
        pooled_dim = n_times // self.specs['pooling']
        self.pool_list = [
            tf.keras.layers.Dense(
                pooled_dim,
                use_bias=False,
                activation='sigmoid'
            )
            for _ in range(self.specs['n_latent'])
        ]
        # pooled = list()
        # for i, pooling in enumerate(self.pool_list):
        #     pooled.append(pooling(self.tconv_out[:, :, :, i]))

        # self.pooled = tf.stack(pooled, -1)
        self.pool = lambda x: tf.stack(
            [
                pooling(x[:, :, :, i])
                for i, pooling in enumerate(self.pool_list)
            ],
            -1
        )
        self.pooled = self.pool(self.tconv_out)

        dropout = Dropout(
            self.specs['dropout'],
            noise_shape=None
        )(self.pooled)

        self.fin_fc = Dense(size=self.out_dim, nonlin=tf.identity,
                            specs=self.specs)

        y_pred = self.fin_fc(dropout)

        return y_pred

    # def plot_temporal
    def branchwise_loss(self, X, y):
        self_weights_original = self.km.get_weights().copy()
        base_loss, _ = self.km.evaluate(X, y, verbose=0)

        losses = []
        for i in range(self.specs["n_latent"]):
            self_weights = self_weights_original.copy()
            spatial_weights = self_weights[0].copy()
            # spatial_biases = self_weights[1].copy()
            # temporal_biases = self_weights[3].copy()
            spatial_weights[:, i] = 0
            # spatial_biases[i] = 0
            # temporal_biases[i] = 0
            self_weights[0] = spatial_weights
            # self_weights[1] = spatial_biases
            # self_weights[3] = temporal_biases
            self.km.set_weights(self_weights)
            losses.append(self.km.evaluate(X, y, verbose=0)[0])
        self.km.set_weights(self_weights_original)
        self.branch_relevance_loss = base_loss - np.array(losses)

    def tempwise_loss(self, X, y):
        self_weights_original = self.km.get_weights().copy()
        temp_sel_weights = self_weights_original[2:-2]
        base_loss, _ = self.km.evaluate(X, y, verbose=0)
        window_size = 1
        componentslosses = list()
        for i_latent, tem_sel_w in enumerate(temp_sel_weights):
            print(f'Processing branch {i_latent}...', end='')
            timelosses = list()
            for i_timepoint in range(0, len(tem_sel_w), window_size):
                tem_sel_w_copy = tem_sel_w.copy()
                tem_sel_w_copy[:i_timepoint, :] = 0
                tem_sel_w_copy[i_timepoint+min(window_size, len(tem_sel_w) - i_timepoint):, :] = 0
                temp_sel_weights_copy = temp_sel_weights.copy()
                temp_sel_weights_copy[i_latent] = tem_sel_w_copy
                for i in range(len(temp_sel_weights_copy)):
                    if i != i_latent:
                        temp_sel_weights_copy[i] = np.zeros_like(temp_sel_weights_copy[i])
                self_weights = self_weights_original.copy()
                self_weights[2:-2] = temp_sel_weights_copy
                self.km.set_weights(self_weights)
                loss = self.km.evaluate(X, y, verbose=0)[0]
                timelosses += [loss for _ in range(min(window_size, len(tem_sel_w) - i_timepoint))]
            componentslosses.append(timelosses)
            print(f'\tDONE')
        self.km.set_weights(self_weights_original)
        self.temp_relevance_loss = - np.array(componentslosses)

    def compute_patterns(self, data_path=None, *, output='patterns', relevances=True):
        if not data_path:
            print("Computing patterns: No path specified, using validation dataset (Default)")
            ds = self.dataset.val
        elif isinstance(data_path, str) or isinstance(data_path, (list, tuple)):
            ds = self.dataset._build_dataset(
                data_path,
                split=False,
                test_batch=None,
                repeat=True
            )
        elif isinstance(data_path, mf.data.Dataset):
            if hasattr(data_path, 'test'):
                ds = data_path.test
            else:
                ds = data_path.val
        elif isinstance(data_path, tf.data.Dataset):
            ds = data_path
        else:
            raise AttributeError('Specify dataset or data path.')

        X, y = [row for row in ds.take(1)][0]
        lat_tcs = self.dmx(X)

        self.out_w_flat = self.fin_fc.w.numpy()
        self.out_weights = np.reshape(
            self.out_w_flat,
            [-1, self.dmx.size, self.out_dim]
        )
        self.out_biases = self.fin_fc.b.numpy()
        if relevances:
            self.componentwise_loss(X, y)
            self.branchwise_loss(X, y)
            self.tempwise_loss(X, y)

        # compute temporal convolution layer outputs for vis_dics
        tc_out = self.pool(self.tconv(self.dmx(X)).numpy())

        # compute data covariance
        X = X - tf.reduce_mean(X, axis=-2, keepdims=True)
        X = tf.transpose(X, [3, 0, 1, 2])
        X = tf.reshape(X, [X.shape[0], -1])
        self.dcov = tf.matmul(X, tf.transpose(X))

        # get spatial extraction fiter weights
        demx = self.dmx.w.numpy()

        kern = np.squeeze(self.tconv.filters.numpy()).T

        X = X.numpy().T
        if 'patterns' in output:
            if 'old' in output:
                self.patterns = np.dot(self.dcov, demx)
            else:
                patterns = []
                X_filt = np.zeros_like(X)
                for i_comp in range(kern.shape[0]):
                    for i_ch in range(X.shape[1]):
                        x = X[:, i_ch]
                        X_filt[:, i_ch] = np.convolve(x, kern[i_comp, :], mode="same")
                    patterns.append(np.cov(X_filt.T) @ demx[:, i_comp])
                self.patterns = np.array(patterns).T
                # self.lat_tcs_filt = np.dot(demx.T, X_filt.T)
        else:
            self.patterns = demx

        # self.lat_tcs = np.dot(demx.T, X.T)
        lat_tcs_filt = np.squeeze(self.tconv(lat_tcs).numpy())
        self.lat_tcs = np.transpose(np.squeeze(lat_tcs.numpy()), (0, 2, 1))
        self.lat_tcs_filt = np.transpose(lat_tcs_filt, (0, 2, 1))

        del X

        #  Temporal conv stuff
        self.filters = kern.T

    def plot_branch(
        self,
        branch_num: int,
        info: mne.Info,
        params: Optional[list[str]] = ['input', 'output', 'response']
    ):
        info.__setstate__(dict(_unlocked=True))
        info['sfreq'] = 1.
        sorting = np.argsort(self.branch_relevance_loss)[::-1]
        data = self.patterns[:, sorting]
        filters = self.filters[:, sorting]
        relevances = self.branch_relevance_loss - self.branch_relevance_loss.min()
        relevance = sorted([np.round(rel/relevances.sum(), 2) for rel in relevances], reverse=True)[branch_num]
        self.fake_evoked = evoked.EvokedArray(data, info, tmin=0)
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1)
        fig.tight_layout()

        self.fs = self.dataset.h_params['fs']

        out_filter = filters[:, branch_num]
        _, psd = sl.welch(np.reshape(self.lat_tcs, (self.lat_tcs.shape[1], -1))[sorting[branch_num]], fs=self.fs, nperseg=self.fs * 2)
        w, h = (lambda w, h: (w, h))(*sl.freqz(out_filter, 1, worN=self.fs))
        frange = w / np.pi * self.fs / 2
        z = lambda x: (x - x.mean())/x.std()
        finput = psd[:-1]

        for param in params:
            if param == 'input':
                finput = z(finput)
                ax2.plot(frange, finput - finput.min(), color='tab:blue')
            elif param == 'output':
                foutput = np.real(finput * h * np.conj(h))
                foutput = z(foutput)
                ax2.plot(frange, foutput - foutput.min(), color='tab:orange')
            elif param == 'response':
                fresponce = np.abs(h)
                fresponce = z(fresponce)
                ax2.plot(frange, fresponce - fresponce.min(), color='tab:green')
            elif param == 'pattern':
                fpattern = finput * np.abs(h)
                fpattern = z(fpattern)
                ax2.plot(frange, fpattern - fpattern.min(), color='tab:pink')

        kernel_size = 20
        kernel = np.ones(kernel_size) / kernel_size
        ax2.legend([param.capitalize() for param in params])
        ax2.set_xlim(0, 100)
        time_courses_filtered = self.lat_tcs_filt
        # time_courses_env = np.zeros_like(time_courses_filtered)
        # kern = np.squeeze(self.envconv.filters.numpy()).T
        # conv = np.abs(np.convolve(time_courses_filtered[sorting[branch_num], :], kern[sorting[branch_num], :], mode="same"))
        # time_courses_env[sorting[branch_num], :] = conv
        selected_time_course_plot = time_courses_filtered#.reshape(
        #     [-1, self.specs['n_latent'], self.dataset.h_params['n_t']]
        # )

        selected_w =  self.pool_list[sorting[branch_num]].weights[0].numpy()
        selected_temp_relevance_loss = self.temp_relevance_loss[sorting[branch_num]]

        kernel_size = 20
        kernel = np.ones(kernel_size) / kernel_size
        data = np.cov(selected_w)

        #evoked
        ax3.plot(
            np.arange(0, self.dataset.h_params['n_t']/self.fs, 1./self.fs),
            sp.stats.zscore(selected_time_course_plot.mean(0)[sorting[branch_num]])
        )
        temp_course = self.temp_relevance_loss[sorting[branch_num]]
        temp_course_convolved = np.convolve(temp_course, kernel, mode='same')
        ax3.plot(
            np.arange(0, len(temp_course)/self.fs, 1./self.fs),
            np.concatenate([
                [np.nan for _ in range(kernel_size//2)],
                sp.stats.zscore(temp_course_convolved[kernel_size//2:-kernel_size//2]),
                [np.nan for _ in range(kernel_size//2)]
            ])
        )
        temp_weight = self.pool_list[sorting[branch_num]].weights[0].numpy()
        ec = np.real(eigencentrality(data))
        temp_weight_convolved = np.convolve(ec, kernel, mode='same')
        ax3.plot(
            np.arange(0, len(temp_course)/self.fs, 1./self.fs),
            np.concatenate([
                [np.nan for _ in range(kernel_size//2)],
                sp.stats.zscore(temp_weight_convolved[kernel_size//2:-kernel_size//2]),
                [np.nan for _ in range(kernel_size//2)]
            ])
        )
        # ax3.axes.yaxis.set_visible(False)
        ax3.set_ylim(-3, 5)
        ax3.legend(
            ['Envelope evoked', 'Loss-based estimate', 'Temporal pattern'],
            loc="upper right"
        )

        fig.suptitle(f'Branch {branch_num}', y=0.95, x=0.2, fontsize=30)
        fig.set_size_inches(15, 5)
        self.fake_evoked.plot_topomap(
            times=branch_num,
            axes=ax1,
            colorbar=False,
            scalings=1,
            time_format="",
            outlines='head',
        )

        return fig
