from __future__ import print_function
import scipy.io as spio
import numpy as np
import os

DATASETS_PATH = "/storage/Francesco/Datasets/"


class NeuroRepository:

    def __init__(self, dataset_label, session_label):
        self.__tr0_in, self.__tr0_out, self.__tr1_in, self.__tr1_out, self.__trN_in, self.__trN_out, self.__ts_in, self.__ts_out = load_dataset(dataset_label, session_label)
        self.__tr_in = np.concatenate([self.__tr0_in,  self.__tr1_in,  self.__trN_in])
        self.__tr_out = np.concatenate([self.__tr0_out,  self.__tr1_out,  self.__trN_out])

        # Shuffle the Training Set
        train_permutation = np.random.permutation(self.__tr_in.shape[0])
        self.__tr_in = self.__tr_in[train_permutation, :]
        self.__tr_out = self.__tr_out[train_permutation, :]

    def get_data(self, cell_idx=None):
        if cell_idx is None:
            cell_idx = range(self.n_cells)
        return self.__tr_in, self.__tr_out[:, cell_idx], self.__ts_in, self.__ts_out[:, cell_idx]

    def get_testing_data(self, cell_idx=None):
        if cell_idx is None:
            cell_idx = range(self.n_cells)
        return self.__ts_in, self.__ts_out[:, cell_idx]

    def get_training_data(self, cell_idx=None):
        if cell_idx is None:
            cell_idx = range(self.n_cells)
        return self.__tr_in, self.__tr_out[:, cell_idx]

    def cut_spots(self, n_spots):
        # Cut first the spots that are not in the testing set (or the ones featuring the least)
        spots_sorted_by_occurrence = np.argsort(sum(self.__ts_in.astype(bool)))
        spots_to_cut = spots_sorted_by_occurrence[:n_spots]
        spots_to_keep = spots_sorted_by_occurrence[n_spots:]

        # For each spot we want to cut, check in which patterns they appear
        all_ts_samples_to_cut = []
        all_tr1_samples_to_cut = []
        all_trN_samples_to_cut = []

        for spot in spots_to_cut:
            ts_samples_to_cut = np.flatnonzero(self.__ts_in.astype(bool)[:, spot])
            tr1_samples_to_cut = np.flatnonzero(self.__tr1_in.astype(bool)[:, spot])
            trN_samples_to_cut = np.flatnonzero(self.__trN_in.astype(bool)[:, spot])

            all_ts_samples_to_cut.append(ts_samples_to_cut)
            all_tr1_samples_to_cut.append(tr1_samples_to_cut)
            all_trN_samples_to_cut.append(trN_samples_to_cut)

        all_ts_samples_to_cut = np.unique(np.concatenate(all_ts_samples_to_cut))
        all_tr1_samples_to_cut = np.unique(np.concatenate(all_tr1_samples_to_cut))
        all_trN_samples_to_cut = np.unique(np.concatenate(all_trN_samples_to_cut))

        # Remove the samples with the spots to cut
        self.__ts_in = np.delete(self.__ts_in, all_ts_samples_to_cut, 0)
        self.__ts_out = np.delete(self.__ts_out, all_ts_samples_to_cut, 0)

        self.__tr1_in = np.delete(self.__tr_in, all_tr1_samples_to_cut, 0)
        self.__tr1_out = np.delete(self.__tr_out, all_tr1_samples_to_cut, 0)

        self.__trN_in = np.delete(self.__tr_in, all_trN_samples_to_cut, 0)
        self.__trN_out = np.delete(self.__tr_out, all_trN_samples_to_cut, 0)

        # Remove the spots to cut
        self.__tr0_in = np.delete(self.__tr0_in, spots_to_cut, 1)
        self.__tr1_in = np.delete(self.__tr1_in, spots_to_cut, 1)
        self.__trN_in = np.delete(self.__trN_in, spots_to_cut, 1)
        self.__ts_in = np.delete(self.__ts_in, spots_to_cut, 1)

        # Reassemble the total training set
        self.__tr_in = np.concatenate([self.__tr0_in,  self.__tr1_in,  self.__trN_in])
        self.__tr_out = np.concatenate([self.__tr0_out,  self.__tr1_out,  self.__trN_out])

        # Shuffle the Training Set
        train_permutation = np.random.permutation(self.__tr_in.shape[0])
        self.__tr_in = self.__tr_in[train_permutation, :]
        self.__tr_out = self.__tr_out[train_permutation, :]

        return spots_to_keep

    def cut_training_samples(self, n_training_samples):
        all_tr_samples_to_cut = range(self.__tr_in.shape[0] - n_training_samples, self.__tr_in.shape[0])
        self.__tr_in = np.delete(self.__tr_in, all_tr_samples_to_cut, 0)
        self.__tr_out = np.delete(self.__tr_out, all_tr_samples_to_cut, 0)

    # DATA-SET STRUCTURE
    # tr_inputs = input intensity matrix            [n_patterns * len_pattern]
    # tr_outputs = output mean-spike-counts matrix  [n_patterns * n_neurons]
    # ts_inputs = input intensity matrix            [n_patterns * len_pattern]
    # ts_outputs = output mean-spike-counts matrix  [n_patterns * n_neurons]

    @property
    def tr_inputs(self):
        return self.__tr_in

    @property
    def tr_outputs(self):
        return self.__tr_out

    @property
    def tr_len(self):
        return self.__tr_in.shape[0]

    @property
    def tr0_inputs(self):
        return self.__tr0_in

    @property
    def tr0_outputs(self):
        return self.__tr0_out

    @property
    def tr0_len(self):
        return self.__tr0_in.shape[0]

    @property
    def tr1_inputs(self):
        return self.__tr1_in

    @property
    def tr1_outputs(self):
        return self.__tr1_out

    @property
    def tr1_len(self):
        return self.__tr1_in.shape[0]

    @property
    def trN_inputs(self):
        return self.__tr1_in

    @property
    def trN_outputs(self):
        return self.__tr1_out

    @property
    def trN_len(self):
        return self.__trN_in.shape[0]

    @property
    def ts_inputs(self):
        return self.__ts_in

    @property
    def ts_outputs(self):
        return self.__ts_out

    @property
    def ts_len(self):
        return self.__ts_in.shape[0]

    @property
    def n_cells(self):
        return self.__tr_out.shape[1]

    @property
    def input_len(self):
        return self.__tr_in.shape[1]


def load_dataset(dataset_label, session_label):
    try:
        ts_inputs = np.load('datasets/' + dataset_label + '_' + session_label + '/ts_inputs.npy')
        ts_outputs = np.load('datasets/' + dataset_label + '_' + session_label + '/ts_outputs.npy')

        tr0_inputs = np.load('datasets/' + dataset_label + '_' + session_label + '/tr0_inputs.npy')
        tr0_outputs = np.load('datasets/' + dataset_label + '_' + session_label + '/tr0_outputs.npy')

        tr1_inputs = np.load('datasets/' + dataset_label + '_' + session_label + '/tr1_inputs.npy')
        tr1_outputs = np.load('datasets/' + dataset_label + '_' + session_label + '/tr1_outputs.npy')

        trN_inputs = np.load('datasets/' + dataset_label + '_' + session_label + '/trN_inputs.npy')
        trN_outputs = np.load('datasets/' + dataset_label + '_' + session_label + '/trN_outputs.npy')

    except:
        extract_dataset(dataset_label, session_label)

        ts_inputs = np.load('datasets/' + dataset_label + '_' + session_label + '/ts_inputs.npy')
        ts_outputs = np.load('datasets/' + dataset_label + '_' + session_label + '/ts_outputs.npy')

        tr0_inputs = np.load('datasets/' + dataset_label + '_' + session_label + '/tr0_inputs.npy')
        tr0_outputs = np.load('datasets/' + dataset_label + '_' + session_label + '/tr0_outputs.npy')

        tr1_inputs = np.load('datasets/' + dataset_label + '_' + session_label + '/tr1_inputs.npy')
        tr1_outputs = np.load('datasets/' + dataset_label + '_' + session_label + '/tr1_outputs.npy')

        trN_inputs = np.load('datasets/' + dataset_label + '_' + session_label + '/trN_inputs.npy')
        trN_outputs = np.load('datasets/' + dataset_label + '_' + session_label + '/trN_outputs.npy')

    return tr0_inputs, tr0_outputs, tr1_inputs, tr1_outputs, trN_inputs, trN_outputs, ts_inputs, ts_outputs


def extract_dataset(dataset_label, session_label):
    # Unwrap Matlab structs
    # [STIMTYPE]_stims  = input intensity matrix        [n_patterns * len_pattern]
    # [STIMTYPE]_spikes = output spike-count matrix     [n_neurons * n_patterns] [n_repetitions]
    # [STIMTYPE]_frs    = output spike-counts matrix    [n_neurons * n_patterns]

    data = spio.loadmat(DATASETS_PATH + dataset_label + "Matrix.mat")

    zero_stims = data[session_label][0, 0]["stimuli"][0, 0]["zero"]
    zero_spikes = data[session_label][0, 0]["responses"][0, 0]["zero"][0, 0]["spikeCounts"]

    single_stims = data[session_label][0, 0]["stimuli"][0, 0]["single"]
    singles_spikes = data[session_label][0, 0]["responses"][0, 0]["single"][0, 0]["spikeCounts"]

    multi_stims = data[session_label][0, 0]["stimuli"][0, 0]["multi"]
    multi_spikes = data[session_label][0, 0]["responses"][0, 0]["multi"][0, 0]["spikeCounts"]

    test_stims = data[session_label][0, 0]["stimuli"][0, 0]["test"]
    test_spikes = data[session_label][0, 0]["responses"][0, 0]["test"][0, 0]["spikeCounts"]
    test_frs = data[session_label][0, 0]["responses"][0, 0]["test"][0, 0]["firingRates"]

    n_cells = multi_spikes.shape[0]
    n_inputs = multi_stims.shape[1]

    # For the testing set we just need the  mean spike counts corresponding to each patterns
    # ts_inputs = input intensity matrix            [n_patterns * len_pattern]
    # ts_outputs = output mean-spike-counts matrix  [n_patterns * n_neurons]
    ts_inputs = test_stims
    ts_outputs = test_frs.T

    # We want to fit LNP with all the repetitions independently (no mean spike counts)
    # So we unwrap the training repetitions to create 1-to-1 input-output samples

    # tr_inputs = input intensity matrix            [n_patterns * len_pattern]
    # tr_outputs = output mean-spike-counts matrix  [n_patterns * n_neurons]
    n_tr0_patterns = zero_stims.shape[0]
    if n_tr0_patterns > 0:

        tr0_inputs = []
        tr0_outputs = []
        for i_pattern in range(n_tr0_patterns):
            pattern = zero_stims[i_pattern, :]
            responses = np.stack(zero_spikes[:, i_pattern], axis=2)
            for repetition in responses[0, :, :]:
                tr0_inputs.append(pattern)
                tr0_outputs.append(repetition)
        tr0_inputs = np.stack(tr0_inputs).astype('float32')
        tr0_outputs = np.stack(tr0_outputs).astype('float32')
    else:
        tr0_inputs = np.ndarray([0, n_inputs])
        tr0_outputs = np.ndarray([0, n_cells])

    n_tr1_patterns = single_stims.shape[0]
    if n_tr1_patterns > 0:
        tr1_inputs = []
        tr1_outputs = []
        for i_pattern in range(n_tr1_patterns):
            pattern = single_stims[i_pattern, :]
            responses = np.stack(singles_spikes[:, i_pattern], axis=2)
            for repetition in responses[0, :, :]:
                tr1_inputs.append(pattern)
                tr1_outputs.append(repetition)
        tr1_inputs = np.stack(tr1_inputs).astype('float32')
        tr1_outputs = np.stack(tr1_outputs).astype('float32')
    else:
        tr1_inputs = np.ndarray([0, n_inputs])
        tr1_outputs = np.ndarray([0, n_cells])

    n_trN_patterns = multi_stims.shape[0]
    if n_trN_patterns > 0:
        trN_inputs = []
        trN_outputs = []
        for i_pattern in range(n_trN_patterns):
            pattern = multi_stims[i_pattern, :]
            responses = np.stack(multi_spikes[:, i_pattern], axis=2)
            for repetition in responses[0, :, :]:
                trN_inputs.append(pattern)
                trN_outputs.append(repetition)
        trN_inputs = np.stack(trN_inputs).astype('float32')
        trN_outputs = np.stack(trN_outputs).astype('float32')
    else:
        trN_inputs = np.ndarray([0, n_inputs])
        trN_outputs = np.ndarray([0, n_cells])

    # Shuffle the Training Set
    train_permutation = np.random.permutation(tr0_inputs.shape[0])
    tr0_inputs = tr0_inputs[train_permutation, :]
    tr0_outputs = tr0_outputs[train_permutation, :]

    train_permutation = np.random.permutation(tr1_inputs.shape[0])
    tr1_inputs = tr1_inputs[train_permutation, :]
    tr1_outputs = tr1_outputs[train_permutation, :]

    train_permutation = np.random.permutation(trN_inputs.shape[0])
    trN_inputs = trN_inputs[train_permutation, :]
    trN_outputs = trN_outputs[train_permutation, :]

    # save everything
    if not os.path.exists('datasets/' + dataset_label + '_' + session_label + '/'):
        os.makedirs('datasets/' + dataset_label + '_' + session_label + '/')

    np.save('datasets/' + dataset_label + '_' + session_label + '/ts_inputs.npy', ts_inputs)
    np.save('datasets/' + dataset_label + '_' + session_label + '/ts_outputs.npy', ts_outputs)

    np.save('datasets/' + dataset_label + '_' + session_label + '/tr0_inputs.npy', tr0_inputs)
    np.save('datasets/' + dataset_label + '_' + session_label + '/tr0_outputs.npy', tr0_outputs)

    np.save('datasets/' + dataset_label + '_' + session_label + '/tr1_inputs.npy', tr1_inputs)
    np.save('datasets/' + dataset_label + '_' + session_label + '/tr1_outputs.npy', tr1_outputs)

    np.save('datasets/' + dataset_label + '_' + session_label + '/trN_inputs.npy', trN_inputs)
    np.save('datasets/' + dataset_label + '_' + session_label + '/trN_outputs.npy', trN_outputs)
