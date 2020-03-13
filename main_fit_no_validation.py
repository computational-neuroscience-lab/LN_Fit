from data_repository import NeuroRepository
import ln_model
import ln_fit
import ln_repository
import numpy as np
import gc

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# PARAMETERS
dataset_label = "20191011_grid"
session_label = "DHGridBlock"
model_type = "LNP"
model_label = "LNP07"
l2 = 1e-7

data_repo = NeuroRepository(dataset_label, session_label)
cells = np.arange(data_repo.n_cells)
cells_labels = cells + 1                # add one to switch to Matlab Notation

# The dataset should have more or less equal number of 0-spots and 1-spots patterns
n_tiles_0 = data_repo.tr1_len // data_repo.tr0_len

# tile the 0-spots patterns.
tr0_inputs = np.tile(data_repo.tr0_inputs, (n_tiles_0, 1))
tr0_outputs = np.tile(data_repo.tr0_outputs, (n_tiles_0, 1))

# put everything together.
tr_inputs = np.concatenate((tr0_inputs, data_repo.tr1_inputs))
tr_outputs = np.concatenate((tr0_outputs, data_repo.tr1_outputs))

# shuffle
train_permutation = np.random.permutation(tr_inputs.shape[0])
tr_inputs = tr_inputs[train_permutation, :]
tr_outputs = tr_outputs[train_permutation, :]

# FIT
for cell in cells:
    print("Neuron #" + str(cell) + ' of ' + str(data_repo.n_cells))

    print("\t<<< FITTING LNP MODEL... >>>")
    # try:
    #     m = ln_repository.load_ln_model(dataset_label, model_label, session_label+str(cell))
    #     print("\tmodel loaded succesfully")
    #
    # except:
    print("\tmodel did no exist. Creating new one..")
    print("Model Fitting....")
    m = ln_model.build_model(model_type, data_repo.input_len, l2)

    h = ln_fit.lnp_fit_no_validation(m, tr_inputs, tr_outputs[:, cell])
    ln_repository.save_ln_model(m, dataset_label, model_label, session_label+str(cell))

    del m
    gc.collect()

