from data_repository import NeuroRepository
import ln_fit
import ln_repository
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# PARAMETERS
dataset_label = "20200131_dh"
session_label = "DHMulti"
model_type = "LNP"
model_label = "LNP"

data_repo = NeuroRepository(dataset_label, session_label)
cells = np.arange(data_repo.n_cells)

# The dataset should have more or less equal number of 0-spots and 1-spots patterns
n_tiles_0 = data_repo.tr1_len // data_repo.tr0_len

# tile the 0-spots patterns.
tr0_inputs = np.tile(data_repo.tr0_inputs, (n_tiles_0, 1))
tr0_outputs = np.tile(data_repo.tr0_outputs, (n_tiles_0, 1))

# put everything together.
tr_inputs = np.concatenate((tr0_inputs, data_repo.tr1_inputs, data_repo.trN_inputs))
tr_outputs = np.concatenate((tr0_outputs, data_repo.tr1_outputs, data_repo.trN_outputs))

# shuffle
train_permutation = np.random.permutation(tr_inputs.shape[0])
tr_inputs = tr_inputs[train_permutation, :]
tr_outputs = tr_outputs[train_permutation, :]

# FIT
for cell in cells:
    print("Neuron #" + str(cell))

    print("\t<<< FITTING LNP MODEL... >>>")
    try:
        m = ln_repository.load_ln_model(dataset_label, model_label, session_label+str(cell))
        print("\tmodel loaded succesfully")
        # print("Model Fitting....")
        # m = ln_fit.lnp_meta_fit(data_repo.get_data(cell), model_type, m)

    except:
        print("\tmodel did no exist. Creating new one..")
        print("Model Fitting....")
        m = ln_fit.lnp_meta_fit(data_repo.tr_inputs, data_repo.tr_outputs[:, cell], data_repo.ts_inputs, data_repo.ts_outputs[:, cell], model_type)

    ln_repository.save_ln_model(m, dataset_label, model_label, session_label+str(cell))
    print("Done.\n")
