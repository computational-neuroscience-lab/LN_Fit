from data_repository import NeuroRepository
import ln_repository as ln_repo
import ln_fit as ln

import scipy.io as spio
import numpy as np
from pathlib import Path
import gc

# PARAMETERS
dataset_label = "20200131_dh"
session_label = "DHMulti"
model_label = "LNP"

# PATHS
output_dir = "outputs/" + dataset_label + '/' + session_label
output_file = output_dir + '/' + model_label + '.mat'
Path(output_dir).mkdir(parents=True, exist_ok=True)

# EVALUATION
data_repo = NeuroRepository(dataset_label, session_label)
cells = np.arange(data_repo.n_cells)    # Python Notation np.array([17, 61, 83, 85, 45, 51, 56, 60, 64, 100, 101, 133, 137]) - 1 #
cells_labels = cells + 1                # add one to switch to Matlab Notation
n_cells = len(cells)

w_array = np.zeros([n_cells, data_repo.input_len])
b_array = np.zeros(n_cells)
c_array = np.zeros(n_cells)
mu_array = np.zeros([n_cells, data_repo.input_len])
std_array = np.zeros([n_cells, data_repo.input_len])
mse_array = np.zeros(n_cells)
acc_array = np.zeros(n_cells)

predictions_array = np.zeros([n_cells, data_repo.ts_len])
(ts_in, ts_out) = data_repo.get_testing_data()

for i_cell in range(len(cells)):
    cell = cells[i_cell]
    cell_label = str(cells_labels[i_cell])
    print("Neuron #" + cell_label)
    # try:
    m = ln_repo.load_ln_model(dataset_label, model_label, session_label+str(cell))
    print("\tevaluating model...")
    # except Exception:
    #     print("\tmodel did no exist..")
    #     continue

    metrics = ln.lnp_evaluate(m, ts_in, ts_out[:, cell])
    predictions = ln.lnp_predict(m, ts_in)

    for value, label in zip(metrics, m.metrics_names):
        print("\t" + label + ": " + str(value))

    w_array[i_cell] = m.get_layer('filter').get_weights()[0].T
    b_array[i_cell] = m.get_layer('filter').get_weights()[1][0]
    mu_array[i_cell] = m.get_layer('filter').get_weights()[0].T
    std_array[i_cell] = m.get_layer('filter').get_weights()[1].T

    try:
        c_array[i_cell] = m.get_layer('scaling').get_weights()[0].T
    except:
        pass

    mse_array[i_cell] = metrics[1]
    acc_array[i_cell] = metrics[2]
    predictions_array[i_cell, :] = predictions.T

    del m
    del metrics
    del predictions
    gc.collect()

# save everything
spio.savemat(output_file, {"cells": cells_labels,
                           "ws": w_array,
                           "b": b_array,
                           "c": c_array,
                           "mu": mu_array,
                           "std": std_array,
                           "mse": mse_array,
                           "accuracy": acc_array,
                           "predictions": predictions_array,
                           "truths": ts_out[:, cells].T})
