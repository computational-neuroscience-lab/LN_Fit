from data_repository import NeuroRepository
import ln_fit
import ln_repository

import scipy.io as spio
import numpy as np
import os

# PARAMETERS
data_label = "20190821"
model_type = "LNP"
spots_cuts = [30, 40, 50, 60, 70]
cell_labels = [20, 36, 41, 43, 46, 57, 64, 81]  # Matlab Notation
cells = np.array(cell_labels) - 1  # switch to Python Notation
n_cells = len(cells)

# PATHS
output_path = '/storage/Francesco/outputs/' + data_label + '/'

# FIT
for cut_size in spots_cuts:

    data_repo = NeuroRepository(data_label)
    spots_kept = np.array(data_repo.cut_spots(cut_size))

    w_array = np.zeros([n_cells, data_repo.input_len])
    b_array = np.zeros(n_cells)
    mse_array = np.zeros(n_cells)
    acc_array = np.zeros(n_cells)
    predictions_array = np.zeros([n_cells, data_repo.ts_len])

    model_label = model_type + "_" + str(data_repo.input_len) + "spots"
    output_file = output_path + 'performances_' + model_label + '.mat'

    print(str(cut_size) + " spots cut.")
    print("current number of spots: " + str(data_repo.input_len))
    print("current number of training samples: " + str(data_repo.tr_len))
    print("current number of testing samples: " + str(data_repo.ts_len))
    print("")

    for i_cell in range(len(cells)):
        cell = cells[i_cell]

        print("Neuron #" + str(cell))

        print("\t<<< FITTING LNP MODEL... >>>")
        m = ln_fit.lnp_meta_fit(data_repo.get_data(cell), model_type)
        ln_repository.save_ln_model(m, data_label, model_label, str(cell))
        print("Done.\n")

        metrics = ln_fit.lnp_evaluate(m, data_repo.get_testing_data(cell))
        predictions = ln_fit.lnp_predict(m, data_repo.ts_inputs)

        for value, label in zip(metrics, m.metrics_names):
            print("\t" + label + ": " + str(value))

        w_array[i_cell] = m.get_layer('filter').get_weights()[0].T
        b_array[i_cell] = m.get_layer('filter').get_weights()[1][0]

        mse_array[i_cell] = metrics[1]
        acc_array[i_cell] = metrics[2]
        predictions_array[i_cell, :] = predictions.T

    # save everything
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    spio.savemat(output_file, {"cells": cell_labels,
                               "spots": spots_kept + 1,  # Matlab Notation
                               "tr_len": data_repo.tr_len,
                               "ws": w_array,
                               "b": b_array,
                               "mse": mse_array,
                               "accuracy": acc_array,
                               "predictions": predictions_array,
                               "truths": data_repo.ts_outputs[:, cells].T})
