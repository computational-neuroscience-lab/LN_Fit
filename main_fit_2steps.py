from data_repository import NeuroRepository
import ln0_fit
import ln_repository
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# PARAMETERS
dataset_label = "20170614"
model_type = "LNP0"
model_label = "LNP_bias"
train_only_on_singles = False
cell_labels = np.arange(1, 46)  # Matlab Notation


# FIT
cells = np.array(cell_labels) - 1  # subtract one to switch to Python Notation
data_repo = NeuroRepository(dataset_label, train_only_on_singles)  # True for only single spots

for cell in cells:
    print("Neuron #" + str(cell))

    print("\t<<< FITTING LNP MODEL... >>>")
    m = ln0_fit.lnp0_meta_fit(data_repo.get_baseline_data(cell), data_repo.get_data(cell), model_type)

    ln_repository.save_ln_model(m, dataset_label, model_label, str(cell))
    print("Done.\n")
