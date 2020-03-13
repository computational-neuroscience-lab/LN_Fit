from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.models import load_model
from ln_fit import pearson_corr
from ln_model import Normalizer
import os.path


storage_folder = "storage"
weight_file_suffix = ".h5"
model_file_suffix = ".json"


def load_ln_weights(model, dataset, type, id):
    weight_storage_path = os.path.join(storage_folder, dataset, type, id + weight_file_suffix)
    model.load_weights(weight_storage_path)


def save_ln_weights(model, dataset, type, id):
    weight_storage_path = os.path.join(storage_folder, dataset, type, id + weight_file_suffix)
    os.makedirs(os.path.dirname(weight_storage_path), exist_ok=True)
    model.save_weights(weight_storage_path)


def load_ln_model(dataset, type, id):
    model_storage_path = os.path.join(storage_folder, dataset, type, id + model_file_suffix)
    with open(model_storage_path, "r") as json_file:
        loaded_model_json = json_file.read()

    with CustomObjectScope({'Normalizer': Normalizer}):
        loaded_model = model_from_json(loaded_model_json)
    load_ln_weights(loaded_model, dataset, type, id)
    return loaded_model


def save_ln_model(model, dataset, type, id):
    model_storage_path = os.path.join(storage_folder, dataset, type, id + model_file_suffix)
    os.makedirs(os.path.dirname(model_storage_path), exist_ok=True)
    model_json = model.to_json()
    with open(model_storage_path, "w") as json_file:
        json_file.write(model_json)
    save_ln_weights(model, dataset, type, id)