from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import tensorflow.keras as keras
from ln_model import build_model
import tensorflow as tf
import numpy as np
import math
import datetime


# LOSS / METRICS METHODS

def pearson_corr(x, y):
    mx, my = K.mean(x), K.mean(y)
    num = K.mean(tf.multiply(x - mx, y - my))
    std_x = K.sqrt(K.mean(K.square(x - mx)))
    std_y = K.sqrt(K.mean(K.square(y - my)))
    den = tf.multiply(std_x, std_y)
    return num / den


# TESTS METHODS

def lnp_evaluate(model, ts_inputs, ts_outputs):
    batch = len(ts_inputs)
    model.compile(loss='poisson', metrics=['mse', pearson_corr], optimizer='rmsprop')
    metrics = model.evaluate(ts_inputs, ts_outputs, batch, verbose=0)
    return metrics


def lnp_predict(model, ts_inputs):
    batch = len(ts_inputs)
    y = model.predict(ts_inputs, batch, verbose=0)
    return y


# TRAINING METHODS

def lnp_fit(model, tr_inputs, tr_outputs, ts_inputs, ts_outputs):
    batch = len(tr_inputs)
    epochs = int(10e3)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    es_cbk = EarlyStopping(monitor='val_mean_squared_error', mode='min', patience=100, restore_best_weights=True)

    model.compile(loss='poisson', metrics=['mse', pearson_corr], optimizer=keras.optimizers.SGD(lr=0.01, nesterov=True))
    model.get_layer('normalizer').set(tr_inputs)
    history = model.fit(tr_inputs, tr_outputs,
                        epochs=epochs,
                        batch_size=batch,
                        validation_data=(ts_inputs, ts_outputs),
                        callbacks=[es_cbk],
                        verbose=0)
    return history


def lnp_fit_no_validation(model, tr_inputs, tr_outputs):
    batch = len(tr_inputs)
    epochs = int(10e3)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    es_cbk = EarlyStopping(monitor='loss', mode='min', patience=100, restore_best_weights=True)

    model.compile(loss='poisson', metrics=['mse', pearson_corr], optimizer=tf.keras.optimizers.SGD(lr=0.01, nesterov=True))
    model.get_layer('normalizer').set(tr_inputs)
    history = model.fit(tr_inputs, tr_outputs,
                        epochs=epochs,
                        batch_size=batch,
                        callbacks=[es_cbk],
                        verbose=1)
    return history


def lnp_meta_fit(tr_inputs, tr_outputs, ts_inputs, ts_outputs, model_type, initial_model=None):
    input_shape = tr_inputs.shape[1]

    if initial_model is None:
        initial_model = build_model(model_type, input_shape)
    best_err = math.inf
    best_model = initial_model
    initial_weights = initial_model.get_weights()

    # loop over several l1 values to find the best one
    for l in [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:

        model = build_model(model_type, input_shape, l)
        model.set_weights(initial_weights)
        h = lnp_fit(model, tr_inputs, tr_outputs, ts_inputs, ts_outputs)
        initial_weights = model.get_weights()

        err = h.history['val_mean_squared_error'][-1]
        acc = h.history['val_pearson_corr'][-1]
        loss = h.history['loss'][-1]
        print("l2=" + str(l) + "\t\t==>\t\tloss = " + str(loss) + ",\tvalidation mse = " + str(
            err) + ",\taccuracy = " + str(acc))

        if err < best_err:
            best_err = err
            best_model = model

    best_l = best_model.get_layer('filter').kernel_regularizer.get_config()['l2']
    print("SELECTED l1=" + str(best_l))
    return best_model