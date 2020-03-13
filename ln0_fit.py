from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from ln_model import build_model
import tensorflow as tf
import numpy as np
import math


# LOSS / METRICS METHODS

def pearson_corr(x, y):
    mx, my = K.mean(x), K.mean(y)
    num = K.mean(tf.multiply(x - mx, y - my))
    std_x = K.sqrt(K.mean(K.square(x - mx)))
    std_y = K.sqrt(K.mean(K.square(y - my)))
    den = tf.multiply(std_x, std_y)
    return num / den


# TESTS METHODS

def lnp0_evaluate(model, validation_data):
    ts_inputs, ts_outputs = validation_data
    batch = len(ts_inputs)
    bias_input = np.ones([batch, 1])

    model.compile(loss='poisson', metrics=['mse', pearson_corr], optimizer='rmsprop')
    metrics = model.evaluate([bias_input, ts_inputs], ts_outputs, batch, verbose=0)
    return metrics


def lnp0_predict(model, ts_inputs):
    batch = len(ts_inputs)
    bias_input = np.ones([batch, 1])

    y = model.predict([bias_input, ts_inputs], batch, verbose=0)
    return y


# TRAINING METHODS
def lnp0_fit_bias(model, data):
    tr0_inputs, tr0_outputs = data
    batch = len(tr0_inputs)
    bias_input = np.ones([batch, 1])

    epochs = int(10e3)

    model.get_layer('bias').trainable = True
    model.get_layer('filter').trainable = False
    es_cbk = EarlyStopping(monitor='loss', mode='min', patience=100, restore_best_weights=True)
    model.compile(loss='poisson', metrics=['mse', pearson_corr], optimizer='rmsprop')

    history = model.fit([bias_input, tr0_inputs], tr0_outputs,
                        epochs=epochs,
                        batch_size=batch,
                        callbacks=[es_cbk],
                        verbose=0)
    return history


def lnp0_fit_kernel(model, data):
    tr_inputs, tr_outputs, ts_inputs, ts_outputs = data
    batch = len(tr_inputs)
    bias_input = np.ones(batch)
    bias_input_ts = np.ones(len(ts_inputs))

    epochs = int(10e3)

    model.get_layer('bias').trainable = False
    model.get_layer('filter').trainable = True
    es_cbk = EarlyStopping(monitor='val_mean_squared_error', mode='min', patience=100, restore_best_weights=True)
    model.compile(loss='poisson', metrics=['mse', pearson_corr], optimizer='rmsprop')
    history = model.fit([bias_input, tr_inputs], tr_outputs,
                        epochs=epochs,
                        batch_size=batch,
                        validation_data=([bias_input_ts, ts_inputs], ts_outputs),
                        callbacks=[es_cbk],
                        verbose=0)
    return history


def lnp0_meta_fit(data0, data, model_type):
    input_shape = data[0].shape[1]

    print("creating model...")
    initial_model = build_model(model_type, input_shape)

    b = initial_model.get_layer('bias').get_weights()[0][0]
    w = initial_model.get_layer('filter').get_weights()[0]
    print("kernel = <" + str(w[0]) + ", " + str(w[1]) + ", " + str(w[2]) + "...>")
    print("bias = " + str(b))
    print("")

    print("fitting the bias first...")
    h = lnp0_fit_bias(initial_model, data0)

    b = initial_model.get_layer('bias').get_weights()[0]
    w = initial_model.get_layer('filter').get_weights()[0]
    print("kernel = <" + str(w[0]) + ", " + str(w[1]) + ", " + str(w[2]) + "...>")
    print("bias = " + str(b))
    print("")

    best_err = math.inf
    best_model = initial_model
    initial_weights = initial_model.get_weights()

    print("fitting the kernel...")
    # loop over several l1 values to find the best one
    for l in [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:

        model = build_model(model_type, input_shape, l)
        model.set_weights(initial_weights)

        h = lnp0_fit_kernel(model, data)
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

    b = best_model.get_layer('bias').get_weights()[0]
    w = best_model.get_layer('filter').get_weights()[0]
    print("kernel = <" + str(w[0]) + ", " + str(w[1]) + ", " + str(w[2]) + "...>")
    print("bias = " + str(b))
    print("")

    return best_model
