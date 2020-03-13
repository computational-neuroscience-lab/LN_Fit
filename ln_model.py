import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
tfd = tfp.distributions

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Layer
from tensorflow.keras.activations import softplus
from tensorflow.keras import regularizers

import numpy as np


class Normalizer(Layer):
    def __init__(self, **kwargs):
        super(Normalizer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_mu = self.add_variable(name='input_mu', shape=(input_shape[1],), initializer='zeros', trainable=False)
        self.input_std = self.add_variable(name='input_std', shape=(input_shape[1],), initializer='ones', trainable=False)
        super(Normalizer, self).build(input_shape)

    def set(self, tr_inputs):
        mu = np.mean(tr_inputs, 0)
        std = np.std(tr_inputs, 0)
        self.set_weights([mu, std])

    def call(self, inputs, **kwargs):
        return (inputs - self.input_mu) / self.input_std


def prior_trainable(kernel_size, bias_size=1, dtype=None):
  n = kernel_size + bias_size
  return tf.keras.Sequential([
      tfp.layers.VariableLayer(n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t, scale=1),
          reinterpreted_batch_ndims=1)),
  ])

def post_trainable(kernel_size, bias_size=1, dtype=None):
  n = kernel_size + bias_size
  return tf.keras.Sequential([
      tfp.layers.VariableLayer(n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t, scale=1),
          reinterpreted_batch_ndims=1)),
  ])


# MODEL SPECIFIC BUILT METHODS

def build_lnp(n_inputs, l2=0.0):
    inputs = Input(shape=(n_inputs,), name='input')
    normalization = Normalizer(name='normalizer')(inputs)
    potential = Dense(1, use_bias=True, name='filter',
                      kernel_regularizer=regularizers.l2(l2))(normalization)
    firing_rate = Activation(softplus, name='non_linearity')(potential)
    m = Model(inputs=inputs, outputs=firing_rate)
    return m


def build_lnp_bayesian(n_inputs):
    inputs = Input(shape=(n_inputs,), name='input')
    normalization = Normalizer(name='normalizer')(inputs)
    potential = tfp.layers.DenseFlipout(1, name='filter')(normalization)
    firing_rate = Activation(softplus, name='non_linearity')(potential)
    m = Model(inputs=inputs, outputs=firing_rate)
    return m


def build_lnp0(n_inputs, l2=0.0):
    input_bias = keras.layers.Input(shape=(1,), name='input_bias')
    b = keras.layers.Dense(1, use_bias=False, name='bias')(input_bias)

    inputs = Input(shape=(n_inputs,), name='input')
    normalization = Normalizer(name='normalizer')(inputs)
    w = Dense(1, use_bias=False, name='filter', kernel_regularizer=regularizers.l2(l2))(normalization)

    potential = keras.layers.Add()([b, w])
    firing_rate = Activation(softplus, name='non_linearity')(potential)
    m = Model(inputs=[input_bias, inputs], outputs=firing_rate)
    return m


def build_lnp_plus(n_inputs, l2=0.0):
    inputs = Input(shape=(n_inputs,), name='input')
    normalization = Normalizer(name='normalizer')(inputs)
    potential = Dense(1, use_bias=True, name='filter',
                      kernel_regularizer=regularizers.l2(l2),
                      bias_regularizer=regularizers.l2(l2))(normalization)
    firing_rate = Activation(softplus, name='non_linearity')(potential)
    norm_rate = Dense(1, use_bias=False, name='scaling',
                      kernel_regularizer=regularizers.l2(1e-9))(firing_rate)
    lnp = Model(inputs=inputs, outputs=norm_rate)
    return lnp


def upgrade_lnp_plus(lnp):
    input_length = lnp.get_input_shape_at(0)[1]
    norm = lnp.get_layer('normalizer').get_weights()
    weights = lnp.get_layer('filter').get_weights()
    l2 = lnp.get_layer('filter').kernel_regularizer.get_config()['l2']

    lnp_plus = build_lnp_plus(input_length, l2)
    lnp_plus.get_layer('filter').set_weights(weights)
    lnp_plus.get_layer('normalizer').set_weights(norm)
    lnp_plus.get_layer('scaling').set_weights([np.ones([1, 1])])
    return lnp_plus


# GENERAL MODEL BUILT METHODS
# (all LNP_L1 must be listed with a label in the inner map)

def build_model(model_label, n_inputs, l2=0.0):
    label2model = {"LNP": build_lnp,
                   "LNP+": build_lnp_plus,
                   "LNP0": build_lnp0}
    constructor = label2model[model_label]
    return constructor(n_inputs, l2)
