import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np

input_bias = tf.keras.layers.Input(shape=(1,), name='ib')
b = tf.keras.layers.Dense(1, use_bias=False, name='b')(input_bias)

inputs = tf.keras.layers.Input(shape=(3,), name='iw')
d = tf.keras.layers.Dense(1, use_bias=False, name='w')(inputs)

output = tf.keras.layers.Add()([b, d])
model = tf.keras.models.Model(inputs=[input_bias, inputs], outputs=output)


x = np.random.rand(10000, 3)
b = np.ones([10000, 1])
y = np.sum(x, 1) + 15

x0 = np.zeros([1000, 3])
b0 = np.ones([1000, 1])
y0 = np.sum(x0, 1) + 15

x2 = np.random.rand(10, 3)
b2 = np.ones([10, 1])
y2 = np.sum(x2, 1) + 15


print("PARAMS")
print("b = " + str(model.get_layer('b').get_weights()))
print("w = " + str(model.get_layer('w').get_weights()))
print("")

model.get_layer('w').trainable = False
model.get_layer('b').trainable = False
model.compile(loss='mse', optimizer='rmsprop')

model.summary()

print("FITTING BIAS...")
model.fit([b0, x0], y0, epochs=50, verbose=0)


print("PARAMS")
print("b = " + str(model.get_layer('b').get_weights()))
print("w = " + str(model.get_layer('w').get_weights()))
print("")

model.get_layer('w').trainable = False
model.get_layer('b').trainable = False
model.compile(loss='mse', optimizer='rmsprop')

model.summary()

print("FITTING KERNEL...")
model.fit([b, x], y, epochs=50, verbose=0)

print("PARAMS")
print("b = " + str(model.get_layer('b').get_weights()))
print("w = " + str(model.get_layer('w').get_weights()))
print("")

print("PREDICTING...")
yy2 = model.predict([b2, x2])


for i in range(10):
    print(str(y2[i]) + " " + str(yy2[i]))


