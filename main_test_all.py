import tensorflow_probability as tfp
import tensorflow as tf
import ln_model as lnm
import ln_fit as lnf

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# DATA SET
n_samples = 1000
x0 = np.random.normal(0, 1, [n_samples])
x1 = np.random.normal(1, 10, [n_samples])
x2 = np.random.normal(10, 3, [n_samples])
x = np.stack([x0, x1, x2]).T


# CELL
my_cell = lnm.build_model('LNP', 3)
my_weights = [np.random.uniform(-0.5, 0.5, [3, 1]), np.array([0])]
my_cell.get_layer('filter').set_weights(my_weights)
my_cell.get_layer('normalizer').set(x)
y = my_cell.predict(x, verbose=1)

# MODELS
m = lnm.build_lnp(3, 1e-6)
mb = lnm.build_lnp_bayesian(3)

# FIT
print('training ml model')
lnf.lnp_fit_no_validation(m, x, y)
print('training b model')
lnf.lnp_fit_no_validation(mb, x, y)

print('CELL')
print(my_cell.get_layer('filter').get_weights())

print('ML MODEL')
print(m.get_layer('filter').get_weights())

print('BAYESIAN MODEL')
print(mb.get_layer('filter').get_weights())

sns.distplot(x0)
sns.distplot(x1)
sns.distplot(x2)
plt.title('Data Distribution')
plt.show()



# print('DATA SET')
# print(x.shape)
# print('')
#
# # CELL

#
#
# my_model = ln_model.build_model('LNP', 10)
# ln_fit.lnp_fit_no_validation(my_model, x, y)
#
# print('CELL')
# print(my_cell.get_weights())
#
# print('MODEL')
# print(my_model.get_weights())

# print('normalizer')
# print(m.get_layer('normalizer').get_weights())
#
# m.get_layer('normalizer').set(x1)
#
# print('normalizer 2')
# print(m.get_layer('normalizer').get_weights())
