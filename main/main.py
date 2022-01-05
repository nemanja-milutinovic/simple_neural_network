import numpy as np

from network.Network import Network
from activation_layer.ActivationLayer import ActivationLayer
from fclayer.FCLayer import FCLayer
from Activation import tanh, tanh_prime
from Loss import mse, mse_prime

# training data
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network();
net.add_layer(FCLayer(2, 3))
net.add_layer(ActivationLayer(tanh, tanh_prime))
net.add_layer(FCLayer(3, 1))
net.add_layer(ActivationLayer(tanh, tanh_prime))

# train
net.set_loss(mse, mse_prime)
net.train(x_train, y_train, epochs=4500, learning_rate=0.1)

# test
x_test = np.array([[[1, 0]]])
out = net.predict(x_test)
print(out)