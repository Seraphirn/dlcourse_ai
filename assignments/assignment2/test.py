
import numpy as np
# import matplotlib.pyplot as plt

from dataset import load_svhn, random_split_train_val
from gradient_check import check_layer_gradient  #\
# , check_layer_param_gradient, check_model_gradient
from layers import FullyConnectedLayer, ReLULayer
# from model import TwoLayerNet
# from trainer import Trainer, Dataset
# from optim import SGD, MomentumSGD
# from metrics import multiclass_accuracy

# def prepare_for_neural_network(train_X, test_X):
#     train_flat = train_X.reshape(train_X.shape[0], -1).astype(float) / 255.0
#     test_flat = test_X.reshape(test_X.shape[0], -1).astype(float) / 255.0

#     # Subtract mean
#     mean_image = np.mean(train_flat, axis = 0)
#     train_flat -= mean_image
#     test_flat -= mean_image

#     return train_flat, test_flat

# train_X, train_y, test_X, test_y = load_svhn("data", max_train=10000, max_test=1000)
# train_X, test_X = prepare_for_neural_network(train_X, test_X)
# # Split train into train and val
# train_X, train_y, val_X, val_y = random_split_train_val(train_X, train_y, num_val = 1000)

X = np.array([[1, -2, 3],
              [-1, 2, 0.1]])

assert check_layer_gradient(ReLULayer(), X)
