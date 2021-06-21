import numpy as np
import matplotlib.pyplot as plt

from dataset import load_svhn, random_split_train_val
from gradient_check import check_layer_gradient, check_layer_param_gradient, check_model_gradient
from layers import FullyConnectedLayer, ReLULayer, ConvolutionalLayer, MaxPoolingLayer, Flattener
from model import ConvNet
from trainer import Trainer, Dataset
from optim import SGD, MomentumSGD
from metrics import multiclass_accuracy

def prepare_for_neural_network(train_X, test_X):
    train_X = train_X.astype(float) / 255.0
    test_X = test_X.astype(float) / 255.0

    # Subtract mean
    mean_image = np.mean(train_X, axis = 0)
    train_X -= mean_image
    test_X -= mean_image

    return train_X, test_X

train_X, train_y, test_X, test_y = load_svhn("data", max_train=10000,
                                             max_test=1000)
train_X, test_X = prepare_for_neural_network(train_X, test_X)
# Split train into train and val
train_X, train_y, val_X, val_y = random_split_train_val(train_X, train_y,
                                                        num_val = 1000)

# TODO: In model.py, implement missed functions function for ConvNet model

# No need to use L2 regularization
# model = ConvNet(input_shape=(32, 32, 3), n_output_classes=10, conv1_channels=2, conv2_channels=2)
# loss = model.compute_loss_and_gradients(train_X[:2], train_y[:2])

# # TODO Now implement backward pass and aggregate all of the params
# check_model_gradient(model, train_X[:2], train_y[:2])
model = ConvNet(input_shape=(32,32,3), n_output_classes=10, conv1_channels=2, conv2_channels=2)
dataset = Dataset(train_X[:16], train_y[:16], val_X[:16], val_y[:16])
trainer = Trainer(model, dataset, SGD(), batch_size=16, learning_rate=1e-3)

loss_history, train_history, val_history = trainer.fit()
